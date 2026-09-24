from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from datetime import datetime, timezone
from threading import Event
from threading import RLock

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from model.navigation.geodata import GlobalGeography, get_global_geography
from model.rl.navigation_env import OceanNavigationEnv

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'data' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / 'navruna_ppo.pt'
META_PATH = MODEL_DIR / 'navruna_ppo.json'
HISTORY_PATH = MODEL_DIR / 'training_history.json'


class PolicyNet(nn.Module):
    def __init__(self, obs_size=25, action_size=7):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(obs_size,256),nn.LayerNorm(256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,128),nn.Tanh())
        self.policy=nn.Linear(128,action_size)
        self.value=nn.Linear(128,1)
    def forward(self,x):
        z=self.body(x); return self.policy(z),self.value(z).squeeze(-1)


@dataclass
class TrainReport:
    device: str
    updates: int
    transitions: int
    episodes: int
    success_rate: float
    mean_return: float
    elapsed_s: float
    model_path: str


class PPOTrainer:
    def __init__(self, seed=20260922):
        torch.manual_seed(seed); np.random.seed(seed)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy=PolicyNet().to(self.device)
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=3e-4)
        self.total_updates=0
        self.episodes=0
        self.trained=MODEL_PATH.exists()
        self.training_status='idle'
        self.training_history=[]
        self._lock=RLock()
        self._training_lock=RLock()
        self.stop_requested=Event()
        self.activity_log=[]
        if HISTORY_PATH.exists():
            try: self.training_history=json.loads(HISTORY_PATH.read_text(encoding='utf-8'))[-2000:]
            except Exception: self.training_history=[]
        self.last_report={"device":str(self.device),"updates":0,"episodes":0,"success_rate":0.0,"mean_return":0.0,"model_path":str(MODEL_PATH)}
        if MODEL_PATH.exists():
            try: self.policy.load_state_dict(torch.load(MODEL_PATH,map_location=self.device,weights_only=True))
            except Exception: self.trained=False

    def train(self, total_steps=20000, n_envs=64, rollout=128, lr=3e-4, continuous=True):
        with self._training_lock:
            return self._train_locked(total_steps, n_envs, rollout, lr, continuous)

    def _train_locked(self, total_steps=20000, n_envs=64, rollout=128, lr=3e-4, continuous=True):
        self.stop_requested.clear()
        start=time.time(); self.training_status='training'; self._log('training_started',f"PPO {'continuous ' if continuous else ''}run started: {total_steps:,} steps per update target across {n_envs} environments"); self.optimizer.param_groups[0]['lr']=lr
        geo=get_global_geography(); envs=[OceanNavigationEnv(seed=1000+i,geography=geo) for i in range(n_envs)]
        curriculum_max=2600
        obs=np.stack([e.reset(min_route_nm=300,max_route_nm=curriculum_max) for e in envs])
        returns=[]; successes=0; finished=0; updates=0
        while not self.stop_requested.is_set() and (continuous or updates*rollout*n_envs < total_steps):
            O=[]; A=[]; LP=[]; V=[]; R=[]; D=[]; MASK=[]
            # Curriculum: learn point-to-point control in open water first, then enable the hard land constraint.
            curriculum_progress = (updates*rollout*n_envs) / max(1,total_steps)
            for e in envs: e.land_enabled = curriculum_progress >= 0.35
            for _ in range(rollout):
                ot=torch.tensor(obs,dtype=torch.float32,device=self.device)
                masks=np.stack([e.action_mask_fast(5.0) for e in envs])
                mask_t=torch.tensor(masks,device=self.device)
                with torch.no_grad():
                    logits,vals=self.policy(ot); logits=logits.masked_fill(~mask_t,-1e9); dist=Categorical(logits=logits); act=dist.sample(); lp=dist.log_prob(act)
                actions=act.detach().cpu().numpy();
                nxt=np.empty_like(obs); rr=np.zeros(n_envs,np.float32); dd=np.zeros(n_envs,np.float32)
                for i,e in enumerate(envs):
                    no,r,d,info=e.step(int(actions[i]));
                    if d:
                        finished+=1; successes += int(info.get('success',False)); returns.append(e.total_reward)
                        curriculum_max=min(9000, curriculum_max + 60)
                        no=e.reset(min_route_nm=300,max_route_nm=curriculum_max)
                    nxt[i]=no; rr[i]=r; dd[i]=float(d)
                O.append(obs.copy()); A.append(actions); LP.append(lp.detach().cpu().numpy()); V.append(vals.detach().cpu().numpy()); R.append(rr); D.append(dd); MASK.append(masks); obs=nxt
            with torch.no_grad():
                _,next_v=self.policy(torch.tensor(obs,dtype=torch.float32,device=self.device)); next_v=next_v.detach().cpu().numpy()
            O=np.asarray(O); A=np.asarray(A); LP=np.asarray(LP); V=np.asarray(V); R=np.asarray(R); D=np.asarray(D); MASK=np.asarray(MASK)
            adv=np.zeros_like(R); gae=np.zeros(n_envs,np.float32)
            for t in reversed(range(rollout)):
                nv=next_v if t==rollout-1 else V[t+1]
                delta=R[t]+0.99*nv*(1-D[t])-V[t]
                gae=delta+0.95*0.99*(1-D[t])*gae; adv[t]=gae
            ret=adv+V
            flat=lambda x: torch.tensor(x.reshape(-1),dtype=torch.float32,device=self.device)
            obs_t=torch.tensor(O.reshape(-1,O.shape[-1]),dtype=torch.float32,device=self.device)
            mask_t=torch.tensor(MASK.reshape(-1,MASK.shape[-1]),dtype=torch.bool,device=self.device)
            act_t=torch.tensor(A.reshape(-1),dtype=torch.long,device=self.device)
            oldlp_t=flat(LP); adv_t=flat(adv); ret_t=flat(ret)
            adv_t=(adv_t-adv_t.mean())/(adv_t.std()+1e-8)
            for _epoch in range(4):
                idx=np.random.permutation(obs_t.shape[0])
                for s in range(0,len(idx),2048):
                    ids=torch.tensor(idx[s:s+2048],device=self.device)
                    logits,val=self.policy(obs_t[ids]); logits=logits.masked_fill(~mask_t[ids],-1e9); dist=Categorical(logits=logits); logp=dist.log_prob(act_t[ids]); entropy=dist.entropy().mean()
                    ratio=torch.exp(logp-oldlp_t[ids]); a=adv_t[ids]
                    loss_pi=-torch.min(ratio*a,torch.clamp(ratio,.8,1.2)*a).mean()
                    loss_v=0.5*(val-ret_t[ids]).pow(2).mean(); loss=loss_pi+loss_v-0.01*entropy
                    approx_kl=(oldlp_t[ids]-logp).mean().detach()
                    self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.policy.parameters(),0.7); self.optimizer.step()
            updates+=1
            recent=returns[-100:]
            elapsed=time.time()-start
            point={"update": self.total_updates+updates, "steps": updates*rollout*n_envs, "steps_per_sec": (updates*rollout*n_envs)/max(1e-6,elapsed), "episodes": finished, "success_rate": successes/max(1,finished), "mean_return": float(np.mean(recent)) if recent else 0.0, "loss_policy": float(loss_pi.detach().cpu()), "loss_value": float(loss_v.detach().cpu()), "entropy": float(entropy.detach().cpu()), "approx_kl": float(approx_kl.cpu()), "learning_rate": lr, "elapsed_s": time.time()-start}
            self.training_history.append(point); self.training_history=self.training_history[-2000:]
            self._log('update',f"Update {point['update']}: {point['steps']:,} steps, {point['steps_per_sec']:.0f} steps/s, success {point['success_rate']*100:.1f}%")
            HISTORY_PATH.write_text(json.dumps(self.training_history,indent=2),encoding='utf-8')
            time.sleep(0)
        torch.save(self.policy.state_dict(),MODEL_PATH)
        self.trained=True
        rate=successes/max(1,finished)
        report=TrainReport(str(self.device),updates,updates*rollout*n_envs,finished,rate,float(np.mean(returns[-500:]) if returns else 0),time.time()-start,str(MODEL_PATH))
        self.total_updates+=updates; self.episodes+=finished
        self.last_report={**report.__dict__,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        self.training_status='idle'
        self._log('training_stopped' if self.stop_requested.is_set() else 'training_complete',f"Training {'stopped' if self.stop_requested.is_set() else 'complete'}: {report.transitions:,} transitions, {report.success_rate*100:.1f}% success")
        META_PATH.write_text(json.dumps(self.last_report,indent=2),encoding='utf8')
        return self.last_report

    def request_stop(self):
        self.stop_requested.set()
        self._log('stop_requested','Training stop requested by the operator')

    def _log(self, kind, message):
        self.activity_log.append({"time":datetime.now(timezone.utc).isoformat(timespec='seconds'),"kind":kind,"message":message})
        self.activity_log=self.activity_log[-300:]

    def action_logits(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits,_=self.policy(torch.tensor(obs,dtype=torch.float32,device=self.device)); return logits.detach().cpu().numpy()

    def act(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.action_logits(obs)))

    def analytics(self):
        return {**self.last_report,"trained":self.trained,"cuda_available":torch.cuda.is_available(),"gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',"training_status":self.training_status,"parameter_count":sum(p.numel() for p in self.policy.parameters()),"history_points":len(self.training_history),"log_count":len(self.activity_log)}

    def training_view(self):
        return {"status": self.training_status, "report": self.last_report, "history": self.training_history[-500:], "logs": self.activity_log[-150:], "hyperparameters": {"algorithm":"PPO","gamma":0.99,"gae_lambda":0.95,"clip_range":0.2,"entropy_coef":0.01,"architecture":"256-256-128 MLP","device":str(self.device)}}

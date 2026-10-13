from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import platform
import time
from datetime import datetime, timezone
from uuid import uuid4
from threading import Event
from threading import RLock

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from model.navigation.geodata import GlobalGeography, get_global_geography
from model.rl.navigation_env import NAVIGATION_MODEL_VERSION, POLICY_ACTION_SIZE, RECOVERY_ACTION, OceanNavigationEnv
from model.scenarios import app_version, dataset_fingerprints
from model.weather.simulator import WEATHER_MODEL_VERSION

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'data' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / 'navruna_ppo.pt'
META_PATH = MODEL_DIR / 'navruna_ppo.json'
HISTORY_PATH = MODEL_DIR / 'training_history.json'


def _atomic_write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNet(nn.Module):
    def __init__(self, obs_size=25, action_size=POLICY_ACTION_SIZE):
        super().__init__()
        self.body = nn.Sequential(
            layer_init(nn.Linear(obs_size, 256)),
            nn.LayerNorm(256),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
        )
        self.policy = layer_init(nn.Linear(128, action_size), std=0.01)
        self.value = layer_init(nn.Linear(128, 1), std=1.0)

    def forward(self, x):
        z = self.body(x)
        return self.policy(z), self.value(z).squeeze(-1)


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
        self.seed = int(seed)
        torch.manual_seed(seed); np.random.seed(seed)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy=PolicyNet(action_size=POLICY_ACTION_SIZE).to(self.device)
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=3e-4)
        self.total_updates=0
        self.episodes=0
        self.trained=False
        self.checkpoint_status='missing'
        self.training_status='idle'
        self.training_history=[]
        self._lock=RLock()
        self._training_lock=RLock()
        self.stop_requested=Event()
        self.activity_log=[]
        self.training_run=None
        if HISTORY_PATH.exists():
            try: self.training_history=json.loads(HISTORY_PATH.read_text(encoding='utf-8'))[-2000:]
            except Exception: self.training_history=[]
        self.last_report={"device":str(self.device),"updates":0,"episodes":0,"success_rate":0.0,"mean_return":0.0,"model_path":str(MODEL_PATH),"checkpoint_status":self.checkpoint_status}
        if MODEL_PATH.exists():
            try:
                checkpoint=torch.load(MODEL_PATH,map_location=self.device,weights_only=True)
                if isinstance(checkpoint,dict) and checkpoint.get('format_version') == 5 and checkpoint.get('navigation_model_version') == NAVIGATION_MODEL_VERSION:
                    if checkpoint.get('weather_model_version') != WEATHER_MODEL_VERSION:
                        self.checkpoint_status='incompatible_weather_model'
                        self._log('checkpoint_incompatible',f"Saved policy weather model {checkpoint.get('weather_model_version') or 'unversioned'} does not match {WEATHER_MODEL_VERSION}; bootstrap controller remains active.")
                    else:
                        self.policy.load_state_dict(checkpoint['model'])
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
                        self.total_updates=int(checkpoint.get('updates',0))
                        self.episodes=int(checkpoint.get('episodes',0))
                        self.training_run=checkpoint.get('training_run')
                        self.trained=True
                        self.checkpoint_status='compatible'
                elif isinstance(checkpoint,dict) and checkpoint.get('format_version') in (2, 3, 4, 5):
                    self.checkpoint_status='incompatible_navigation_model'
                    self._log('checkpoint_incompatible',f"Saved policy navigation interface {checkpoint.get('navigation_model_version') or 'legacy'} does not match {NAVIGATION_MODEL_VERSION}; bootstrap controller remains active until retrained.")
                else:
                    self.checkpoint_status='legacy_unversioned'
                    self._log('checkpoint_incompatible','Saved policy has no weather-model provenance; bootstrap controller remains active until retrained.')
            except Exception as exc:
                self.checkpoint_status='invalid'
                self._log('checkpoint_error',f'Checkpoint could not be loaded: {exc}')
        if META_PATH.exists():
            try:
                saved_report=json.loads(META_PATH.read_text(encoding='utf-8'))
                if isinstance(saved_report,dict):
                    self.last_report={**self.last_report,**saved_report}
                    self.training_run=saved_report.get('training_run',self.training_run)
            except Exception:
                pass
        self.last_report['trained']=self.trained
        self.last_report['checkpoint_status']=self.checkpoint_status

    def _save_checkpoint(self, updates: int, episodes: int):
        checkpoint={
            'format_version':5,
            'weather_model_version':WEATHER_MODEL_VERSION,
            'navigation_model_version':NAVIGATION_MODEL_VERSION,
            'observation_size':self.policy.body[0].in_features,
            'action_size':self.policy.policy.out_features,
            'model':self.policy.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'updates':int(updates),
            'episodes':int(episodes),
            'training_run':self.training_run,
        }
        temporary=MODEL_PATH.with_suffix('.pt.tmp')
        torch.save(checkpoint,temporary)
        os.replace(temporary,MODEL_PATH)

    def restore_last_committed_checkpoint(self) -> bool:
        """Reload the latest compatible atomic checkpoint after a failed update."""
        if not MODEL_PATH.exists():
            return False
        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get('format_version') != 5 or checkpoint.get('navigation_model_version') != NAVIGATION_MODEL_VERSION or checkpoint.get('weather_model_version') != WEATHER_MODEL_VERSION:
            return False
        self.policy.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_updates = int(checkpoint.get('updates', 0))
        self.episodes = int(checkpoint.get('episodes', 0))
        self.trained = True
        self.checkpoint_status = 'compatible'
        return True

    def train(self, total_steps=20000, n_envs=64, rollout=128, lr=3e-4, continuous=False):
        with self._training_lock:
            return self._train_locked(total_steps, n_envs, rollout, lr, continuous)

    def _train_locked(self, total_steps=20000, n_envs=64, rollout=128, lr=3e-4, continuous=False):
        started_at=datetime.now(timezone.utc).isoformat(timespec='seconds')
        speed_profile=(14.0,16.0,17.0,19.0,22.0,24.0)
        self.training_run={"run_id":str(uuid4()),"started_at_utc":started_at,"seed":self.seed,"requested_steps":int(total_steps),"parallel_environments":int(n_envs),"rollout_steps":int(rollout),"learning_rate":float(lr),"continuous":bool(continuous),"environment_seed_scheme":"1000 + environment index","training_cruise_speeds_kn":list(speed_profile),"weather_model_version":WEATHER_MODEL_VERSION,"navigation_model_version":NAVIGATION_MODEL_VERSION,"dataset_sha256":dataset_fingerprints(),"app_version":app_version(),"python_version":platform.python_version(),"numpy_version":np.__version__,"torch_version":torch.__version__,"code_revision":os.getenv('NAVRUNA_GIT_REVISION')}
        start=time.time(); self.training_status='training'; self._log('training_started',f"PPO {'continuous ' if continuous else ''}run started: {total_steps:,} steps per update target across {n_envs} environments"); self.optimizer.param_groups[0]['lr']=lr
        geo=get_global_geography(); envs=[OceanNavigationEnv(seed=1000+i,geography=geo,cruise_speed_kn=speed_profile[i%len(speed_profile)]) for i in range(n_envs)]
        curriculum_max=2600
        obs=np.stack([e.reset(min_route_nm=300,max_route_nm=curriculum_max) for e in envs])
        returns=[]; successes=0; finished=0; updates=0
        while not self.stop_requested.is_set() and (continuous or updates*rollout*n_envs < total_steps):
            O=[]; A=[]; LP=[]; V=[]; R=[]; D=[]; MASK=[]
            safe_counts=[]; no_safe_action_steps=0; recovery_maneuvers=0
            # The course shield is hard from the first transition; curriculum
            # expands the route length after successful terminal voyages.
            for e in envs: e.land_enabled = True
            for _ in range(rollout):
                mask_rows=[]
                for i,e in enumerate(envs):
                    mask=e.safe_action_mask(5.0)
                    # Do not turn an empty safety mask into an all-actions-allowed
                    # row. Recover under the same hard vector shield as fleet runs,
                    # then let PPO choose only after at least one legal action exists.
                    for recovery_attempt in range(4):
                        if mask.any():
                            break
                        no_safe_action_steps+=1
                        maneuver=e.safe_recovery_maneuver(1.0,5.0)
                        recovery_maneuvers+=int(maneuver is not None)
                        if maneuver is None:
                            _,_,done,info=e.hold(1.0)
                        else:
                            _,_,done,info=e.step(-1,1.0,turn_degrees=maneuver['turn_degrees'],speed_kn=maneuver['speed_kn'])
                        if done or recovery_attempt==3:
                            if not done:
                                e.done=True; e.failure='no_safe_action'; e.total_reward-=2.0
                            finished+=1
                            successes+=int(bool(info.get('success')))
                            returns.append(e.total_reward)
                            e.reset(min_route_nm=300,max_route_nm=curriculum_max)
                        mask=e.safe_action_mask(5.0)
                    if not mask.any():
                        # A freshly reset episode should have a safe departure. Fail
                        # loudly rather than contaminating PPO with an unsafe sample.
                        raise RuntimeError('Could not produce a safety-checked training action after recovery and reset.')
                    mask_rows.append(mask)
                    no_safe_action_steps+=int(mask[RECOVERY_ACTION])
                    obs[i]=e.observe()
                masks=np.stack(mask_rows)
                ot=torch.tensor(obs,dtype=torch.float32,device=self.device)
                counts=masks.sum(axis=1)
                safe_counts.extend(counts.tolist())
                mask_t=torch.tensor(masks,device=self.device)
                with torch.no_grad():
                    logits,vals=self.policy(ot); logits=logits.masked_fill(~mask_t,-1e9); dist=Categorical(logits=logits); act=dist.sample(); lp=dist.log_prob(act)
                actions=act.detach().cpu().numpy();
                nxt=np.empty_like(obs); rr=np.zeros(n_envs,np.float32); dd=np.zeros(n_envs,np.float32)
                for i,e in enumerate(envs):
                    no,r,d,info=e.step(int(actions[i]));
                    recovery_maneuvers+=int(bool(info.get('recovery')))
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
            oldlp_t=flat(LP); oldv_t=flat(V); adv_t=flat(adv); ret_t=flat(ret)
            adv_t=(adv_t-adv_t.mean())/(adv_t.std()+1e-8)
            stop_epoch=False
            ent_coef = max(0.003, 0.015 * (1.0 - 0.7 * (updates * rollout * n_envs) / max(1, total_steps)))
            for _epoch in range(4):
                idx=np.random.permutation(obs_t.shape[0])
                for s in range(0,len(idx),2048):
                    ids=torch.tensor(idx[s:s+2048],device=self.device)
                    logits,val=self.policy(obs_t[ids]); logits=logits.masked_fill(~mask_t[ids],-1e9); dist=Categorical(logits=logits); logp=dist.log_prob(act_t[ids]); entropy=dist.entropy().mean()
                    ratio=torch.exp(logp-oldlp_t[ids]); a=adv_t[ids]
                    a=(a-a.mean())/(a.std()+1e-8)
                    loss_pi=-torch.min(ratio*a,torch.clamp(ratio,.8,1.2)*a).mean()
                    v_loss_unclipped=(val-ret_t[ids]).pow(2)
                    v_clipped=oldv_t[ids]+torch.clamp(val-oldv_t[ids],-0.2,0.2)
                    v_loss_clipped=(v_clipped-ret_t[ids]).pow(2)
                    loss_v=0.5*torch.max(v_loss_unclipped,v_loss_clipped).mean()
                    loss=loss_pi+loss_v-ent_coef*entropy
                    log_ratio=logp-oldlp_t[ids]
                    approx_kl=((torch.exp(log_ratio)-1.0)-log_ratio).mean().detach()
                    self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.policy.parameters(),0.7); self.optimizer.step()
                    if float(approx_kl.cpu())>0.03:
                        stop_epoch=True
                        break
                if stop_epoch:
                    break
            current_lr=max(1e-5,lr*(1.0-(updates*rollout*n_envs)/max(1,total_steps)))
            for param_group in self.optimizer.param_groups:
                param_group['lr']=current_lr
            with torch.no_grad():
                y_pred=val.detach().cpu().numpy()
                y_true=ret_t[ids].detach().cpu().numpy()
                var_y=float(np.var(y_true))
                explained_var=float(1.0-np.var(y_true-y_pred)/var_y) if var_y>1e-6 else 0.0
            updates+=1
            recent=returns[-100:]
            elapsed=time.time()-start
            point={"run_id":self.training_run["run_id"],"update": self.total_updates+updates, "steps": updates*rollout*n_envs, "steps_per_sec": (updates*rollout*n_envs)/max(1e-6,elapsed), "episodes": finished, "success_rate": successes/max(1,finished), "mean_return": float(np.mean(recent)) if recent else 0.0, "mean_safe_actions": float(np.mean(safe_counts)) if safe_counts else 0.0, "no_safe_action_steps": no_safe_action_steps, "safe_recovery_maneuvers": recovery_maneuvers, "loss_policy": float(loss_pi.detach().cpu()), "loss_value": float(loss_v.detach().cpu()), "explained_variance": round(explained_var, 4), "entropy": float(entropy.detach().cpu()), "approx_kl": float(approx_kl.cpu()), "learning_rate": current_lr, "elapsed_s": time.time()-start}
            self.training_history.append(point); self.training_history=self.training_history[-2000:]
            self._log('update',f"Update {point['update']}: {point['steps']:,} steps, {point['steps_per_sec']:.0f} steps/s, success {point['success_rate']*100:.1f}%, expl_var {point['explained_variance']:.2f}")
            _atomic_write_json(HISTORY_PATH,self.training_history)
            if updates % 25 == 0:
                self._save_checkpoint(self.total_updates+updates,self.episodes+finished)
            time.sleep(0)
        if updates > 0:
            self.trained=True
        rate=successes/max(1,finished)
        report=TrainReport(str(self.device),updates,updates*rollout*n_envs,finished,rate,float(np.mean(returns[-500:]) if returns else 0),time.time()-start,str(MODEL_PATH))
        self.total_updates+=updates; self.episodes+=finished
        self.last_report={**report.__dict__,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        self.training_run={**self.training_run,"finished_at_utc":datetime.now(timezone.utc).isoformat(timespec='seconds'),"actual_transitions":report.transitions,"completed_episodes":finished,"success_rate":rate,"mean_return":report.mean_return,"elapsed_s":report.elapsed_s,"status":"stopped" if self.stop_requested.is_set() else "completed"}
        self.last_report["training_run"]=self.training_run
        if updates > 0:
            self.checkpoint_status='compatible'
        self.last_report['checkpoint_status']=self.checkpoint_status
        self.last_report['trained']=self.trained
        if updates > 0:
            self._save_checkpoint(self.total_updates,self.episodes)
        self.training_status='idle'
        self._log('training_stopped' if self.stop_requested.is_set() else 'training_complete',f"Training {'stopped' if self.stop_requested.is_set() else 'complete'}: {report.transitions:,} transitions, {report.success_rate*100:.1f}% success")
        _atomic_write_json(META_PATH,self.last_report)
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
        return {**self.last_report,"updates":self.total_updates,"episodes":self.episodes,"trained":self.trained,"controller":"PPO checkpoint" if self.trained else "bootstrap heading heuristic","cuda_available":torch.cuda.is_available(),"gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',"training_status":self.training_status,"parameter_count":sum(p.numel() for p in self.policy.parameters()),"history_points":len(self.training_history),"log_count":len(self.activity_log)}

    def training_view(self):
        report={**self.last_report,"updates":self.total_updates,"episodes":self.episodes}
        if self.training_run:
            report["training_run"]=self.training_run
            run_id=self.training_run.get("run_id")
            run_points=[point for point in self.training_history if point.get("run_id")==run_id]
            if self.training_status in ("starting","training","stopping") and run_points:
                latest=run_points[-1]
                report.update({
                    "updates":latest.get("update",self.total_updates),
                    "episodes":self.episodes+latest.get("episodes",0),
                    "transitions":latest.get("steps",0),
                    "success_rate":latest.get("success_rate",0.0),
                    "mean_return":latest.get("mean_return",0.0),
                    "elapsed_s":latest.get("elapsed_s"),
                })
        return {"status": self.training_status, "report": report, "history": self.training_history[-500:], "logs": self.activity_log[-150:], "hyperparameters": {"algorithm":"PPO with hard action masking, a learned recovery action, and KL early stopping","gamma":0.99,"gae_lambda":0.95,"clip_range":0.2,"entropy_coef":0.01,"target_kl":0.03,"action_count":POLICY_ACTION_SIZE,"architecture":"256-256-128 MLP","device":str(self.device)}}

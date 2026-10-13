from __future__ import annotations

import math
import random
import uuid
import copy
import numpy as np
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any

from model.navigation.geodata import GlobalGeography, load_ports, get_global_geography
from model.rl.navigation_env import NAVIGATION_MODEL_VERSION, RECOVERY_ACTION, OceanNavigationEnv, bootstrap_heading_action, haversine_nm
from model.weather.simulator import WeatherSimulator

VESSEL_TYPES = [
    ("container", "Container Ship", 24),
    ("tanker", "Oil Tanker", 16),
    ("bulk", "Bulk Carrier", 14),
    ("lng", "LNG Tanker", 19),
    ("vlcc", "VLCC", 14),
    ("general", "General Cargo", 17),
]
CARGO = ["containers", "grain", "crude oil", "LNG", "ore", "vehicles", "mixed cargo"]



@dataclass
class Vessel:
    id: str
    origin: str
    destination: str
    vessel_type: str
    vessel_class: str
    cargo: str
    load_pct: int
    speed_kn: float
    distance_nm: float
    progress: float
    status: str
    fuel_capacity_t: float
    fuel_remaining_t: float
    fuel_used_t: float = 0.0
    co2_t: float = 0.0
    risk: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    heading: float = 0.0
    land_clearance_km: float = 999.0
    ai_action: int = 3
    ai_action_name: str = "STRAIGHT"
    reward_last: float = 0.0
    steps: int = 0
    reoptimizations: int = 0
    failure_reason: str = ""
    wind_kn: float = 0.0
    wind_dir_deg: float = 0.0
    wave_m: float = 0.0
    current_kn: float = 0.0
    current_dir_deg: float = 0.0
    storm: float = 0.0
    precipitation: float = 0.0
    visibility_nm: float = 20.0
    observation: list[float] = field(default_factory=list)
    voyages_completed: int = 0
    distance_travelled_nm: float = 0.0
    minimum_clearance_km: float = 999.0
    recovery_maneuvers: int = 0
    recovery_streak: int = 0
    env: Any = field(default=None, repr=False, compare=False)

    def public(self):
        d = {name: getattr(self,name) for name in self.__dataclass_fields__ if name != 'env'}
        return d


class FleetSimulator:
    """Multi-vessel online simulator using the shared PPO navigation policy."""

    def __init__(self, seed=20260922, trainer=None, scenario_config=None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.geo = get_global_geography()
        self.ports = load_ports()
        self.trainer = trainer
        self.policy_snapshot = None
        self.weather = WeatherSimulator(seed=seed)
        self.vessels: list[Vessel] = []
        self.sim_hours = 0.0
        self.events: list[dict] = []
        self.metrics_state = self._empty_metrics()
        self.scenario_config = scenario_config
        self.run_metadata: dict[str, Any] | None = None

    @staticmethod
    def _empty_metrics():
        return {"land_rejections":0,"collisions":0,"weather_losses":0,"near_misses":0,"voyages_started":0,"voyages_completed":0,"voyage_failures":0,"failure_reasons":{},"action_mask_steps":0,"safe_action_total":0,"no_safe_action_steps":0,"safe_recoveries":0,"fleet_fuel_used_t":0.0,"fleet_distance_nm":0.0}

    def _record_failure(self, reason: str):
        reason=reason.upper()
        self.metrics_state['voyage_failures']+=1
        counts=self.metrics_state['failure_reasons']
        counts[reason]=counts.get(reason,0)+1

    def attach_trainer(self, trainer): self.trainer = trainer

    def reset(self, clear_run: bool = False):
        self.vessels=[]; self.sim_hours=0.0; self.events=[]
        self.metrics_state=self._empty_metrics()
        if clear_run:
            self.scenario_config=None
            self.run_metadata=None

    def _pair(self):
        a,b=self.rng.sample(self.ports,2)
        for _ in range(20):
            if haversine_nm((a['lat'],a['lon']),(b['lat'],b['lon']))>450: return a,b
            a,b=self.rng.sample(self.ports,2)
        candidates=[]
        for origin in self.ports:
            for destination in self.ports:
                if origin['id'] != destination['id']:
                    distance=haversine_nm((origin['lat'],origin['lon']),(destination['lat'],destination['lon']))
                    if distance > 450:
                        candidates.append((distance, origin, destination))
        if not candidates:
            raise RuntimeError('Port catalog contains no valid voyage pair longer than 450 nm.')
        _, origin, destination = self.rng.choice(candidates)
        return origin, destination

    def generate_fleet(self,count=60):
        self.reset()
        self.policy_snapshot = None
        if self.trainer and getattr(self.trainer, 'trained', False):
            self.policy_snapshot = copy.deepcopy(self.trainer.policy).to('cpu').eval()
            for parameter in self.policy_snapshot.parameters():
                parameter.requires_grad_(False)
        if self.scenario_config is None:
            from model.scenarios import make_scenario
            self.scenario_config = make_scenario(self.seed, count).model_dump(mode="json")
        else:
            self.scenario_config = {**self.scenario_config, "vessel_count": count}
        self.run_metadata = {
            "run_id": str(uuid.uuid4()),
            "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": self.seed,
            "scenario": self.scenario_config,
            "controller_at_start": "PPO checkpoint" if self.policy_snapshot is not None else "bootstrap heading heuristic",
            "navigation_model_version": NAVIGATION_MODEL_VERSION,
            "training_run_id": (getattr(self.trainer,"training_run",None) or {}).get("run_id") if self.policy_snapshot is not None else None,
        }
        for i in range(count):
            for _attempt in range(100):
                origin,dest=self._pair()
                try:
                    env=OceanNavigationEnv(seed=self.seed+i*31,weather_seed=self.seed,geography=self.geo,origin=origin,destination=dest)
                    break
                except ValueError:
                    if _attempt == 99:
                        raise RuntimeError('Unable to create a valid offshore voyage from the port catalog.')
            vt,vc,cruise=self.rng.choice(VESSEL_TYPES)
            env.cruise_speed_kn=float(cruise); env.speed=float(cruise)
            load=self.rng.randint(35,100)
            fuel=max(600.0, env.initial_distance*(1.05+self.rng.random()*.45))
            env.fuel_capacity=fuel; env.fuel=fuel
            v=Vessel(
                id=f"NV-{i+1:03d}",origin=origin['name'],destination=dest['name'],
                vessel_type=vt,vessel_class=vc,cargo=self.rng.choice(CARGO),load_pct=load,
                speed_kn=env.speed,distance_nm=env.initial_distance,progress=0.0,status='active',
                fuel_capacity_t=fuel,fuel_remaining_t=fuel,lat=env.lat,lon=env.lon,heading=env.heading,
                land_clearance_km=self.geo.distance_to_land_km(env.lat,env.lon),minimum_clearance_km=self.geo.distance_to_land_km(env.lat,env.lon),env=env,
            )
            w0=self.weather.at(v.lat,v.lon,0)
            v.wind_kn=w0.wind_kn; v.wind_dir_deg=w0.wind_dir_deg; v.wave_m=w0.wave_m; v.current_kn=w0.current_kn; v.current_dir_deg=w0.current_dir_deg; v.storm=w0.storm; v.precipitation=w0.precipitation; v.visibility_nm=w0.visibility_nm
            self.vessels.append(v)
        self.events=[{"type":"simulation_started","message":f"{count} RL vessels dispatched","time":0}]
        self.metrics_state['voyages_started']=count
        return self.snapshot(include_routes=False)

    def _choose_action(self,v):
        v.env.weather_time=self.sim_hours
        obs=v.env.observe(); v.observation=obs.tolist()
        candidates=v.env.safe_actions(5.0)
        self.metrics_state['action_mask_steps']+=1
        self.metrics_state['safe_action_total']+=len(candidates)
        if not any(action<RECOVERY_ACTION for action in candidates):
            self.metrics_state['no_safe_action_steps']+=1
            self.metrics_state['land_rejections']+=1
        if not candidates:
            return None
        if self.policy_snapshot is not None:
            import torch
            with torch.no_grad():
                logits,_=self.policy_snapshot(torch.as_tensor(obs,dtype=torch.float32))
            logits=logits.numpy()
            action=max(candidates,key=lambda a: float(logits[a]))
        else:
            # Greedy one-step progress with a coastline-clearance cost, under the
            # same hard vector shield used by the learned policy.
            action=bootstrap_heading_action(v.env,candidates)
        return int(action)

    def _assign_next_voyage(self, v):
        v.env.weather_time=self.sim_hours
        for _attempt in range(100):
            origin, destination = self._pair()
            try:
                v.env.reset(origin, destination)
                break
            except ValueError:
                if _attempt == 99:
                    raise RuntimeError(f'Unable to assign a next voyage to {v.id}.')
        v.origin=origin['name']; v.destination=destination['name']
        self.metrics_state['voyages_started']+=1
        v.distance_nm=v.env.initial_distance; v.progress=0.0; v.status='active'
        v.failure_reason=''; v.lat=v.env.lat; v.lon=v.env.lon; v.heading=v.env.heading
        v.recovery_streak=0
        v.speed_kn=v.env.speed; v.fuel_capacity_t=max(600.0,v.env.initial_distance*(1.05+self.rng.random()*.45)); v.fuel_remaining_t=v.fuel_capacity_t; v.fuel_used_t=0.0
        v.env.fuel_capacity=v.fuel_capacity_t; v.env.fuel=v.fuel_capacity_t
        v.land_clearance_km=self.geo.distance_to_land_km(v.lat,v.lon)
        v.minimum_clearance_km=v.land_clearance_km; v.distance_travelled_nm=0.0
        w=self.weather.at(v.lat,v.lon,self.sim_hours); v.wind_kn=w.wind_kn; v.wind_dir_deg=w.wind_dir_deg; v.wave_m=w.wave_m; v.current_kn=w.current_kn; v.current_dir_deg=w.current_dir_deg; v.storm=w.storm; v.precipitation=w.precipitation; v.visibility_nm=w.visibility_nm

    def step(self,hours=6.0):
        hours=max(0.0, float(hours))
        moved_vessels=set()
        processed_vessels=set()
        motion_paths={}
        full_steps=int(hours)
        increments=[1.0]*full_steps
        if hours-full_steps > 1e-9: increments.append(hours-full_steps)
        for elapsed in increments:
            self.sim_hours+=elapsed
            for v in self.vessels:
                if v.status!='active': continue
                processed_vessels.add(v.id)
                old_position=(v.env.lat,v.env.lon)
                fuel_before=v.env.fuel
                action=self._choose_action(v)
                recovery=None
                if action is None:
                    recovery=v.env.safe_recovery_maneuver(elapsed,5.0)
                    if recovery is not None:
                        action=-1
                        obs,reward,done,info=v.env.step(
                            -1, elapsed, turn_degrees=recovery['turn_degrees'], speed_kn=recovery['speed_kn']
                        )
                        v.recovery_maneuvers+=1; v.recovery_streak=0; v.reoptimizations+=1
                        self.metrics_state['safe_recoveries']+=1
                        v.ai_action_name=f"SAFE RECOVERY {recovery['turn_degrees']:+.0f}° / {recovery['speed_kn']:.0f} kt"
                        self.events.append({"type":"safe_recovery","message":f"{v.id} used a vector-checked recovery course","time":round(self.sim_hours,1)})
                    else:
                        action=-1
                        obs,reward,done,info=v.env.hold(elapsed)
                        v.recovery_streak+=1; v.reoptimizations+=1
                        v.ai_action_name='HOLD · SEARCHING SAFE COURSE'
                        if v.recovery_streak>=3 and not done:
                            v.env.done=True; v.env.failure='no_safe_action'; done=True
                            info={**info,"failure":"no_safe_action"}
                    v.ai_action=-1
                    v.reward_last=float(reward)
                    if recovery is None:
                        self.events.append({"type":"holding","message":f"{v.id} is holding while the safety shield searches for an escape course","time":round(self.sim_hours,1)})
                else:
                    obs,reward,done,info=v.env.step(action, elapsed)
                    v.ai_action=action
                    if action==RECOVERY_ACTION and info.get('recovery'):
                        maneuver=info['recovery']
                        v.recovery_maneuvers+=1; v.recovery_streak=0; v.reoptimizations+=1
                        self.metrics_state['safe_recoveries']+=1
                        v.ai_action_name=f"SAFE RECOVERY {maneuver['turn_degrees']:+.0f}° / {maneuver['speed_kn']:.0f} kt"
                        self.events.append({"type":"safe_recovery","message":f"{v.id} used a vector-checked recovery course","time":round(self.sim_hours,1)})
                    elif action==RECOVERY_ACTION:
                        v.ai_action_name='RECOVERY COURSE UNAVAILABLE'
                    else:
                        names=['HARD LEFT','LEFT','SOFT LEFT','GENTLE LEFT','STRAIGHT','GENTLE RIGHT','SOFT RIGHT','RIGHT','HARD RIGHT']
                        v.ai_action_name=names[action]
                    v.reward_last=float(reward)
                traveled=haversine_nm(old_position,(v.env.lat,v.env.lon))
                if traveled>1e-6: moved_vessels.add(v.id)
                v.lat=v.env.lat; v.lon=v.env.lon; v.heading=v.env.heading; v.speed_kn=v.env.speed
                v.distance_travelled_nm+=traveled
                self.metrics_state['fleet_distance_nm']+=traveled
                v.fuel_remaining_t=max(0,v.env.fuel); v.fuel_used_t=v.fuel_capacity_t-v.fuel_remaining_t
                self.metrics_state['fleet_fuel_used_t']+=max(0.0,fuel_before-v.env.fuel)
                v.co2_t=v.fuel_used_t*3.114
                v.steps=v.env.steps; v.distance_nm=v.env.initial_distance
                v.progress=max(0,min(1,1-info.get('distance_nm',v.env.prev_distance)/max(1,v.env.initial_distance)))
                v.land_clearance_km=self.geo.distance_to_land_km(v.lat,v.lon)
                v.minimum_clearance_km=min(v.minimum_clearance_km,v.land_clearance_km)
                w=self.weather.at(v.lat,v.lon,self.sim_hours)
                v.wind_kn=w.wind_kn; v.wind_dir_deg=w.wind_dir_deg; v.wave_m=w.wave_m; v.current_kn=w.current_kn; v.current_dir_deg=w.current_dir_deg; v.storm=w.storm; v.precipitation=w.precipitation; v.visibility_nm=w.visibility_nm
                v.risk=max(0,min(100,(25-v.land_clearance_km)*3 + w.wave_m*7 + w.storm*18 + w.precipitation*5))
                if done:
                    if info.get('success'):
                        completed_destination=v.destination
                        v.voyages_completed += 1; self.metrics_state['voyages_completed'] += 1
                        self.events.append({"type":"voyage_completed","message":f"{v.id} reached {completed_destination}","time":round(self.sim_hours,1)})
                        self._assign_next_voyage(v)
                        self.events.append({"type":"voyage_started","message":f"{v.id} continuing from {v.origin} to {v.destination}","time":round(self.sim_hours,1)})
                    elif info.get('failure')=='land_collision':
                        v.status='destroyed'; v.failure_reason='LAND COLLISION'; self.metrics_state['collisions']+=1
                        self._record_failure('LAND_COLLISION')
                        self.events.append({"type":"land_collision","message":f"{v.id} destroyed by land boundary","time":round(self.sim_hours,1)})
                    else:
                        v.status='failed'; v.failure_reason=info.get('failure','FUEL EXHAUSTED').replace('_',' ').upper()
                        self._record_failure(info.get('failure','fuel_exhausted'))
                        if 'fuel' in info.get('failure',''): self.metrics_state['weather_losses']+=1
                        self.events.append({"type":"voyage_failed","message":f"{v.id} {v.failure_reason.lower()}","time":round(self.sim_hours,1)})
                motion_paths.setdefault(v.id,[]).append({"lat":v.lat,"lon":v.lon,"heading":v.heading})
        result=self.snapshot(include_routes=False)
        result['tick']={'hours':hours,'processed_vessels':len(processed_vessels),'moved_vessels':len(moved_vessels),'active_vessels':result['metrics']['active']}
        result['motion_paths']=motion_paths
        return result

    def snapshot(self,include_routes=False):
        return {"sim_hours":round(self.sim_hours,1),"vessels":[v.public() for v in self.vessels],"metrics":self.analytics(),"events":self.events[-40:],"run":self.run_metadata,"scenario":self.scenario_config}

    def analytics(self):
        total=len(self.vessels); completed=sum(v.status=='completed' for v in self.vessels); active=sum(v.status=='active' for v in self.vessels)
        failed=total-completed-active
        avgfuel=sum(v.fuel_used_t for v in self.vessels)/max(1,total)
        avgco2=sum(v.co2_t for v in self.vessels)/max(1,total)
        avgrisk=sum(v.risk for v in self.vessels)/max(1,total)
        distance_nm=self.metrics_state['fleet_distance_nm']
        mean_min_clearance=sum(v.minimum_clearance_km for v in self.vessels)/max(1,total)
        attempts=self.metrics_state['voyages_completed']+self.metrics_state['voyage_failures']
        success_rate=self.metrics_state['voyages_completed']/max(1,attempts)*100
        failures=self.metrics_state['failure_reasons']
        pct=lambda count:round(count/max(1,attempts)*100,1)
        controller=(self.run_metadata or {}).get("controller_at_start","bootstrap heading heuristic")
        return {"total_vessels":total,"active":active,"completed":completed,"completed_vessels":completed,"failed":failed,"destroyed":sum(v.status=='destroyed' for v in self.vessels),"success_rate":round(success_rate,1),"voyage_success_rate":round(success_rate,1),"collision_rate":pct(self.metrics_state['collisions']),"fuel_failure_rate":pct(failures.get('FUEL_EXHAUSTED',0)),"timeout_rate":pct(failures.get('TIME_LIMIT',0)),"no_safe_action_rate":pct(failures.get('NO_SAFE_ACTION',0)),"mean_min_clearance_km":round(mean_min_clearance,2),"fuel_per_nm":round(self.metrics_state['fleet_fuel_used_t']/max(1,distance_nm),4),"fleet_fuel_used_t":round(self.metrics_state['fleet_fuel_used_t'],2),"fleet_distance_nm":round(distance_nm,2),"mean_safe_actions":round(self.metrics_state['safe_action_total']/max(1,self.metrics_state['action_mask_steps']),2),"no_safe_action_steps":self.metrics_state['no_safe_action_steps'],"safe_recoveries":self.metrics_state['safe_recoveries'],"avg_fuel_t":round(avgfuel,1),"avg_co2_t":round(avgco2,1),"avg_risk":round(avgrisk,1),"routes":0,"policy":controller,"land_rejections":self.metrics_state['land_rejections'],"collisions":self.metrics_state['collisions'],"weather_losses":self.metrics_state['weather_losses'],"near_misses":self.metrics_state['near_misses'],"voyages_started":self.metrics_state['voyages_started'],"voyages_completed":self.metrics_state['voyages_completed'],"voyages_failed":self.metrics_state['voyage_failures'],"voyage_failures":self.metrics_state['voyage_failures'],"failure_reasons":dict(failures)}

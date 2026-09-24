from __future__ import annotations

import math
import random
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Any

from model.navigation.geodata import GlobalGeography, load_ports, get_global_geography
from model.rl.navigation_env import OceanNavigationEnv, haversine_nm
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
    env: Any = field(default=None, repr=False, compare=False)

    def public(self):
        d = {name: getattr(self,name) for name in self.__dataclass_fields__ if name != 'env'}
        return d


class FleetSimulator:
    """Multi-vessel online simulator using the shared PPO navigation policy."""

    def __init__(self, seed=20260922, trainer=None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.geo = get_global_geography()
        self.ports = load_ports()
        self.trainer = trainer
        self.weather = WeatherSimulator(seed=seed)
        self.vessels: list[Vessel] = []
        self.sim_hours = 0.0
        self.events: list[dict] = []
        self.metrics_state = {"land_rejections":0,"collisions":0,"weather_losses":0,"near_misses":0,"voyages_completed":0,"voyage_failures":0}

    def attach_trainer(self, trainer): self.trainer = trainer

    def reset(self):
        self.vessels=[]; self.sim_hours=0.0; self.events=[]
        self.metrics_state={"land_rejections":0,"collisions":0,"weather_losses":0,"near_misses":0,"voyages_completed":0,"voyage_failures":0}

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
        for i in range(count):
            for _attempt in range(100):
                origin,dest=self._pair()
                try:
                    env=OceanNavigationEnv(seed=self.seed+i*31,geography=self.geo,origin=origin,destination=dest)
                    break
                except ValueError:
                    if _attempt == 99:
                        raise RuntimeError('Unable to create a valid offshore voyage from the port catalog.')
            vt,vc,cruise=self.rng.choice(VESSEL_TYPES)
            load=self.rng.randint(35,100)
            fuel=max(600.0, env.initial_distance*(1.05+self.rng.random()*.45))
            env.fuel_capacity=fuel; env.fuel=fuel
            v=Vessel(
                id=f"NV-{i+1:03d}",origin=origin['name'],destination=dest['name'],
                vessel_type=vt,vessel_class=vc,cargo=self.rng.choice(CARGO),load_pct=load,
                speed_kn=env.speed,distance_nm=env.initial_distance,progress=0.0,status='active',
                fuel_capacity_t=fuel,fuel_remaining_t=fuel,lat=env.lat,lon=env.lon,heading=env.heading,
                land_clearance_km=self.geo.distance_to_land_km(env.lat,env.lon),env=env,
            )
            w0=self.weather.at(v.lat,v.lon,0)
            v.wind_kn=w0.wind_kn; v.wind_dir_deg=w0.wind_dir_deg; v.wave_m=w0.wave_m; v.current_kn=w0.current_kn; v.current_dir_deg=w0.current_dir_deg; v.storm=w0.storm; v.precipitation=w0.precipitation; v.visibility_nm=w0.visibility_nm
            self.vessels.append(v)
        self.events=[{"type":"simulation_started","message":f"{count} RL vessels dispatched","time":0}]
        return self.snapshot(include_routes=False)

    def _choose_action(self,v):
        obs=v.env.observe(); v.observation=obs.tolist()
        safe=np.flatnonzero(v.env.action_mask_fast(5.0)).tolist()
        verified=[a for a in range(v.env.action_size) if v.env.is_action_safe(a,5.0)]
        candidates=verified
        if not candidates:
            v.reoptimizations += 1
            return None
        if self.trainer and getattr(self.trainer,'trained',False):
            logits=self.trainer.action_logits(obs)
            action=max(candidates,key=lambda a: float(logits[a]))
        else:
            # Bootstrap controller: point at destination, but still pass through the
            # same hard safety shield used by the learned policy.
            rel=obs[5]*180
            from model.rl.navigation_env import ACTIONS
            action=min(candidates,key=lambda a:abs(float(ACTIONS[a])-rel))
        return int(action)

    def _assign_next_voyage(self, v):
        for _attempt in range(100):
            origin, destination = self._pair()
            try:
                v.env.reset(origin, destination)
                break
            except ValueError:
                if _attempt == 99:
                    raise RuntimeError(f'Unable to assign a next voyage to {v.id}.')
        v.origin=origin['name']; v.destination=destination['name']
        v.distance_nm=v.env.initial_distance; v.progress=0.0; v.status='active'
        v.failure_reason=''; v.lat=v.env.lat; v.lon=v.env.lon; v.heading=v.env.heading
        v.speed_kn=v.env.speed; v.fuel_capacity_t=max(600.0,v.env.initial_distance*(1.05+self.rng.random()*.45)); v.fuel_remaining_t=v.fuel_capacity_t; v.fuel_used_t=0.0
        v.land_clearance_km=self.geo.distance_to_land_km(v.lat,v.lon)
        w=self.weather.at(v.lat,v.lon,self.sim_hours); v.wind_kn=w.wind_kn; v.wind_dir_deg=w.wind_dir_deg; v.wave_m=w.wave_m; v.current_kn=w.current_kn; v.current_dir_deg=w.current_dir_deg; v.storm=w.storm; v.precipitation=w.precipitation; v.visibility_nm=w.visibility_nm

    def step(self,hours=6.0):
        hours=max(0.0, float(hours))
        moved_vessels=set()
        processed_vessels=set()
        full_steps=int(hours)
        increments=[1.0]*full_steps
        if hours-full_steps > 1e-9: increments.append(hours-full_steps)
        for elapsed in increments:
            self.sim_hours+=elapsed
            for v in self.vessels:
                if v.status!='active': continue
                processed_vessels.add(v.id)
                old=v.env.observe()
                action=self._choose_action(v)
                if action is None:
                    self.metrics_state['land_rejections'] += 1
                    v.reward_last=-2.0; v.ai_action=-1; v.ai_action_name='HOLD / REPLAN'; v.status='failed'; v.failure_reason='NO SAFE ACTION'; v.fuel_remaining_t=max(0.0,v.fuel_remaining_t-0.2); v.fuel_used_t=v.fuel_capacity_t-v.fuel_remaining_t; v.co2_t=v.fuel_used_t*3.114; v.steps+=1; v.reoptimizations+=1
                    self.metrics_state['voyage_failures']+=1
                    self.events.append({"type":"voyage_failed","message":f"{v.id} held because no safe action was available","time":round(self.sim_hours,1)})
                    continue
                obs,reward,done,info=v.env.step(action, elapsed)
                moved_vessels.add(v.id)
                v.reward_last=float(reward); v.ai_action=action
                names=['HARD LEFT','LEFT','SOFT LEFT','STRAIGHT','SOFT RIGHT','RIGHT','HARD RIGHT']; v.ai_action_name=names[action]
                v.lat=v.env.lat; v.lon=v.env.lon; v.heading=v.env.heading; v.speed_kn=v.env.speed
                v.fuel_remaining_t=max(0,v.env.fuel); v.fuel_used_t=v.fuel_capacity_t-v.fuel_remaining_t
                v.co2_t=v.fuel_used_t*3.114
                v.steps=v.env.steps; v.distance_nm=v.env.initial_distance
                v.progress=max(0,min(1,1-v.env.prev_distance/max(1,v.env.initial_distance)))
                v.land_clearance_km=self.geo.distance_to_land_km(v.lat,v.lon)
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
                        self.metrics_state['voyage_failures']+=1
                        self.events.append({"type":"land_collision","message":f"{v.id} destroyed by land boundary","time":round(self.sim_hours,1)})
                    else:
                        v.status='failed'; v.failure_reason=info.get('failure','FUEL EXHAUSTED').replace('_',' ').upper()
                        self.metrics_state['voyage_failures']+=1
                        if 'fuel' in info.get('failure',''): self.metrics_state['weather_losses']+=1
                        self.events.append({"type":"voyage_failed","message":f"{v.id} {v.failure_reason.lower()}","time":round(self.sim_hours,1)})
        result=self.snapshot(include_routes=False)
        result['tick']={'hours':hours,'processed_vessels':len(processed_vessels),'moved_vessels':len(moved_vessels),'active_vessels':result['metrics']['active']}
        return result

    def snapshot(self,include_routes=False):
        return {"sim_hours":round(self.sim_hours,1),"vessels":[v.public() for v in self.vessels],"metrics":self.analytics(),"events":self.events[-40:]}

    def analytics(self):
        total=len(self.vessels); completed=sum(v.status=='completed' for v in self.vessels); active=sum(v.status=='active' for v in self.vessels)
        failed=total-completed-active
        avgfuel=sum(v.fuel_used_t for v in self.vessels)/max(1,total)
        avgco2=sum(v.co2_t for v in self.vessels)/max(1,total)
        avgrisk=sum(v.risk for v in self.vessels)/max(1,total)
        attempts=self.metrics_state['voyages_completed']+self.metrics_state['voyage_failures']
        success_rate=self.metrics_state['voyages_completed']/max(1,attempts)*100
        return {"total_vessels":total,"active":active,"completed":completed,"failed":failed,"destroyed":sum(v.status=='destroyed' for v in self.vessels),"success_rate":round(success_rate,1),"avg_fuel_t":round(avgfuel,1),"avg_co2_t":round(avgco2,1),"avg_risk":round(avgrisk,1),"routes":0,"policy":"PPO / constrained action masking","land_rejections":self.metrics_state['land_rejections'],"collisions":self.metrics_state['collisions'],"weather_losses":self.metrics_state['weather_losses'],"near_misses":self.metrics_state['near_misses'],"voyages_completed":self.metrics_state['voyages_completed'],"voyage_failures":self.metrics_state['voyage_failures']}

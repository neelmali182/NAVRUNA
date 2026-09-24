from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from model.navigation.geodata import GlobalGeography, load_ports, get_global_geography
from model.weather.simulator import WeatherSimulator

EARTH_RADIUS_NM = 3440.065
ACTIONS = np.array([-35, -20, -10, 0, 10, 20, 35], dtype=np.float32)
SPEEDS = np.array([10.0, 16.0, 22.0], dtype=np.float32)


def haversine_nm(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = math.radians(b[0]-a[0]), math.radians(b[1]-a[1])
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*EARTH_RADIUS_NM*math.asin(min(1.0, math.sqrt(h)))


def bearing_deg(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1]-a[1])
    y = math.sin(dl)*math.cos(p2)
    x = math.cos(p1)*math.sin(p2)-math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(y,x))+360)%360


def destination_point(lat, lon, bearing, distance_nm):
    d = distance_nm/EARTH_RADIUS_NM
    p1 = math.radians(lat); l1 = math.radians(lon); br = math.radians(bearing)
    p2 = math.asin(math.sin(p1)*math.cos(d)+math.cos(p1)*math.sin(d)*math.cos(br))
    l2 = l1 + math.atan2(math.sin(br)*math.sin(d)*math.cos(p1), math.cos(d)-math.sin(p1)*math.sin(p2))
    return math.degrees(p2), ((math.degrees(l2)+180)%360)-180


@dataclass
class EpisodeResult:
    reward: float
    success: bool
    failure: str
    steps: int


class OceanNavigationEnv:
    """Single-vessel navigation environment for PPO.

    The agent chooses heading changes. No route polyline is supplied to the policy.
    Land is a hard constraint: a proposed swept movement entering the land buffer
    terminates the episode with -2.
    """
    obs_size = 25
    action_size = len(ACTIONS)

    def __init__(self, seed=1, geography: GlobalGeography | None = None, origin=None, destination=None):
        self.rng = random.Random(seed)
        self.geo = geography or get_global_geography()
        self.ports = load_ports()
        self.weather = WeatherSimulator(seed=seed)
        self.reset(origin, destination)

    def reset(self, origin=None, destination=None, min_route_nm=500, max_route_nm=12000):
        if origin is None or destination is None:
            for _attempt in range(1000):
                origin, destination = self.rng.sample(self.ports, 2)
                d = haversine_nm((origin['lat'],origin['lon']), (destination['lat'],destination['lon']))
                if not min_route_nm <= d <= max_route_nm:
                    continue
                try:
                    start = self.geo.offshore_point(origin, 8)
                    goal = self.geo.offshore_point(destination, 8)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError('Unable to select a valid random voyage from the port catalog.')
        else:
            start = self.geo.offshore_point(origin, 8)
            goal = self.geo.offshore_point(destination, 8)
        self.origin = origin; self.destination = destination
        self.start = start
        self.goal = goal
        self.lat, self.lon = self.start
        self.heading = bearing_deg(self.start, self.goal)
        self.speed = float(SPEEDS[1])
        self.fuel_capacity = max(800.0, haversine_nm(self.start,self.goal)*1.35 + self.rng.uniform(300,900))
        self.fuel = self.fuel_capacity
        self.prev_distance = haversine_nm((self.lat,self.lon), self.goal)
        self.initial_distance = self.prev_distance
        self.steps = 0
        self.max_steps = int(max(100, min(1000, self.prev_distance/4)))
        self.done = False
        self.land_enabled = True
        self.failure = ''
        self.total_reward = 0.0
        return self.observe()

    def _environment(self):
        w = self.weather.at(self.lat, self.lon, self.steps)
        return w.wind_kn, w.wind_dir_deg, w.wave_m, w.current_kn, w.current_dir_deg, w.storm

    def observe(self):
        dist = haversine_nm((self.lat,self.lon), self.goal)
        br = bearing_deg((self.lat,self.lon), self.goal)
        rel = ((br-self.heading+180)%360)-180
        wind, wind_dir, wave, current, current_dir, storm = self._environment()
        rel_wind = ((wind_dir-self.heading+180)%360)-180
        rel_current = ((current_dir-self.heading+180)%360)-180
        land_dirs = []
        for off in (-90,-60,-30,0,30,60,90):
            ang = math.radians(self.heading+off)
            min_d = 100.0
            for d in (5,15,30,50,80,100):
                la = self.lat + (d/111.32)*math.cos(ang)
                lo = self.lon + (d/(111.32*max(.18,math.cos(math.radians(self.lat)))))*math.sin(ang)
                if not self.geo.safe_ocean(la,lo,5):
                    min_d=d; break
            land_dirs.append(min_d/100.0)
        return np.asarray([
            self.lat/90, self.lon/180, math.sin(math.radians(self.heading)), math.cos(math.radians(self.heading)),
            dist/10000, rel/180, self.fuel/self.fuel_capacity, self.speed/24, self.prev_distance/max(1.0,self.initial_distance),
            wind/40, math.sin(math.radians(rel_wind)), math.cos(math.radians(rel_wind)), wave/6,
            current/4, math.sin(math.radians(rel_current)), math.cos(math.radians(rel_current)), storm,
            *land_dirs, self.steps/max(1.0, self.max_steps)
        ], dtype=np.float32)

    def is_action_safe(self, action: int, safety_km: float = 5.0) -> bool:
        action=int(np.clip(action,0,self.action_size-1))
        heading=(self.heading+float(ACTIONS[action]))%360
        new_lat,new_lon=destination_point(self.lat,self.lon,heading,self.speed)
        current=self._environment()[3]; current_dir=self._environment()[4]
        new_lat,new_lon=destination_point(new_lat,new_lon,current_dir,current*0.35)
        return not self.geo.segment_hits_land((self.lat,self.lon),(new_lat,new_lon),safety_km=safety_km)

    def action_mask_fast(self, safety_km: float = 5.0) -> np.ndarray:
        mask=np.ones(self.action_size,dtype=bool)
        current=self._environment()[3]; current_dir=self._environment()[4]
        for a in range(self.action_size):
            heading=(self.heading+float(ACTIONS[a]))%360
            nl,nlo=destination_point(self.lat,self.lon,heading,self.speed)
            nl,nlo=destination_point(nl,nlo,current_dir,current*0.35)
            for t in np.linspace(.2,1.0,5):
                la=self.lat+(nl-self.lat)*float(t); lo=self.lon+((nlo-self.lon+180)%360-180)*float(t)
                if not self.geo.safe_ocean(la,lo,safety_km):
                    mask[a]=False; break
        if not mask.any(): mask[:]=True
        return mask

    def step(self, action: int, hours: float = 1.0):
        hours = max(0.0, float(hours))
        if self.done:
            return self.observe(), 0.0, True, {"failure": self.failure}
        action = int(np.clip(action, 0, self.action_size-1))
        turn = float(ACTIONS[action])
        self.heading = (self.heading + turn) % 360
        self.speed = float(SPEEDS[1])
        if action in (0,6): self.speed = 14.0
        if action in (1,5): self.speed = 18.0
        distance_nm = self.speed * hours
        new_lat, new_lon = destination_point(self.lat,self.lon,self.heading,distance_nm)
        env = self._environment()
        current = env[3]; current_dir = env[4]
        # Add a small environmental drift.
        drift_lat, drift_lon = destination_point(new_lat,new_lon,current_dir,current*0.35)
        new_lat, new_lon = drift_lat, drift_lon
        if self.land_enabled and self.geo.segment_hits_land((self.lat,self.lon),(new_lat,new_lon),safety_km=5.0):
            self.done=True; self.failure='land_collision'; self.total_reward += -2.0
            return self.observe(), -2.0, True, {"failure": self.failure}

        old_distance = self.prev_distance
        self.lat, self.lon = new_lat, new_lon
        new_distance = haversine_nm((self.lat,self.lon), self.goal)
        fuel_rate = (1.6 + (self.speed/24)**2*2.5 + env[2]*0.08 + abs(turn)/35*0.25) * hours
        self.fuel -= fuel_rate
        self.steps += hours
        progress = (old_distance-new_distance)/max(1.0,old_distance)
        reward = float(np.clip(progress*0.7, -0.08, 0.08) - fuel_rate/self.fuel_capacity*0.25)
        clearance = self.geo.distance_to_land_km(self.lat,self.lon)
        if clearance < 25: reward -= (25-clearance)/2500
        if new_distance < 20:
            self.done=True; self.total_reward += 2.0; return self.observe(), 2.0, True, {"success": True}
        if self.fuel <= 0 or self.steps >= self.max_steps:
            self.done=True; self.failure='fuel_exhausted' if self.fuel<=0 else 'time_limit'; self.total_reward += -2.0
            return self.observe(), -2.0, True, {"failure": self.failure}
        self.prev_distance = new_distance
        self.total_reward += reward
        return self.observe(), reward, False, {"distance_nm":new_distance,"fuel":self.fuel,"land_clearance_km":clearance}

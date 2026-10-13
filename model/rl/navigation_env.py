from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from model.navigation.geodata import GlobalGeography, load_ports, get_global_geography
from model.weather.simulator import WeatherSimulator

EARTH_RADIUS_NM = 3440.065
ACTIONS = np.array([-60, -35, -20, -10, 0, 10, 20, 35, 60], dtype=np.float32)
CRUISE_SPEED_KN = 22.0
RECOVERY_ACTION = len(ACTIONS)
POLICY_ACTION_SIZE = RECOVERY_ACTION + 1
NAVIGATION_MODEL_VERSION = "ocean-navigation-v3-nine-course-plus-recovery"


def haversine_nm(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = math.radians(b[0]-a[0]), math.radians(b[1]-a[1])
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*EARTH_RADIUS_NM*math.asin(min(1.0, math.sqrt(h)))


def segment_distance_nm(start, end, point):
    """Approximate the shortest distance from a short course segment to a point."""
    latitude = math.radians(point[0])
    lon_scale = max(0.0, math.cos(latitude))

    def local_xy(position):
        delta_lon = ((position[1] - point[1] + 180.0) % 360.0) - 180.0
        return delta_lon * lon_scale * 60.0, (position[0] - point[0]) * 60.0

    ax, ay = local_xy(start)
    bx, by = local_xy(end)
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    fraction = 0.0 if length_squared <= 1e-12 else max(0.0, min(1.0, -(ax * dx + ay * dy) / length_squared))
    return math.hypot(ax + fraction * dx, ay + fraction * dy)


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
    action_size = POLICY_ACTION_SIZE

    def __init__(self, seed=1, geography: GlobalGeography | None = None, origin=None, destination=None, weather_seed=None, cruise_speed_kn: float = CRUISE_SPEED_KN):
        self.rng = random.Random(seed)
        self.geo = geography or get_global_geography()
        self.ports = load_ports()
        self.cruise_speed_kn = float(np.clip(cruise_speed_kn, 4.0, 35.0))
        self.weather = WeatherSimulator(seed=seed if weather_seed is None else weather_seed)
        # Training/evaluation use voyage-relative time. The live fleet can set a
        # shared simulation clock so every vessel sees the field shown on the globe.
        self.weather_time: float | None = None
        self.reset(origin, destination)

    def reset(self, origin=None, destination=None, min_route_nm=500, max_route_nm=12000):
        if origin is None or destination is None:
            for _attempt in range(1000):
                origin, destination = self.rng.sample(self.ports, 2)
                d = haversine_nm((origin['lat'],origin['lon']), (destination['lat'],destination['lon']))
                if not min_route_nm <= d <= max_route_nm:
                    continue
                try:
                    start = self.geo.offshore_point(origin, 15)
                    goal = self.geo.offshore_point(destination, 8)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError('Unable to select a valid random voyage from the port catalog.')
        else:
            start = self.geo.offshore_point(origin, 15)
            goal = self.geo.offshore_point(destination, 8)
        self.origin = origin; self.destination = destination
        self.start = start
        self.goal = goal
        self.lat, self.lon = self.start
        self.heading = bearing_deg(self.start, self.goal)
        self.speed = self.cruise_speed_kn
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
        self.minimum_clearance_km = self.geo.distance_to_land_km(self.lat,self.lon)
        return self.observe()

    def _environment(self):
        hours = self.steps if self.weather_time is None else self.weather_time
        w = self.weather.at(self.lat, self.lon, hours)
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

    def predict_motion(self, turn_degrees: float, hours: float = 1.0, speed_kn: float | None = None):
        """Return the endpoint for one course/speed command, including current drift."""
        hours = max(0.0, float(hours))
        speed = self.speed if speed_kn is None else max(0.0, float(speed_kn))
        heading = (self.heading + float(turn_degrees)) % 360
        lat, lon = destination_point(self.lat, self.lon, heading, speed * hours)
        current, current_dir = self._environment()[3:5]
        return destination_point(lat, lon, current_dir, current * 0.35 * hours)

    def is_action_safe(self, action: int, safety_km: float = 5.0, hours: float = 1.0) -> bool:
        action = int(action)
        if action == RECOVERY_ACTION:
            return self.safe_recovery_maneuver(hours,safety_km) is not None
        action = int(np.clip(action, 0, len(ACTIONS) - 1))
        endpoint = self.predict_motion(float(ACTIONS[action]), hours, self.cruise_speed_kn)
        return not self.geo.segment_hits_land((self.lat, self.lon), endpoint, safety_km=safety_km)

    def action_mask_fast(self, safety_km: float = 5.0) -> np.ndarray:
        mask=np.ones(len(ACTIONS),dtype=bool)
        for a in range(len(ACTIONS)):
            nl,nlo=self.predict_motion(float(ACTIONS[a]),1.0,self.cruise_speed_kn)
            for t in np.linspace(.2,1.0,5):
                la=self.lat+(nl-self.lat)*float(t); lo=self.lon+((nlo-self.lon+180)%360-180)*float(t)
                if not self.geo.safe_ocean(la,lo,safety_km):
                    mask[a]=False; break
        return mask

    def safe_action_mask(self, safety_km: float = 5.0, hours: float = 1.0) -> np.ndarray:
        """Fast-prefilter actions, then use the vector sweep as final authority."""
        max_dist_km = (self.cruise_speed_kn * float(hours) * 1.852) + safety_km + 20.0
        if self.geo.distance_to_land_km(self.lat, self.lon) > max_dist_km:
            mask = np.zeros(self.action_size, dtype=bool)
            mask[:len(ACTIONS)] = True
            return mask
        fast=self.action_mask_fast(safety_km)
        mask=np.zeros(self.action_size,dtype=bool)
        check=lambda action: self.is_action_safe(int(action),safety_km) if hours==1.0 else self.is_action_safe(int(action),safety_km,hours)
        for action in np.flatnonzero(fast):
            mask[action]=check(action)
        # If the raster prefilter was too conservative, check every heading
        # before declaring that no safe action exists.
        if not mask[:len(ACTIONS)].any() and not fast.all():
            for action in np.flatnonzero(~fast):
                mask[action]=check(action)
        if not mask[:len(ACTIONS)].any() and self.safe_recovery_maneuver(hours,safety_km) is not None:
            mask[RECOVERY_ACTION]=True
        return mask

    def safe_actions(self, safety_km: float = 5.0) -> list[int]:
        return np.flatnonzero(self.safe_action_mask(safety_km)).tolist()

    def hold(self, hours: float = 1.0):
        """Keep the vessel stationary while an escape course is unavailable."""
        hours = max(0.0, float(hours))
        if self.done:
            return self.observe(), 0.0, True, {"failure": self.failure}
        self.speed = 0.0
        fuel_rate = 1.6 * hours
        self.fuel -= fuel_rate
        self.steps += hours
        reward = -0.02 * hours - fuel_rate / max(1.0, self.fuel_capacity) * 0.25
        if self.fuel <= 0 or self.steps >= self.max_steps:
            self.done = True
            self.failure = 'fuel_exhausted' if self.fuel <= 0 else 'time_limit'
            reward -= 2.0
        self.total_reward += reward
        return self.observe(), reward, self.done, {"failure": self.failure if self.done else "", "distance_nm": self.prev_distance, "fuel": self.fuel}

    def safe_recovery_maneuver(self, hours: float = 1.0, safety_km: float = 5.0):
        """Find a slow, wide-turn escape course; every candidate uses the vector shield."""
        hours = max(0.0, float(hours))
        if hours <= 0:
            return None
        old_distance = haversine_nm((self.lat, self.lon), self.goal)
        best = None
        turns = sorted(set(float(x) for x in range(-120, 121, 15)) | set(float(x) for x in ACTIONS))
        for speed in (self.cruise_speed_kn, self.cruise_speed_kn*.75, self.cruise_speed_kn*.55, self.cruise_speed_kn*.35, self.cruise_speed_kn*.2):
            for turn in turns:
                endpoint = self.predict_motion(turn, hours, speed)
                if self.geo.segment_hits_land((self.lat, self.lon), endpoint, safety_km=safety_km):
                    continue
                new_distance = haversine_nm(endpoint, self.goal)
                clearance = self.geo.distance_to_land_km(*endpoint)
                distance_gain = old_distance - new_distance
                score = (
                    distance_gain
                    + min(clearance, 100.0) * 0.18
                    - abs(turn) * 0.035
                    - (self.cruise_speed_kn - speed) * 0.12
                )
                candidate = {
                    "turn_degrees": turn,
                    "speed_kn": speed,
                    "clearance_km": clearance,
                    "distance_gain_nm": distance_gain,
                    "score": score,
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate
        return best

    def step(self, action: int, hours: float = 1.0, *, turn_degrees: float | None = None, speed_kn: float | None = None):
        hours = max(0.0, float(hours))
        if self.done:
            return self.observe(), 0.0, True, {"failure": self.failure}
        recovery = None
        if turn_degrees is None:
            action = int(np.clip(action, 0, self.action_size - 1))
            if action == RECOVERY_ACTION:
                recovery = self.safe_recovery_maneuver(hours,5.0)
                if recovery is None:
                    self.done=True; self.failure='no_safe_action'; self.total_reward-=2.0
                    return self.observe(),-2.0,True,{"failure":self.failure,"distance_nm":self.prev_distance}
                turn=float(recovery["turn_degrees"])
                speed_kn=float(recovery["speed_kn"])
            else:
                turn = float(ACTIONS[action])
        else:
            turn = float(np.clip(turn_degrees, -180.0, 180.0))
        self.heading = (self.heading + turn) % 360
        self.speed = self.cruise_speed_kn if speed_kn is None else max(0.0, float(speed_kn))
        env = self._environment()
        new_lat, new_lon = self.predict_motion(0.0, hours, self.speed)
        if self.land_enabled and self.geo.segment_hits_land((self.lat,self.lon),(new_lat,new_lon),safety_km=5.0):
            self.done=True; self.failure='land_collision'; self.total_reward += -2.0
            return self.observe(), -2.0, True, {"failure": self.failure,"distance_nm":self.prev_distance,"minimum_clearance_km":self.minimum_clearance_km}

        old_position = (self.lat, self.lon)
        old_distance = haversine_nm(old_position, self.goal)
        self.lat, self.lon = new_lat, new_lon
        new_distance = haversine_nm((self.lat,self.lon), self.goal)
        arrival_distance = segment_distance_nm(old_position, (self.lat, self.lon), self.goal)
        fuel_rate = (1.6 + (self.speed/24)**2*2.5 + env[2]*0.08 + min(2.0, abs(turn)/35)*0.25) * hours
        self.fuel -= fuel_rate
        self.steps += hours
        self.prev_distance = new_distance
        progress = (old_distance-new_distance)/max(1.0,old_distance)
        reward = float(np.clip(progress*0.7, -0.08, 0.08) - fuel_rate/self.fuel_capacity*0.25)
        clearance = self.geo.distance_to_land_km(self.lat,self.lon)
        self.minimum_clearance_km=min(self.minimum_clearance_km,clearance)
        if clearance < 25: reward -= (25-clearance)/2500
        recovery_info={"recovery":recovery} if recovery is not None else {}
        if arrival_distance < 20:
            self.done=True; self.total_reward += 2.0; return self.observe(), 2.0, True, {"success": True, "distance_nm": new_distance,"minimum_clearance_km":self.minimum_clearance_km,**recovery_info}
        if self.fuel <= 0 or self.steps >= self.max_steps:
            self.done=True; self.failure='fuel_exhausted' if self.fuel<=0 else 'time_limit'; self.total_reward += -2.0
            return self.observe(), -2.0, True, {"failure": self.failure,"distance_nm":new_distance,"minimum_clearance_km":self.minimum_clearance_km}
        self.total_reward += reward
        return self.observe(), reward, False, {"distance_nm":new_distance,"fuel":self.fuel,"land_clearance_km":clearance,"minimum_clearance_km":self.minimum_clearance_km,**recovery_info}


def bootstrap_heading_action(env: OceanNavigationEnv, safe_actions: list[int]) -> int:
    """Choose the legal course with the best immediate progress and coast margin."""
    if not safe_actions:
        raise ValueError("bootstrap heading selection requires at least one safe action")
    safe_actions=[action for action in safe_actions if action < RECOVERY_ACTION]
    if not safe_actions:
        return RECOVERY_ACTION
    old_distance = haversine_nm((env.lat, env.lon), env.goal)
    best_action = safe_actions[0]
    best_score = -math.inf
    for action in safe_actions:
        endpoint = env.predict_motion(float(ACTIONS[action]), 1.0, env.cruise_speed_kn)
        gain = old_distance - haversine_nm(endpoint, env.goal)
        clearance = env.geo.distance_to_land_km(*endpoint)
        score = gain + min(clearance, 50.0) * 0.08 - abs(float(ACTIONS[action])) * 0.035
        if score > best_score:
            best_action, best_score = action, score
    return int(best_action)

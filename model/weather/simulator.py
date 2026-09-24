from __future__ import annotations

import math
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class WeatherState:
    wind_kn: float
    wind_dir_deg: float
    wave_m: float
    current_kn: float
    current_dir_deg: float
    storm: float
    precipitation: float
    visibility_nm: float
    pressure_hpa: float
    sea_state: int

    def public(self):
        return asdict(self)


class WeatherSimulator:
    """Deterministic, fully offline procedural weather/ocean field.

    The field is smooth in latitude/longitude/time so the RL agent sees spatially
    coherent conditions rather than independent random noise.
    """
    def __init__(self, seed: int = 20260922):
        self.seed = seed

    def at(self, lat: float, lon: float, hours: float = 0.0) -> WeatherState:
        s = self.seed * 0.00013
        p = math.radians(lat * 2.15 + lon * 0.43) + hours * 0.025 + s
        q = math.radians(lon * 1.65 - lat * 0.75) - hours * 0.017 + s * 0.7
        storm = max(0.0, min(1.0, (math.sin(p * 1.9 + math.sin(q)) + 0.55) / 1.55))
        wind = 9.0 + 24.0 * (0.35 + 0.65 * abs(math.sin(p)))
        wind_dir = (125.0 + 80.0 * math.sin(q) + 22.0 * math.sin(p * .7)) % 360.0
        wave = 0.6 + 3.8 * (0.25 + 0.75 * abs(math.cos(q))) * (0.75 + 0.65 * storm)
        current = 0.3 + 2.2 * (0.35 + 0.65 * abs(math.sin(q * .8)))
        current_dir = (wind_dir + 45.0 + 35.0 * math.sin(p * .55)) % 360.0
        precipitation = max(0.0, min(1.0, storm * 0.75 + 0.25 * abs(math.sin(q * 1.7))))
        visibility = max(1.0, 20.0 - precipitation * 15.0 - storm * 5.0)
        pressure = 1018.0 - storm * 30.0 + 5.0 * math.sin(q)
        sea_state = max(0, min(9, int(round(wave * 1.35))))
        return WeatherState(wind, wind_dir, wave, current, current_dir, storm, precipitation, visibility, pressure, sea_state)

    def field(self, hours: float = 0.0, lat_step: float = 8.0, lon_step: float = 10.0):
        points = []
        lat = -75.0
        while lat <= 75.0:
            lon = -180.0
            while lon < 180.0:
                w = self.at(lat, lon, hours)
                points.append({"lat": lat, "lon": lon, **w.public()})
                lon += lon_step
            lat += lat_step
        return points

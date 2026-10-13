from __future__ import annotations

import math
from dataclasses import asdict, dataclass

WEATHER_MODEL_VERSION = "procedural-ocean-v4"

@dataclass(frozen=True)
class WeatherState:
    wind_kn: float
    wind_dir_deg: float
    wave_m: float
    wave_dir_deg: float
    wave_period_s: float
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
        longitude = math.radians(((lon + 180.0) % 360.0) - 180.0)
        # Longitude has no physical meaning at either pole. Taper its influence
        # there so the global field closes smoothly instead of forming a seam.
        longitude_weight = max(0.0, math.cos(math.radians(lat)))
        p = math.radians(lat * 2.15) + 0.55 * longitude_weight * math.sin(longitude) + hours * 0.025 + s
        q = math.radians(-lat * 0.75) + longitude_weight * (1.35 * math.sin(longitude) + 0.42 * math.cos(2.0 * longitude)) - hours * 0.017 + s * 0.7
        storm = max(0.0, min(1.0, (math.sin(p * 1.9 + math.sin(q)) + 0.55) / 1.55))
        wind = 9.0 + 24.0 * (0.35 + 0.65 * abs(math.sin(p)))
        wind_dir = (125.0 + 80.0 * math.sin(q) + 22.0 * math.sin(p * .7)) % 360.0
        wave = 0.6 + 3.8 * (0.25 + 0.75 * abs(math.cos(q))) * (0.75 + 0.65 * storm)
        wave_dir = (wind_dir + 12.0 * math.sin(p + q)) % 360.0
        wave_period = max(3.0, min(18.0, 4.0 + wave * 1.6 + storm * 1.8))
        current = 0.3 + 2.2 * (0.35 + 0.65 * abs(math.sin(q * .8)))
        current_dir = (wind_dir + 45.0 + 35.0 * math.sin(p * .55)) % 360.0
        precipitation = max(0.0, min(1.0, storm * 0.75 + 0.25 * abs(math.sin(q * 1.7))))
        visibility = max(1.0, 20.0 - precipitation * 15.0 - storm * 5.0)
        pressure = 1018.0 - storm * 30.0 + 5.0 * math.sin(q)
        sea_state = max(0, min(9, int(round(wave * 1.35))))
        return WeatherState(wind, wind_dir, wave, wave_dir, wave_period, current, current_dir, storm, precipitation, visibility, pressure, sea_state)

    def field(self, hours: float = 0.0, lat_step: float = 8.0, lon_step: float = 10.0):
        return self.field_with_grid(hours, lat_step, lon_step)[0]

    def field_with_grid(self, hours: float = 0.0, lat_step: float = 8.0, lon_step: float = 10.0):
        """Return a globally wrapped regular grid and its exact sampling geometry."""
        lat_step = max(2.0, float(lat_step))
        lon_step = max(2.0, float(lon_step))
        lat_count = max(2, math.ceil(180.0 / lat_step) + 1)
        lon_count = max(4, math.ceil(360.0 / lon_step))
        actual_lat_step = 180.0 / (lat_count - 1)
        actual_lon_step = 360.0 / lon_count
        points = []
        for row in range(lat_count):
            lat = -90.0 + row * actual_lat_step
            for column in range(lon_count):
                lon = -180.0 + column * actual_lon_step
                w = self.at(lat, lon, hours)
                points.append({"lat": lat, "lon": lon, **w.public()})
        grid = {
            "lat_min": -90.0,
            "lat_max": 90.0,
            "lat_step": actual_lat_step,
            "lat_count": lat_count,
            "lon_min": -180.0,
            "lon_step": actual_lon_step,
            "lon_count": lon_count,
        }
        return points, grid

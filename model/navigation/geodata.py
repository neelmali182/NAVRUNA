from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parents[2]
LAND_PATH = ROOT / "data" / "geography" / "land.geojson"
PORT_PATH = ROOT / "data" / "ports" / "ports.json"


def load_ports() -> list[dict]:
    return json.loads(PORT_PATH.read_text(encoding="utf-8"))


class GlobalGeography:
    """Offline GSHHS coastline + raster safety field.

    The vector geometry is the final collision authority. A 0.5° occupancy grid is
    used for fast RL observations and action screening. The simulator additionally
    performs a continuous swept-segment vector check, so an action cannot tunnel
    through a thin strip of land between two ocean endpoints.
    """

    def __init__(self, land_path: Path = LAND_PATH, grid_step: float = 0.25):
        raw = json.loads(land_path.read_text(encoding="utf-8"))
        self.land = unary_union([shape(f["geometry"]) for f in raw["features"]])
        self.prepared = prep(self.land)
        self.grid_step = float(grid_step)
        self._port_cache: dict[str, tuple[float, float]] = {}
        self._build_grid()

    @staticmethod
    def norm_lon(lon: float) -> float:
        return ((lon + 180.0) % 360.0) - 180.0

    def _build_grid(self):
        step=self.grid_step
        self.lat_axis=np.arange(-90+step/2,90,step,dtype=np.float32)
        self.lon_axis=np.arange(-180+step/2,180,step,dtype=np.float32)
        land=np.zeros((len(self.lat_axis),len(self.lon_axis)),dtype=np.uint8)
        # Vectorized enough for the ~360k-cell grid; use a prepared polygon and only
        # sample cell centers.
        for i,lat in enumerate(self.lat_axis):
            # Fast candidate window using bounds; most cells are ocean.
            for j,lon in enumerate(self.lon_axis):
                if self.prepared.contains(Point(float(lon),float(lat))): land[i,j]=1
        # Distance in cells -> approximate km. Latitude spacing is fixed; longitude
        # distance is corrected locally at query time.
        self.land_grid=land
        self.ocean_distance_cells=distance_transform_edt(1-land)

    def _grid_index(self, lat, lon):
        i=int(np.clip(round((lat+90-self.grid_step/2)/self.grid_step),0,len(self.lat_axis)-1))
        j=int(np.clip(round((self.norm_lon(lon)+180-self.grid_step/2)/self.grid_step),0,len(self.lon_axis)-1))
        return i,j

    def on_land_fast(self,lat,lon):
        i,j=self._grid_index(lat,lon); return bool(self.land_grid[i,j])

    def distance_to_land_km(self, lat: float, lon: float) -> float:
        i,j=self._grid_index(lat,lon)
        cells=float(self.ocean_distance_cells[i,j])
        return cells*self.grid_step*111.32*max(0.18,math.cos(math.radians(lat)))

    def on_land(self, lat: float, lon: float) -> bool:
        return self.prepared.contains(Point(self.norm_lon(lon),lat))

    def vector_distance_to_land_km(self, lat: float, lon: float) -> float:
        ddeg=self.land.distance(Point(self.norm_lon(lon),lat))
        return float(ddeg*111.32*max(.18,math.cos(math.radians(lat))))

    def safe_ocean(self, lat: float, lon: float, safety_km: float = 5.0) -> bool:
        if self.on_land_fast(lat,lon): return False
        return self.distance_to_land_km(lat,lon)>=safety_km

    def segment_hits_land(self,a,b,safety_km=5.0,samples=None):
        lat1,lon1=a; lat2,lon2=b
        dl=((lon2-lon1+180)%360)-180
        span_km=math.hypot((lat2-lat1)*111.32,dl*111.32*max(.18,math.cos(math.radians((lat1+lat2)/2))))
        n=samples or max(12,int(span_km/8))
        for i in range(1,n+1):
            t=i/n; lat=lat1+(lat2-lat1)*t; lon=self.norm_lon(lon1+dl*t)
            if self.on_land(lat,lon) or self.distance_to_land_km(lat,lon)<safety_km: return True
        return False

    def offshore_point(self, port: dict, safety_km: float = 8.0):
        key=port['id']
        if key in self._port_cache: return self._port_cache[key]
        lat0,lon0=port['lat'],port['lon']
        for radius in (5,10,20,35,50,75,100):
            for bearing in np.linspace(0,2*np.pi,48,endpoint=False):
                lat=lat0+(radius/111.32)*math.cos(bearing)
                lon=lon0+(radius/(111.32*max(.18,math.cos(math.radians(lat0)))))*math.sin(bearing)
                lon=self.norm_lon(lon)
                if -85<lat<85 and (not self.on_land(lat,lon)) and self.vector_distance_to_land_km(lat,lon)>=safety_km:
                    self._port_cache[key]=(lat,lon); return lat,lon
        raise ValueError(f'Unable to find a safe offshore point for port {key}.')

    def validate_route(self,points:Iterable[tuple[float,float]],safety_km=5.0):
        pts=list(points); violations=[]
        for idx,p in enumerate(pts):
            if idx in (0,len(pts)-1): continue
            if not self.safe_ocean(*p,safety_km): violations.append({'index':idx,'lat':p[0],'lon':p[1]})
        for idx,(a,b) in enumerate(zip(pts,pts[1:])):
            if self.segment_hits_land(a,b,safety_km): violations.append({'segment':idx})
        return {'valid':not violations,'violations':violations}

from functools import lru_cache

@lru_cache(maxsize=2)
def get_global_geography() -> GlobalGeography:
    return GlobalGeography()

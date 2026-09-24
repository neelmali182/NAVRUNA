import json
import asyncio
from pathlib import Path
import torch

from model.navigation.geodata import get_global_geography
from model.rl.navigation_env import OceanNavigationEnv, destination_point
from model.rl.ppo import PolicyNet
from model.simulation.fleet_simulator import FleetSimulator
from backend.routers import simulation as simulation_router


def test_port_catalog():
    ports=json.loads(Path('data/ports/ports.json').read_text())
    assert len(ports) >= 250
    assert any(p['name']=='Mumbai' for p in ports)
    assert any(p['name']=='Rotterdam' for p in ports)


def test_land_authority_and_ocean():
    geo=get_global_geography()
    assert geo.on_land(28.6,77.2)
    assert not geo.on_land(25.0,55.0)
    assert geo.segment_hits_land((10,75),(20,75),safety_km=0)
    assert not geo.segment_hits_land((15,65),(15,70),safety_km=0)


def test_rl_observation_and_network():
    geo=get_global_geography(); env=OceanNavigationEnv(seed=4,geography=geo)
    obs=env.reset(min_route_nm=300,max_route_nm=1000)
    assert obs.shape==(25,)
    net=PolicyNet(); logits,value=net(torch.tensor(obs).unsqueeze(0))
    assert logits.shape==(1,7); assert value.shape==(1,)


def test_fleet_has_no_preplanned_routes():
    geo=get_global_geography(); sim=FleetSimulator(seed=4)
    data=sim.generate_fleet(8)
    assert len(data['vessels'])==8
    assert all('route' not in v for v in data['vessels'])
    d=sim.step(1)
    assert d['metrics']['collisions']==0
    assert d['tick']['processed_vessels']==8
    assert 0 <= d['tick']['moved_vessels'] <= 8
    pub=sim.vessels[0].public(); assert 'lat' in pub and 'lon' in pub


def test_fractional_step_and_progress_observation():
    sim=FleetSimulator(seed=4)
    sim.generate_fleet(1)
    before=sim.sim_hours
    data=sim.step(1.4)
    assert data['sim_hours']==before+1.4
    env=sim.vessels[0].env
    obs=env.observe()
    assert obs.shape==(25,)
    assert obs[8] != obs[9] or env.steps == 0


def test_completed_voyage_gets_next_destination():
    sim=FleetSimulator(seed=4)
    sim.generate_fleet(1)
    vessel=sim.vessels[0]
    vessel.env.goal=destination_point(vessel.lat,vessel.lon,vessel.heading,0.1)
    vessel.env.prev_distance=0.1
    data=sim.step(1)
    assert data['metrics']['voyages_completed']==1
    assert data['metrics']['success_rate']==100.0
    assert vessel.status=='active'
    assert vessel.progress==0.0
    assert vessel.env.destination['name']==vessel.destination


def test_hold_replan_is_a_failed_voyage():
    sim=FleetSimulator(seed=4)
    sim.generate_fleet(1)
    vessel=sim.vessels[0]
    vessel.env.action_mask_fast=lambda safety_km: __import__('numpy').zeros(vessel.env.action_size,dtype=bool)
    vessel.env.is_action_safe=lambda action,safety_km: False
    result=sim.step(1)
    assert vessel.status=='failed'
    assert vessel.ai_action_name=='HOLD / REPLAN'
    assert vessel.failure_reason=='NO SAFE ACTION'
    assert result['metrics']['voyage_failures']==1


def test_training_submission_is_reserved_before_worker_starts(monkeypatch):
    class FakeFuture:
        pass

    class FakePool:
        def submit(self, *args):
            return FakeFuture()

    previous_status = simulation_router.TRAINER.training_status
    monkeypatch.setattr(simulation_router, 'TRAIN_POOL', FakePool())
    simulation_router.TRAINER.training_status = 'idle'
    request = simulation_router.TrainRequest(steps=1024, envs=8, rollout=32, continuous=False)
    try:
        first = asyncio.run(simulation_router.train_simulation(request))
        second = asyncio.run(simulation_router.train_simulation(request))
    finally:
        simulation_router.TRAINER.training_status = previous_status
    assert first['accepted'] is True
    assert second == {'accepted': False, 'status': 'training', 'message': 'A training run is already active.'}

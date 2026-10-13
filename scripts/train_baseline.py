"""Pre-train and bundle the baseline NAVRUNA PPO neural policy."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time
import torch
from model.rl.ppo import PPOTrainer

def main():
    print("=" * 60)
    print("NAVRUNA v5.0 Baseline Neural Policy Generator")
    print("=" * 60)
    trainer = PPOTrainer(seed=20260922)
    print(f"Device: {trainer.device}")
    
    t0 = time.time()
    # 25 updates * 32 steps * 16 envs = 12,800 transitions
    # Rapid and highly focused convergence
    report = trainer.train(
        total_steps=12800,
        n_envs=16,
        rollout=32,
        lr=3e-4,
        continuous=False
    )
    elapsed = time.time() - t0
    print(f"\n[DONE] Baseline policy trained in {elapsed:.1f}s!")
    print(f"Updates: {report.get('updates')} | Transitions: {report.get('transitions'):,}")
    print(f"Success Rate: {report.get('success_rate', 0.0)*100:.1f}%")
    print(f"Mean Return: {report.get('mean_return', 0.0):.3f}")
    print(f"Saved Checkpoint: {report.get('model_path')}")
    print("Checkpoint Status:", trainer.checkpoint_status)
    print("Trainer trained flag:", trainer.trained)

if __name__ == "__main__":
    main()

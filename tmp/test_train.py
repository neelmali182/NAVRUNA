import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
from model.rl.ppo import PPOTrainer

print("Initializing trainer...")
trainer = PPOTrainer(seed=20260922)
print("Device:", trainer.device)
t0 = time.time()
print("Running rapid training...")
report = trainer.train(total_steps=4096, n_envs=16, rollout=64, lr=3e-4)
print(f"Finished in {time.time()-t0:.2f}s: updates={report.get('updates')}, transitions={report.get('transitions')}")
print("Checkpoint status:", trainer.checkpoint_status)
print("Trained:", trainer.trained)

"""
Independent verification (Claude, fresh session) of M14-Core's evaluation
harness -- NOT copied from tests/test_evaluation_harness.py. Checks:
  1. All 5 methods load and produce physical actions in valid ranges.
  2. TDPK and Greedy reproduce their known M10/M13 behavioral signatures.
  3. The PKTD3-TD action_fn's unnormalize_action() wrapping is actually
     correct -- not just "in range" but matching what select_action's raw
     normalized output SHOULD map to, computed independently by hand here.
  4. run_batch() caching actually avoids re-simulation on second call.
"""
import math
import time
import numpy as np

from uav_trajectory_rl.evaluation.harness import get_method_specs, run_episode, run_batch
from uav_trajectory_rl.config import ACTION_CLIP_C, V_MAX

specs = get_method_specs(device="cpu", k=10)
print("Loaded methods:", list(specs.keys()))
assert set(specs.keys()) == {"TDPK", "Greedy", "DuelingDQL", "PPO", "PKTD3-TD"}

# --- 1. Action range validity for all 5, using a real env state ---
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv
env = UAVTrajectoryEnv(k=10, rng=np.random.default_rng(0))
state = env.reset()

for name, spec in specs.items():
    v, lam, rho = spec.action_fn(state, env)
    ok = (0.0 <= v <= V_MAX + 1e-6) and (0.0 - 1e-5 <= lam <= math.pi + 1e-5) and (-math.pi - 1e-5 <= rho <= math.pi + 1e-5)
    print(f"{name:12s} action=({v:.3f}, {lam:.3f}, {rho:.3f})  valid_range={ok}")
    assert ok, f"{name} produced an out-of-range action"

# --- 2. TDPK and Greedy known behavior ---
tdpk_log = run_episode(specs["TDPK"], seed=0, k=10)
print(f"\nTDPK: arrived={tdpk_log.arrived} steps={tdpk_log.steps_taken}")
assert tdpk_log.arrived, "TDPK should arrive (M10 100% arrival baseline)"
assert 60 <= tdpk_log.steps_taken <= 120, f"TDPK step count {tdpk_log.steps_taken} outside expected ~89-step ballpark"

greedy_log = run_episode(specs["Greedy"], seed=0, k=10)
print(f"Greedy: arrived={greedy_log.arrived} steps={greedy_log.steps_taken}")
assert greedy_log.steps_taken == 200, "Greedy should run the full 200 steps (arrives at forced final step)"

# --- 3. Independently re-derive the PKTD3-TD unnormalize mapping by hand ---
from uav_trajectory_rl.td3_agent import TD3Agent
td3 = TD3Agent(state_dim=env.state_dim, device="cpu")
td3.load("checkpoints/run4/td3_agent_final.pt")
raw = td3.select_action(state)
print(f"\nPKTD3-TD raw actor output (should be in [-c,c]^3): {raw}")
c = ACTION_CLIP_C
v_manual = (raw[0] + c) / (2.0 * c) * V_MAX
lam_manual = (raw[1] + c) / (2.0 * c) * math.pi
rho_manual = raw[2] * (math.pi / c)
v_wrapped, lam_wrapped, rho_wrapped = specs["PKTD3-TD"].action_fn(state, env)
print(f"Manual unnormalize:  v={v_manual:.6f} lam={lam_manual:.6f} rho={rho_manual:.6f}")
print(f"Harness action_fn:   v={v_wrapped:.6f} lam={lam_wrapped:.6f} rho={rho_wrapped:.6f}")
assert abs(v_manual - v_wrapped) < 1e-4
assert abs(lam_manual - lam_wrapped) < 1e-4
assert abs(rho_manual - rho_wrapped) < 1e-4
print("PKTD3-TD unnormalize_action wrapping independently confirmed correct.")

# --- 4. Cache actually skips re-simulation ---
import shutil
cache_dir = "/tmp/m14_cache_verify"
shutil.rmtree(cache_dir, ignore_errors=True)

t0 = time.time()
run_batch("TDPK", seeds=[0, 1, 2], cache_dir=cache_dir)
t_first = time.time() - t0

t0 = time.time()
run_batch("TDPK", seeds=[0, 1, 2], cache_dir=cache_dir)
t_second = time.time() - t0

print(f"\nrun_batch first call:  {t_first:.3f}s")
print(f"run_batch second call (cached): {t_second:.3f}s")
assert t_second < t_first / 3, "Second call should be much faster if actually reading from cache"

print("\nALL INDEPENDENT CHECKS PASSED.")

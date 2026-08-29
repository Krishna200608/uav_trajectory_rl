"""
Direct numerical inspection of actor output and critic Q1-values / gradients at Q_START.

Steps:
0. Sanity check sample_terminal_weighted batch composition.
1. Direct actor output at Q_START across training checkpoints.
2. Critic Q1 values for four concrete reference actions at Q_START:
    A. Prior knowledge toward goal (v=20, lam=pi/2, rho=pi/4)
    B. Hover (v=0, lam=pi/2, rho=0)
    C. Into the wall (v=20, lam=pi/2, rho=-3*pi/4)
    D. Actor's own current output
3. The actual gradient the actor is trained on: grad_a Q1(s0, a) at actor's action.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from uav_trajectory_rl.config import Q_START, Q_END, V_MAX
from uav_trajectory_rl.mdp_environment import UAVTrajectoryEnv
from uav_trajectory_rl.prior_knowledge_policy import normalize_action, unnormalize_action
from uav_trajectory_rl.td3_networks import Actor, TwinCritic, ReplayBuffer


def run_step0_sanity_check():
    print("=" * 80)
    print("STEP 0 — Sanity Check: ReplayBuffer and sample_terminal_weighted")
    print("=" * 80)
    # Simulate a buffer with 2 arrived episodes (89 steps each) and 1 non-arrived (200 steps)
    buf = ReplayBuffer(state_dim=26, action_dim=3, capacity=1000)
    rng = np.random.default_rng(0)

    # Episode 1: arrived, 90 steps
    for i in range(90):
        # state has UAV pos moving from (0,0,50) to (800,800,100)
        uav_pos = np.array([i * 8.8, i * 8.8, 50.0 + i * 0.5])
        s = np.zeros(26, dtype=np.float32)
        s[0:3] = uav_pos / 800.0  # mock state
        steps_from_term = 90 - 1 - i
        buf.add(
            state=s,
            action=np.zeros(3, dtype=np.float32),
            reward=1.0 if steps_from_term > 0 else 20.0,
            next_state=s,
            done=(i == 89),
            arrived=True,
            steps_from_terminal=steps_from_term,
        )

    # Episode 2: non-arrived, 200 steps (stuck at corner)
    for i in range(200):
        s = np.zeros(26, dtype=np.float32)
        s[0:3] = np.array([0.0, 0.0, 50.0]) / 800.0
        steps_from_term = 200 - 1 - i
        buf.add(
            state=s,
            action=np.zeros(3, dtype=np.float32),
            reward=-1.0,
            next_state=s,
            done=(i == 199),
            arrived=False,
            steps_from_terminal=steps_from_term,
        )

    print(f"Buffer populated: total size = {buf.size}")
    arrived_count = int(np.sum(buf.arrived[:buf.size]))
    non_arrived_count = int(np.sum(~buf.arrived[:buf.size]))
    term_window_count = int(np.sum(buf.arrived[:buf.size] & (buf.steps_from_terminal[:buf.size] < 15)))
    print(f"  Arrived transitions: {arrived_count}")
    print(f"  Non-arrived transitions: {non_arrived_count}")
    print(f"  Arrived & steps_from_terminal < 15: {term_window_count}")

    # Sample 3 batches
    for b in range(3):
        states, actions, rewards, next_states, dones = buf.sample_terminal_weighted(
            batch_size=10,
            arrived_fraction=0.3,
            terminal_window=15,
            rng=rng,
        )
        print(f"Batch {b+1} sampled (size 10, arrived_fraction=0.3):")
        # Check how many arrived
        # We can detect by checking if state[0] > 0.8 * 8.8 * 75 / 800
        # In our mock, uav_pos x for last 15 steps is (75..89)*8.8/800 = 0.825..0.979
        term_arrivals = np.sum(states[:, 0] >= (75 * 8.8 / 800.0) - 1e-4)
        corner_non_arrivals = np.sum(states[:, 0] == 0.0)
        print(f"  Transitions with terminal arrival pos: {term_arrivals}")
        print(f"  Transitions with corner/non-arrived pos: {corner_non_arrivals}")

    print("Sanity check completed.\n")


def inspect_checkpoints(ckpt_dir_str: str = "checkpoints/diag_rrand60k"):
    print("=" * 80)
    print(f"INSPECTION OF CHECKPOINTS IN: {ckpt_dir_str}")
    print("=" * 80)

    # 1. Reproducible initial state s0
    env = UAVTrajectoryEnv(k=10, rng=np.random.default_rng(0))
    s0 = env.reset()
    state_dim = env.state_dim  # 26
    action_dim = 3

    print("\nInitial State s0 = env.reset() (seed 0, k=10):")
    print(f"  Shape: {s0.shape}")
    print(f"  UAV Pos: {env.uav_pos}")
    print(f"  Vector elements: {s0.round(4).tolist()}")

    # Reference physical actions
    # Action A: Prior knowledge toward goal: v=20, lam=pi/2, rho=pi/4
    act_phys_A = (20.0, float(np.pi / 2.0), float(np.pi / 4.0))
    # Action B: Hover: v=0, lam=pi/2, rho=0
    act_phys_B = (0.0, float(np.pi / 2.0), 0.0)
    # Action C: Into the wall: v=20, lam=pi/2, rho=-3*pi/4
    act_phys_C = (20.0, float(np.pi / 2.0), float(-3.0 * np.pi / 4.0))

    act_norm_A = np.array(normalize_action(act_phys_A), dtype=np.float32)
    act_norm_B = np.array(normalize_action(act_phys_B), dtype=np.float32)
    act_norm_C = np.array(normalize_action(act_phys_C), dtype=np.float32)

    print("\nReference Actions (Physical -> Normalized [-1, 1]):")
    print(f"  A [PK toward goal] : phys = (v=20 m/s, lam=90 deg, rho=+45 deg) -> norm = {act_norm_A.round(4).tolist()}")
    print(f"  B [Hover]          : phys = (v= 0 m/s, lam=90 deg, rho=  0 deg) -> norm = {act_norm_B.round(4).tolist()}")
    print(f"  C [Into the wall]  : phys = (v=20 m/s, lam=90 deg, rho=-135 deg)-> norm = {act_norm_C.round(4).tolist()}")

    ckpt_dir = Path(ckpt_dir_str)
    ckpts = ["td3_agent_ep200.pt", "td3_agent_ep400.pt", "td3_agent_ep600.pt", "td3_agent_final.pt"]

    s0_t = torch.as_tensor(s0, dtype=torch.float32).unsqueeze(0)
    act_t_A = torch.as_tensor(act_norm_A, dtype=torch.float32).unsqueeze(0)
    act_t_B = torch.as_tensor(act_norm_B, dtype=torch.float32).unsqueeze(0)
    act_t_C = torch.as_tensor(act_norm_C, dtype=torch.float32).unsqueeze(0)

    rows = []

    for ckpt_name in ckpts:
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"Missing {ckpt_path}")
            continue

        data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=1.0)
        actor.load_state_dict(data["actor"])
        actor.eval()

        critic = TwinCritic(state_dim=state_dim, action_dim=action_dim)
        critic.load_state_dict(data["critic"])
        critic.eval()

        # Step 1: Actor raw output at s0
        with torch.no_grad():
            act_norm_D = actor(s0_t).squeeze(0).numpy()
        v_phys_D, lam_phys_D, rho_phys_D = unnormalize_action(act_norm_D)
        act_phys_D = (v_phys_D, lam_phys_D, rho_phys_D)

        # Step 2: Critic Q1 values for A, B, C, D
        act_t_D = torch.as_tensor(act_norm_D, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q1_A = critic.q1_forward(s0_t, act_t_A).item()
            q1_B = critic.q1_forward(s0_t, act_t_B).item()
            q1_C = critic.q1_forward(s0_t, act_t_C).item()
            q1_D = critic.q1_forward(s0_t, act_t_D).item()

        # Step 3: Gradient d Q1 / d a at actor's own current output
        a_var = torch.tensor(act_norm_D, dtype=torch.float32).unsqueeze(0).detach().requires_grad_(True)
        q1_for_grad = critic.q1_forward(s0_t, a_var)
        q1_for_grad.backward()
        grad_a = a_var.grad.squeeze(0).numpy()  # [d Q1 / d v_norm, d Q1 / d lam_norm, d Q1 / d rho_norm]

        rows.append({
            "ckpt": ckpt_name,
            "act_norm_D": act_norm_D,
            "act_phys_D": act_phys_D,
            "q1_A": q1_A,
            "q1_B": q1_B,
            "q1_C": q1_C,
            "q1_D": q1_D,
            "grad_a": grad_a,
        })

    print("\n" + "=" * 100)
    print("STEP 1 & 2: ACTOR OUTPUT & CRITIC Q1 VALUES AT Q_START")
    print("=" * 100)
    print(f"{'Checkpoint':<20} | {'Actor Output (norm)':<28} | {'Actor Action (phys)':<35} | {'Q1(A:PK)':<9} | {'Q1(B:Hov)':<9} | {'Q1(C:Wall)':<10} | {'Q1(D:Actor)':<10}")
    print("-" * 140)
    for r in rows:
        norm_str = f"[{r['act_norm_D'][0]:+.3f}, {r['act_norm_D'][1]:+.3f}, {r['act_norm_D'][2]:+.3f}]"
        phys_str = f"v={r['act_phys_D'][0]:5.1f}m/s, lam={np.degrees(r['act_phys_D'][1]):+5.1f}°, rho={np.degrees(r['act_phys_D'][2]):+6.1f}°"
        print(f"{r['ckpt']:<20} | {norm_str:<28} | {phys_str:<35} | {r['q1_A']:+8.2f} | {r['q1_B']:+8.2f} | {r['q1_C']:+9.2f}  | {r['q1_D']:+9.2f}")

    print("\n" + "=" * 100)
    print("STEP 3: ACTOR GRADIENT grad_a Q1(s0, a) EVALUATED AT ACTOR'S ACTION")
    print("=" * 100)
    print(f"{'Checkpoint':<20} | {'d Q1 / d v_norm':<16} | {'d Q1 / d lam_norm':<18} | {'d Q1 / d rho_norm':<18} | {'Interpretation'}")
    print("-" * 100)
    for r in rows:
        g = r['grad_a']
        interp = "Wants MORE speed" if g[0] > 0 else "Wants LESS speed (pushing to -1)"
        print(f"{r['ckpt']:<20} | {g[0]:+16.4f} | {g[1]:+18.4f} | {g[2]:+18.4f} | {interp}")

    print("\nDone.")


if __name__ == "__main__":
    if Path("checkpoints/diag_annealed_handoff").exists():
        inspect_checkpoints("checkpoints/diag_annealed_handoff")
    inspect_checkpoints("checkpoints/diag_rrand60k")
    if Path("checkpoints/diag_channelfix").exists():
        inspect_checkpoints("checkpoints/diag_channelfix")
    if Path("checkpoints/run3").exists():
        inspect_checkpoints("checkpoints/run3")

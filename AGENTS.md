# AGENTS.md — AI Agent Guidelines & Context for PKTD3-TD

## 1. PROJECT IDENTITY

This repository reproduces the PKTD3-TD (Prior-Knowledge-guided Twin Delayed Deep Deterministic Policy Gradient in Three Dimensions) algorithm from M. Li, M. Dong, H. Wang, and H. Wang, *"3-D Trajectory Design Based on Deep Reinforcement Learning for UAV-Assisted Communication Networks"*, IEEE Transactions on Network Science and Engineering, vol. 13, 2026. This project is developed for the Data Management in Mobile and Sensor Networks (DMMSN) coursework at the Indian Institute of Information Technology (IIIT) Allahabad.  
- **Student Team (Group 5):** Krishna Sikheriya, Kushagra Gupta, Aditya Pankaj Sharma, Vaibhav Sharma  
- **Supervisor:** Nayanjit Talukdar (PhD Scholar, WSN Lab, Department of IT, IIIT Allahabad)

## 2. MANDATORY READING ORDER FOR ANY AGENT STARTING WORK

Before making ANY change, execute this sequence in order:
1. `git pull origin main` — never work from a stale local copy.
2. Read this file (`AGENTS.md`) completely.
3. Read `docs/PKTD3-TD_Tracker.md` completely — this is the single source of truth for every equation, parameter value, paper-vs-code discrepancy, and design decision made so far. If something in this codebase seems to contradict the paper, the tracker's "Review notes" section almost certainly already explains why — check there before "fixing" anything.
4. Read `README.md`'s "Implementation status" section for the current module completion state.

## 3. ARCHITECTURE / DATA FLOW

The system is structured as modular, independent mathematical and physical building blocks composed into a standard reinforcement learning pipeline:

- `config.py` (constants): System-wide physical parameters, simulation geometry, wireless channel metrics, and TD3 hyperparameters from Tables II & III. Imported by all modules.
- `uav_kinematics.py`, `user_mobility.py`, `channel_model.py`, `energy_model.py`: Independent physics primitives with zero cross-dependencies on each other.
  - `uav_kinematics.py`: Implements 3D spherical coordinate kinematics and acceleration limits (eq. 1–3).
  - `user_mobility.py`: Implements 2D Gauss-Markov ground user mobility dynamics and boundary reflections (eq. 4–7).
  - `channel_model.py`: Calculates air-to-ground LoS/NLoS probabilities, path losses, and Shannon transmission throughput (eq. 8–14).
  - `energy_model.py`: Calculates rotary-wing aerodynamic propulsion power and energy consumption with flight-envelope pitch bounding (eq. 15–16).
- `mdp_environment.py` (`UAVTrajectoryEnv`): Composes all four physics primitives above into a Gym-like `reset()` / `step()` environment implementing the continuous state observation (eq. 19), action mapping (eq. 20), and full 6-term composite reward function (eq. 21–29).
- `prior_knowledge_policy.py` (`select_action`): The exploration dispatcher used **DURING TRAINING ONLY**. Switches between heuristic prior-knowledge actions (fixed horizontal polar angle $\lambda = 0.5\pi$ and random azimuth cone $[0, 0.5\pi]$) and noised, tanh-clipped actor network actions based on replay buffer fill level ($R_{\text{ex}}$ vs. $R_{\text{rand}}$).
- `td3_networks.py`: Holds the pure `nn.Module` neural architectures (`Actor`, `TwinCritic`) and the circular NumPy `ReplayBuffer`. Pure data structures and forward inference; contains no training logic.
- `td3_agent.py` (`TD3Agent`): Wraps the actor-critic networks with target copies, Adam optimizers, and TD3 learning math (clipped double-Q Bellman targets, target policy smoothing regularization, delayed policy updates, and Polyak soft updates; eq. 32–38).
- `scripts/train.py`: Wires all of the above into the complete Algorithm 1 (Lines 1–32) episodic training loop with periodic checkpointing and reward logging.
- `src/uav_trajectory_rl/baselines/`: Holds standalone benchmark comparison policies evaluated under the identical MDP environment.
  - `tdpk.py`: Implements the Trajectory Design based on Prior Knowledge (TDPK) baseline (eq. from Section V-A: direct-to-destination 3D spherical direction with uniform random speed).
  - Baselines Dueling DQL (`M11`), PPO (`M12`), and Greedy (`M13`) will reside here. These policies do NOT use the TD3 agent; they are independent heuristics/agents evaluated identically at test time.

## 4. CONVENTIONS ESTABLISHED IN THIS CODEBASE

Follow these principles without exception:
- **Paper-Grounded Equations:** Every equation implemented must be grounded in the actual typeset paper, not reconstructed from general RL or wireless communications textbook knowledge. When in doubt, the literal typeset equation in M. Li et al. (2026) wins, even if it looks unusual (refer to `docs/PKTD3-TD_Tracker.md`'s "Review notes" for concrete examples where textbook assumptions like Shannon $\log_2(1 + \text{SNR})$ were proven wrong for this paper).
- **Explicit Assumptions & Design Decisions:** Any value or behavior not specified by the paper must be implemented as an explicit, clearly documented `ASSUMPTION` or `DESIGN DECISION` in comments — never silently guessed. Add a corresponding entry to `docs/PKTD3-TD_Tracker.md`'s "Review notes".
- **Mirrored Unit Testing:** Tests live in `tests/`, one file per `src/uav_trajectory_rl/` module, mirroring its name (e.g., `channel_model.py` $\to$ `test_channel_model.py`). Run the full test suite (`pytest tests/ -v`) before every commit — never commit with failing tests.
- **Trained Artifacts & Checkpoints:** Large binary artifacts (trained `.pt` model checkpoints, `.npy` reward logs, convergence plots) ARE committed to this repository intentionally under `checkpoints/` (overriding default ignore rules) to preserve experiment records.
- **Synchronized Documentation:** Every commit that completes or fixes a module must update BOTH `docs/PKTD3-TD_Tracker.md`'s module tracker table AND `README.md`'s roadmap checklist + detailed table in the same commit. `AGENTS.md`, `PKTD3-TD_Tracker.md`, and `README.md` must never drift out of sync.
- **Always Push:** `git push origin main` is the mandatory last step of every task. Never leave work committed only locally.

## 5. CURRENT STATE SNAPSHOT

As of the latest repository commits:
- **Completed Modules (M0–M10):** All core environment primitives, TD3 networks, TD3 agent, prior-knowledge policy, training loop (Algorithm 1), and the first evaluation baseline (TDPK direct-to-destination heuristic) are complete, reviewed, and approved with 40 passing unit tests.
- **Trained Model Artifacts (`checkpoints/run1/` & `checkpoints/run2/`):** Both full 6,000-episode runs are **INVALID** and documented as failed records. Root causes for both are now fully understood and resolved: (1) aerodynamic singularity at pitch extremes bounded, (2) unnormalized state coordinates causing tanh saturation normalized, and (3) action-scale mismatch between replay buffer (was physical $[0, 20]$) and network interfaces (normalized $[-1, 1]$) resolved via `normalize_action()`. An 800-episode diagnostic confirmed that the critic now exhibits monotonic $Q_1$ growth with speed toward the goal and the actor produces active displacement (up to 142m in rollouts).
- **Current Development Focus:** A new, clean full 6,000-episode training run (`checkpoints/run3`) on Google Colab T4 GPU to produce the final, healthy model artifact for baseline comparisons and M14 evaluation.

## 6. ENVIRONMENT SETUP QUICK REFERENCE

To configure and verify the environment from scratch:

```bash
# 1. Create and activate a clean virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# 2. Install editable package with development dependencies
pip install -e ".[dev]"

# 3. Run complete test suite to confirm integrity
pytest tests/ -v
```

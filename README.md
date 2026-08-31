# PKTD3-TD: 3-D UAV Trajectory Design via Deep Reinforcement Learning

*A from-scratch reproduction of a prior-knowledge-guided TD3 algorithm for UAV-assisted communication networks.*

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-in%20development-yellow)
![License](https://img.shields.io/badge/license-academic-lightgrey)

## Overview

In wireless mobile networks, an unmanned aerial vehicle (UAV) can serve as an aerial base station to collect data from moving ground users. Operating in a three-dimensional service area, the UAV must decide its flight speed and direction each second to maximize data throughput while minimizing battery propulsion energy, all while respecting physical acceleration, altitude boundaries, and total mission time limits.

This repository implements the PKTD3-TD algorithm, which models the trajectory optimization problem as a continuous Markov Decision Process (MDP) and solves it using Twin Delayed Deep Deterministic Policy Gradient (TD3) reinforcement learning. Rather than relying entirely on uniform random exploration during early training, the algorithm incorporates a heuristic prior-knowledge exploration policy that biases early flights toward the destination, stabilizing training and improving sample efficiency.

Developed as a coursework project for Data Management in Mobile and Sensor Networks (DMMSN) at IIIT Allahabad, this codebase focuses on mathematical fidelity to the source formulation, modular component boundaries, and comprehensive unit testing.

## Table of contents

- [Overview](#overview)
- [Paper and citation](#paper-and-citation)
- [Team and supervision](#team-and-supervision)
- [Quickstart](#quickstart)
- [Project structure](#project-structure)
- [Implementation status](#implementation-status)
- [Tech stack](#tech-stack)
- [Contributing and acknowledgments](#contributing-and-acknowledgments)

## Paper and citation

This project reproduces the system model, MDP formulation, and learning algorithm presented in:

```text
M. Li, M. Dong, H. Wang, and H. Wang, "3-D Trajectory Design Based on
Deep Reinforcement Learning for UAV-Assisted Communication Networks,"
IEEE Transactions on Network Science and Engineering, vol. 13, 2026.
```

This repository is an independent academic reproduction created for educational purposes and is not affiliated with or endorsed by the original authors.

## Team and supervision

- **Student Team (Group 5):**
  - Krishna Sikheriya
  - Kushagra Gupta
  - Aditya Pankaj Sharma
  - Vaibhav Sharma
- **Supervisor:**
  - Nayanjit Talukdar (PhD Scholar, WSN Lab, Department of IT, IIIT Allahabad)

## Quickstart

### Environment setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Krishna200608/uav_trajectory_rl.git
   cd uav_trajectory_rl
   ```

2. Create and activate a virtual environment:

   - Linux / macOS:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Running tests

Run the complete test suite:
```bash
pytest tests/ -v
```

## Project structure

```text
uav_trajectory_rl/
|-- .gitignore
|-- README.md
|-- pyproject.toml
|-- checkpoints/
|   |-- run1/                          # INVALID -- actor saturation, see docs/PKTD3-TD_Tracker.md
|   |-- run2/                          # INVALID -- stand-still collapse (v=0), see docs/PKTD3-TD_Tracker.md
|   |-- run3/                          # Run 3: improved, but unvalidated (0% arrival rate on noise sweep)
|   \-- run4/                          # Run 4: full 6,000-ep annealed run (decisive 0% arrival confirmation)
|-- docs/
|   \-- PKTD3-TD_Tracker.md             # Ground-truth parameters, assumptions, and review notes
|-- notebooks/
|   |-- train_colab.ipynb               # Google Colab T4 training notebook with Drive backup & GitHub push (PKTD3-TD)
|   |-- train_dueling_dql_colab.ipynb   # Google Colab T4 training notebook with Drive backup & GitHub push (Dueling DQL)
|   |-- train_ppo_colab.ipynb           # Google Colab T4 training notebook with Drive backup & GitHub push (PPO)
|   \-- check_actor_saturation_colab.ipynb # Live diagnostic monitor to check actor saturation on Drive
|-- scripts/
|   |-- .gitkeep                        # Execution and training scripts
|   |-- diagnose_displacement.py        # Diagnostic script to test rollout displacement & critic gradients
|   |-- diagnose_noise_sensitivity.py   # Diagnostic sweep of evaluation noise levels on Run 3 checkpoints
|   \-- train.py                        # Full PKTD3-TD training loop (Algorithm 1)
|-- src/
|   \-- uav_trajectory_rl/
|       |-- __init__.py                 # Package root
|       |-- baselines/                  # Evaluation baseline algorithms (M10-M13)
|       |   |-- __init__.py
|       |   |-- dueling_dql.py              # Dueling DQL discrete Q-learning baseline (M11)
|       |   |-- greedy.py                   # Greedy myopic lookahead baseline (M13)
|       |   |-- ppo.py                      # PPO Gaussian-policy continuous baseline (M12)
|       |   \-- tdpk.py                 # Prior-knowledge direct flight baseline (M10)
|       |-- channel_model.py            # LoS/NLoS path loss and Shannon rate (eq. 8-14)
|       |-- config.py                   # System constants, physical bounds, and RL parameters
|       |-- energy_model.py             # Rotary-wing propulsion power and energy (eq. 15-16)
│       ├── mdp_environment.py          # Continuous MDP environment and 6 reward terms (eq. 17-29)
│       ├── prior_knowledge_policy.py   # Heuristic PK generator and action dispatcher (eq. 30-31)
│       ├── td3_agent.py                # TD3 agent with clipped double-Q and delayed updates (eq. 32-38)
│       ├── td3_networks.py             # TD3 actor, twin-critic networks, and replay buffer (eq. 32-38)
│       ├── uav_kinematics.py           # 3D spherical kinematics and acceleration capping (eq. 1-3)
│       └── user_mobility.py            # Gauss-Markov ground user swarm mobility (eq. 4-7)
└── tests/
    ├── test_baselines_greedy.py        # Greedy candidate lookahead and env-immutability tests
    ├── test_baselines_tdpk.py          # TDPK direct-flight geometry and evaluation tests
    ├── test_channel_model.py           # Channel attenuation and transmission rate tests
    ├── test_energy_model.py            # Power components and hovering energy tests
    ├── test_mdp_environment.py         # State dimensionality and MDP transition tests
    ├── test_prior_knowledge_policy.py  # Action generation, un-normalization, and dispatch tests
    ├── test_ppo.py                      # PPO Gaussian policy, GAE hand-verification, clipped surrogate tests
    ├── test_td3_agent.py               # Target computation, delayed updates, and checkpointing tests
    ├── test_td3_networks.py            # Actor bounds, twin-critic equality, and buffer overwrite tests
    ├── test_train_smoke.py             # End-to-end smoke test for training loop execution
    ├── test_uav_kinematics.py          # Spherical motion updates and acceleration limits tests
    └── test_user_mobility.py           # User swarm bounds and Gauss-Markov step tests
```

## Implementation status

> **Note on Training Runs (Runs 1–4) & Investigation Closure:** Across four full 6,000-episode training runs (`checkpoints/run1/` through `run4/`) and 10 systematic diagnostic rounds, full deterministic convergence was **NOT achieved** (0.0% arrival rate across 720 evaluation trials on Run 4, ending in corner boundary lock-in). Direct numerical inspection identified the root cause: at the boundary-heavy start state $Q_{\text{START}} = (0, 0, 50)$, 87.5% of direction space leads to immediate boundary cancellation, producing an undifferentiated, flat critic value surface (spread $< 4\%$ between goal departure and wall collision) that deterministic policy gradient cannot reliably escape under the tested training budget. The training-instability investigation is officially **CLOSED**. All checkpoints are preserved as empirical records but are **NOT** used for downstream baseline comparisons. See [docs/PKTD3-TD_Tracker.md](docs/PKTD3-TD_Tracker.md) for the full 10-round investigation report.

### Module roadmap

- [x] M0 -- Shared config and constants
- [x] M1 -- UAV kinematics (eq. 1-3)
- [x] M2 -- User mobility -- Gauss-Markov (eq. 4-7)
- [x] M3 -- Channel model -- LoS, path loss, rate (eq. 8-14)
- [x] M4 -- UAV energy and propulsion model (eq. 15-16)
- [x] M5 -- MDP env wrapper -- state, action, 6-term reward, step (eq. 17-29)
- [x] M6 -- Prior-knowledge exploration policy (eq. 30-31)
- [x] M7 -- TD3 networks and replay buffer
- [x] M8 -- TD3 update rules (eq. 32-38)
- [x] M9 -- Training loop (Algorithm 1; component-verified, convergence NOT achieved across runs 1–4; investigation closed, see docs/PKTD3-TD_Tracker.md)
- [x] M10 -- Baseline: TDPK
- [x] M11 -- Baseline: Dueling DQL
- [x] M12 -- Baseline: PPO (Gaussian policy, GAE-Lambda, PPO-Clip; implemented -- pending review)
- [x] M13 -- Baseline: Greedy (200-candidate one-step lookahead, deep copy; implemented -- pending review)
- [ ] M14 -- Evaluation and plotting suite (Figs. 4-12, Tables IV-VI)

### Detailed module tracking

| ID | File / Module | Paper Scope | Status |
|---|---|---|---|
| M0 | `config.py` | Tables II, III and Sec. V-A constants | Done (reviewed, approved) |
| M1 | `uav_kinematics.py` | Kinematics and acceleration limits (eq. 1-3) | Done (reviewed, approved) |
| M2 | `user_mobility.py` | Gauss-Markov user mobility (eq. 4-7) | Done (reviewed, approved) |
| M3 | `channel_model.py` | LoS probability, path loss, and rate (eq. 8-14) | Done (reviewed, approved) |
| M4 | `energy_model.py` | Rotary-wing propulsion energy (eq. 15-16) | Done (reviewed, approved) |
| M5 | `mdp_environment.py` | Full MDP env and 6 reward terms (eq. 17-29) | Done (reviewed, approved) |
| M6 | `prior_knowledge_policy.py` | PK guidance and action dispatch (eq. 30-31) | Done (reviewed, approved) |
| M7 | `td3_networks.py` | Actor-critic networks and replay buffer | Done (reviewed, approved) |
| M8 | `td3_agent.py` | Clipped double-Q and target smoothing (eq. 32-38) | Done (reviewed, approved) |
| M9 | `scripts/train.py` | Training loop (Algorithm 1) | Implemented & component-verified (M0–M8 hand-verified); full convergence NOT achieved across runs 1–4 (flat value surface at Q_START); investigation closed, see docs/PKTD3-TD_Tracker.md |
| M10 | `baselines/tdpk.py` | Baseline: TDPK (direct-to-destination flight) | Done (reviewed, approved) |
| M11 | `baselines/dueling_dql.py` | Baseline: Dueling DQL (discrete 200-action Q-learning) | Implemented -- pending review (60 unit tests passing) |
| M12 | `baselines/ppo.py` | Baseline: PPO (Gaussian policy, GAE-Lambda, PPO-Clip, rollout-based) | Implemented -- pending review (60 unit tests passing, 800-ep diagnostic done) |
| M13 | `baselines/greedy.py` | Baseline: Greedy (one-step lookahead over 200 discrete actions) | Implemented -- pending review (60 unit tests passing, 95% arrival rate) |
| M14 | Evaluation | Plotting suite (Figs. 4-12, Tables IV-VI) | Scheduled next |

Full parameter grounding, paper corrections, and review notes: [docs/PKTD3-TD_Tracker.md](docs/PKTD3-TD_Tracker.md)

## Tech stack

- **Python 3.10+**: Core programming language.
- **NumPy**: Matrix operations, vector math, and Gauss-Markov noise generation.
- **Pytest**: Automated test discovery and test assertion framework.
- **PyTorch**: Deep neural network construction (Actor, TwinCritic, TD3 agent).
- **Matplotlib**: Training reward curve visualization and evaluation plotting suite.

## Contributing and acknowledgments

This repository is an academic coursework project developed exclusively by the Group 5 student team at IIIT Allahabad and does not accept external pull requests or issues.

Guidance and technical review are provided by supervisor Nayanjit Talukdar (PhD Scholar, WSN Lab, Department of IT, IIIT Allahabad). We also acknowledge the authors of the original IEEE TNSE publication for their algorithmic formulations.

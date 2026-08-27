# PKTD3-TD: 3-D UAV Trajectory Design via Deep Reinforcement Learning

This repository provides an industry-standard Python implementation reproducing the **PKTD3-TD** (Prior-Knowledge-guided Twin Delayed Deep Deterministic Policy Gradient with Temporal Difference) algorithm and benchmark suite from the paper:

> M. Li, M. Dong, H. Wang, and H. Wang, "3-D Trajectory Design Based on Deep Reinforcement Learning for UAV-Assisted Communication Networks," *IEEE Transactions on Network Science and Engineering*, vol. 13, 2026.

Developed as the course project for **Data Management in Mobile and Sensor Networks (DMMSN)**.

---

## Team & Supervision

- **Student Team (Group 5):**
  - Krishna Sikheriya
  - Kushagra Gupta
  - Aditya Pankaj Sharma
  - Vaibhav Sharma
- **Supervisor:**
  - Nayanjit Talukdar (PhD Scholar, WSN Lab, Department of IT, IIIT Allahabad)

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd uav_trajectory_rl
   ```

2. **Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. **Install the package in editable mode with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

---

## Running Tests

Execute the comprehensive test suite using pytest:
```bash
pytest tests/ -v
```

---

## Project Structure & Module Map

```
uav_trajectory_rl/
├── src/
│   └── uav_trajectory_rl/
│       ├── __init__.py
│       ├── config.py              # System parameters, environment geometry & RL hyperparams
│       ├── uav_kinematics.py       # Spherical motion & acceleration constraints (eq. 1–3)
│       ├── user_mobility.py        # Gauss-Markov ground user mobility model (eq. 4–7)
│       ├── channel_model.py        # Probabilistic LoS/NLoS path loss & rate (eq. 8–14)
│       ├── energy_model.py         # Rotary-wing propulsion energy consumption (eq. 15–16)
│       └── mdp_environment.py     # Gym-like MDP environment wrapper (eq. 17–29, Alg. 1)
├── tests/
│   ├── __init__.py
│   ├── test_uav_kinematics.py      # Kinematics displacement & accel clipping tests
│   ├── test_user_mobility.py       # UserSwarm initialization & Gauss-Markov step tests
│   ├── test_channel_model.py       # LoS probability, path loss & rate tests
│   ├── test_energy_model.py        # Propulsion power & energy tests
│   └── test_mdp_environment.py     # State space dim (2K+6) & MDP step loop tests
├── docs/
│   └── PKTD3-TD_Tracker.md         # Source grounding, parameters, assumptions & review notes
├── scripts/                        # Training and benchmark execution scripts
├── pyproject.toml                  # Packaging configuration (src layout)
├── README.md
└── .gitignore
```

### Module Implementation Map

| ID | File / Module | Paper Section / Equations | Status |
|---|---|---|---|
| **M0** | `src/uav_trajectory_rl/config.py` | Tables II, III & Section V-A constants | **Done — reviewed, approved** |
| **M1** | `src/uav_trajectory_rl/uav_kinematics.py` | Spherical kinematics (eq. 1–3) | **Done — reviewed, approved** |
| **M2** | `src/uav_trajectory_rl/user_mobility.py` | Gauss-Markov user mobility (eq. 4–7) | **Done — reviewed, approved** |
| **M3** | `src/uav_trajectory_rl/channel_model.py` | LoS probability, path loss & rate (eq. 8–14) | **Done — reviewed, approved (eq. 13 verified)** |
| **M4** | `src/uav_trajectory_rl/energy_model.py` | Rotary-wing propulsion power (eq. 15–16) | **Done — reviewed, approved** |
| **M5** | `src/uav_trajectory_rl/mdp_environment.py` | State, action, 6-term reward & step (eq. 17–29) | **Implemented — pending review** |
| **M6** | `prior_knowledge.py` | Prior-knowledge guidance policy (eq. 30–31) | Not started |
| **M7** | `td3_networks.py` | Actor-critic networks & replay buffer | Not started |
| **M8** | `td3_agent.py` | Clipped double-Q, delayed update, target smoothing (eq. 32–38) | Not started |
| **M9** | `train.py` | Training loop / full Algorithm 1 | Not started |
| **M10–M13** | Baselines | TDPK, Dueling DQL, PPO, Greedy | Not started |
| **M14** | Evaluation | Plotting suite (Figs. 4–12, Tables IV–VI) | Not started |

> For full parameter grounding, mathematical derivations, documented paper corrections, and review notes, see [docs/PKTD3-TD_Tracker.md](docs/PKTD3-TD_Tracker.md).

---

## Current Project Status

- **Completed & Verified Modules (M0–M5):**
  - System configuration with typed constants and documented assumptions.
  - 3D spherical UAV kinematics with strict acceleration capping.
  - Reproducible multi-user Gauss-Markov mobility simulator.
  - Air-to-ground probabilistic channel model faithful to literal paper formulation ($\log_2(\text{SNR})$).
  - Aerodynamic propulsion power model derived via momentum theory.
  - Complete MDP environment with 6-term reward structure and corrected destination-proximity formulation.
  - 100% test pass rate across 19 unit and integration tests.
- **Upcoming Modules (M6–M14):**
  - Prior-knowledge policy initialization, TD3 network architectures, training loop, baselines, and evaluation plots.

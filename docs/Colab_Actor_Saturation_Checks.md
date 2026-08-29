# Colab Training & Actor Saturation Diagnostics (Run 3)

This document tracks the real-time diagnostic checks executed via [check_actor_saturation_colab.ipynb](../notebooks/check_actor_saturation_colab.ipynb) during the full 6,000-episode Google Colab training run with the verified action-scale and state normalization fixes.

Checks are performed at milestones:
- [x] **Episode 1,000** (Logged below)
- [ ] **Episode 3,000** (Pending)
- [ ] **Episode 5,000** (Pending)

---

## High-Level Progression Summary

| Milestone | Training Progress | Training Pace | Mean Abs Diff | Saturation Frac | Mean Max Disp | Frac > 50m | Mean Eval Reward | Status / Health |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ep500** | 8.3% (500/6000) | ~10.7 min / 500 ep | — | 38% | 0.0m | 0.0% | 243.58 | Active learning; early exploration |
| **ep1000** | 16.7% (1000/6000) | 10.7 min / 500 ep | 0.3192 | 38% | **142.0m** | **40.0%** | **318.96** | **Healthy, active movement breakout** |
| **ep3000** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Awaiting run progress* |
| **ep5000** | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Awaiting run progress* |

---

## 1. Checkpoint 1,000 Diagnostics

**Recorded:** 2026-08-29  
**Checkpoints detected on Drive:** `[500, 1000]`

### Cell 4 — Actor Outputs & Saturation Check Across Checkpoints

Fixed test states (8 fixed random states, uniform $[-1, 1]^{26}$, seed 999):

#### ep500 (total_updates = 77,137)
```text
[[-0.5293  0.7716 -0.9339]
 [ 0.0321  0.9993 -0.9997]
 [-0.1169 -0.8459 -0.8185]
 [ 1.      0.763  -1.    ]
 [ 0.9962  0.9961 -0.9543]
 [-0.4684  0.9099 -0.9977]
 [ 0.9373  0.9854 -0.9998]
 [ 0.5006  0.9968 -1.    ]]
```

#### ep1000 (total_updates = 177,137)
```text
[[-0.7832  0.3146  0.0598]
 [ 0.6458  0.9982 -1.    ]
 [-0.9436 -0.9994 -0.951 ]
 [ 0.9979  0.6939 -0.9996]
 [ 1.      1.     -0.99  ]
 [ 0.9996  0.9751 -1.    ]
 [ 1.      0.9638 -1.    ]
 [ 0.9916  0.9909  0.9968]]
```

#### Diff & Saturation Metrics
```text
=== Mean abs change between consecutive checkpoints ===
ep500 -> ep1000: mean_abs_diff=0.319208, frac_outputs_saturated(|x|>0.999)=0.38

=== INTERPRETATION ===
OK: actor output is still changing between checkpoints (mean diff = 0.3192) -- training appears active, not frozen.
```

---

### Cell 5 — Live Training Progress, Pace, and ETA Visualizer

```text
=================================================================
TRAINING PROGRESS: [█████░░░░░░░░░░░░░░░░░░░░░░░░░] 16.7% (1000/6000 eps)
=================================================================
  Pace: 10.7 min per 500 episodes (0.78 ep/s)
  Time Remaining (ETA): ~106.8 minutes (1.78 hours)
  Estimated Completion: 08:45 AM (2026-08-29)

Live Rewards: 1000 episodes logged.
  Current Reward: 56.73
  Rolling Avg (last 50): 226.96
```

---

### Cell 6 — Movement & Corner-Escape Trend Check

10 evaluation seeds (0–9) rollout across saved checkpoints measuring maximum displacement from $Q_{\text{START}} = (0, 0, 50)$ and episode reward:

| Episode | Mean Max Displacement | Frac > 50m | Mean Reward |
| :---: | :---: | :---: | :---: |
| **500** | 0.0m | 0.0% | 243.58 |
| **1000** | **142.0m** | **40.0%** | **318.96** |

---

### Key Takeaways from ep1000

1. **No Actor Freezing / Dead Neurons:** `mean_abs_diff = 0.3192` confirms ongoing gradient updates. Output saturation is healthy at 38%.
2. **Decisive Breakout from Corner Inaction:**
   - At episode 500: stationary ($0.0\text{ m}$ displacement, $0.0\%$ exceeding 50m).
   - By episode 1,000: **Mean Max Displacement jumped to 142.0m**, with **40.0% of evaluation seeds exceeding 50m**.
3. **Reward Elevation:** Evaluation mean reward increased from $+243.58$ to **$+318.96$**, confirming that genuine forward flight is being positively rewarded by the critic.

---

## 2. Checkpoint 3,000 Diagnostics

*(To be populated upon receiving outputs for Episode 3,000)*

---

## 3. Checkpoint 5,000 Diagnostics

*(To be populated upon receiving outputs for Episode 5,000)*

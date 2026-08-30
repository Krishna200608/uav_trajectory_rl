# Colab Training, Arrival Rate & Q1 Spread Diagnostics (Run 4)

This document tracks the real-time diagnostic checks executed via [`check_actor_saturation_colab.ipynb`](../notebooks/check_actor_saturation_colab.ipynb) during the full 6,000-episode Google Colab training run with **probabilistic annealed prior-knowledge handoff** (`--r-rand 20000 --anneal-steps 20000 --checkpoint-every 250`).

Checks are performed at milestones:
- [x] **Episode 1,000** (Logged below)
- [x] **Episode 3,000 (Halfway Mark)** (Logged below)
- [ ] **Episode 5,000** (Pending)
- [ ] **Episode 6,000 (Final Completion)** (Pending)

---

## High-Level Progression Summary

| Checkpoint | Training Progress | Total Updates | Mean Abs Diff | Saturation Frac | Mean Max Disp | Frac > 50m | **Arrival Rate** | **Q1 Spread A-vs-C** | Health / Diagnostic Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ep250** | 4.2% (250/6000) | 26,106 | — | 0% | 96.2 m | 13.3% | **0.0%** | **2.68%** | Hand-off transition active; early movement breakout |
| **ep500** | 8.3% (500/6000) | 76,106 | 0.6071 | 4% | 91.1 m | 16.7% | **0.0%** | **9.49%** | Active updates; Q1 spread peaked at 9.49% |
| **ep750** | 12.5% (750/6000) | 126,106 | 0.5555 | 17% | 58.7 m | 16.7% | **0.0%** | **6.77%** | Steady policy change; sustained 16.7% corner escapes |
| **ep1000** | 16.7% (1000/6000) | 176,106 | 0.7817 | 42% | 38.0 m | 10.0% | **0.0%** | **1.81%** | High policy gradient activity (diff 0.7817); Q1 spread 1.81% |
| **ep1250** | 20.8% (1250/6000) | 226,106 | 0.3876 | 50% | 46.9 m | 16.7% | **0.0%** | **4.05%** | Active policy update; 50% saturation |
| **ep1500** | 25.0% (1500/6000) | 276,106 | 0.4859 | 50% | 19.0 m | 10.0% | **0.0%** | **4.08%** | Displacement contracting; Q1 spread flat at 4.08% |
| **ep1750** | 29.2% (1750/6000) | 326,106 | 0.3743 | 62% | 20.4 m | 10.0% | **0.0%** | **4.82%** | Saturation rising to 62%; Q1 spread flat |
| **ep2000** | 33.3% (2000/6000) | 376,106 | 0.2381 | 62% | 20.7 m | 13.3% | **0.0%** | **12.78%** | Transient peak in Q1 spread (12.78%) |
| **ep2250** | 37.5% (2250/6000) | 426,106 | 0.3565 | 75% | 44.4 m | 13.3% | **0.0%** | **0.86%** | Q1 spread collapses back down to 0.86% |
| **ep2500** | 41.7% (2500/6000) | 476,106 | 0.3353 | 62% | 5.0 m | 3.3% | **0.0%** | **2.04%** | Corner lock-in begins; Frac > 50m drops to 3.3% |
| **ep2750** | 45.8% (2750/6000) | 526,106 | 0.4175 | 79% | 18.2 m | 3.3% | **0.0%** | **2.83%** | Saturation hits 79%; minimal displacement |
| **ep3000** | 50.0% (3000/6000) | 576,106 | 0.4431 | 79% | **0.0 m** | **0.0%** | **0.0%** | **1.37%** | **Halfway reached; policy locked at corner (0.0m disp, 1.37% Q1 spread)** |

---

## 1. Checkpoint 1,000 Diagnostics

**Recorded:** 2026-08-30  
**Checkpoints detected on Drive:** `[250, 500, 750, 1000]`  

### Cell 4 — Actor Outputs & Saturation Check Across Checkpoints

Fixed test states (8 fixed random states, uniform $[-1, 1]^{26}$, seed 999):

#### ep250 (total_updates = 26,106)
```text
[[ 0.1158  0.9791  0.7255]
 [-0.4238  0.8658 -0.6476]
 [ 0.9982  1.     -1.    ]
 [-0.2886  0.3085 -0.9784]
 [-0.9724  0.9438  0.9849]
 [-0.0279  1.     -0.9921]
 [ 0.2945  0.9998  0.9362]
 [-0.9924  1.      1.    ]]
```

#### ep500 (total_updates = 76,106)
```text
[[ 0.8582 -0.4353 -0.8334]
 [-0.9944  0.8775 -0.9948]
 [ 0.9987  0.9798 -0.995 ]
 [ 0.7781  0.6873 -0.7423]
 [ 0.8029  0.3968  0.8749]
 [ 1.      0.9952  0.0986]
 [ 0.9984  0.8436 -0.9218]
 [-0.9797  0.9754  0.0929]]
```

#### ep750 (total_updates = 126,106)
```text
[[ 0.2735  0.6324  0.5207]
 [ 0.3703  0.9998 -0.6414]
 [ 0.6197  0.9999  0.8806]
 [ 0.9999  0.993  -0.2613]
 [ 1.      0.9815  0.9607]
 [ 0.9428  0.4015 -0.7564]
 [ 0.997   0.8092 -0.8675]
 [ 0.9726  0.9924  0.8621]]
```

#### ep1000 (total_updates = 176,106)
```text
[[-0.5159  0.295  -0.5548]
 [-0.9976  0.9939  0.9984]
 [ 0.9447  0.9947 -0.5053]
 [ 1.      0.9935 -0.9997]
 [ 1.      1.      0.8134]
 [-1.     -0.9992 -1.    ]
 [-1.     -0.9698  0.5665]
 [-1.      0.9747  1.    ]]
```

#### Diff & Output Saturation Metrics
```text
=== Mean abs change between consecutive checkpoints ===
ep250 -> ep500: mean_abs_diff=0.607108, frac_outputs_saturated(|x|>0.999)=0.04
ep500 -> ep750: mean_abs_diff=0.555499, frac_outputs_saturated(|x|>0.999)=0.17
ep750 -> ep1000: mean_abs_diff=0.781736, frac_outputs_saturated(|x|>0.999)=0.42

=== INTERPRETATION ===
OK: actor output is still changing between checkpoints (mean diff = 0.7817) -- training appears active, not frozen.
```

---

### Cell 5 — Live Training Progress, Pace, and ETA Visualizer

```text
=================================================================
TRAINING PROGRESS: [█████░░░░░░░░░░░░░░░░░░░░░░░░░] 16.7% (1000/6000 eps)
=================================================================
  Pace: 10.7 min per 500 episodes (0.78 ep/s)
  Time Remaining (ETA): ~106.9 minutes (1.78 hours)
  Estimated Completion: 06:41 AM (2026-08-30)

Live Rewards: 1000 episodes logged.
  Current Reward: 59.31
  Rolling Avg (last 50): 132.50
```

---

### Cell 6 — Decisive Signal Monitor (30-Seed Behavioral Check & Q1 Spread at $Q_{\text{START}}$)

Evaluated across 30 deterministic seeds (seeds 0–29, $K=10$):

| Checkpoint | Mean Max Displacement | Frac > 50m | **Arrival Rate** | **Q1 Spread A-vs-C at $Q_{\text{START}}$** |
| :---: | :---: | :---: | :---: | :---: |
| `ep250` | 96.2 m | 13.3% | **0.0%** | **2.68%** |
| `ep500` | 91.1 m | 16.7% | **0.0%** | **9.49%** |
| `ep750` | 58.7 m | 16.7% | **0.0%** | **6.77%** |
| `ep1000` | 38.0 m | 10.0% | **0.0%** | **1.81%** |

---

## Key Analysis of the Episode 1,000 Check-in

1. **Training Health & Active Adaptation:**
   - The actor is strongly active and dynamic between consecutive checkpoints (`mean_abs_diff = 0.607` $\rightarrow$ `0.555` $\rightarrow$ `0.782`). There is zero policy freeze or standstill collapse.
   - Training pace is very steady at **0.78 ep/s** (~10.7 min per 500 episodes), with completion of the full 6,000 episodes projected in ~1.78 hours.

2. **Corner Escapes vs. Destination Arrival:**
   - Between 10.0% and 16.7% of evaluation rollouts achieve escape from the initial corner (`Frac > 50m`), with mean maximum displacements of 38.0 m to 96.2 m.
   - However, **`ArrivalRate` is currently 0.0% across all 4 checkpoints** (0 arrivals out of 120 total evaluation episodes).

3. **Value Surface Differentiation ($Q_1$ Spread):**
   - At `ep500` and `ep750`, the Q1 spread between Action A ("Toward goal") and Action C ("Into the wall") climbed to **9.49% and 6.77%** — a noticeable increase above the 0.5%–3.5% seen in earlier abrupt-switch diagnostics.
   - At `ep1000`, the spread settled back down to **1.81%**, reflecting the persistent flat-value-surface difficulty at the corner boundary.
   - We will monitor whether this spread widens past the >15–20% mark as training progresses toward episode 3,000 and 5,000.

---

## 2. Checkpoint 3,000 Diagnostics (Halfway Mark)

**Recorded:** 2026-08-30  
**Checkpoints detected on Drive:** `[250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]` (12 checkpoints)

### Cell 4 — Actor Outputs & Saturation Check Across Checkpoints

Fixed test states (8 fixed random states, uniform $[-1, 1]^{26}$, seed 999):

#### ep1250 (total_updates = 226,106)
```text
[[-0.9632 -0.4266  0.9764]
 [ 1.      0.4652  0.8842]
 [ 1.     -0.4351 -1.    ]
 [ 0.997   0.9893 -1.    ]
 [ 0.9964  1.      0.9999]
 [-1.     -0.9999 -1.    ]
 [-1.     -0.7813 -0.9981]
 [-1.      0.9431  1.    ]]
```

#### ep1500 (total_updates = 276,106)
```text
[[ 0.9527  0.9874 -0.6041]
 [ 1.      0.5778 -0.992 ]
 [ 1.      0.999  -1.    ]
 [ 1.      0.9998 -1.    ]
 [ 1.      1.      0.999 ]
 [-1.      0.9733 -1.    ]
 [-1.      0.4558 -0.9989]
 [-0.9472  0.8973  1.    ]]
```

#### ep1750 (total_updates = 326,106)
```text
[[ 0.2909  0.8888  0.9945]
 [ 1.      0.9656  1.    ]
 [ 1.      1.     -1.    ]
 [ 1.      0.9786 -0.9833]
 [ 1.      1.     -0.9999]
 [-1.      1.     -1.    ]
 [-1.      0.6754 -1.    ]
 [ 0.9597  0.8451  1.    ]]
```

#### ep2000 (total_updates = 376,106)
```text
[[ 0.9827  0.1779  0.9903]
 [ 1.      0.9999  1.    ]
 [ 1.      1.     -1.    ]
 [ 1.      0.9595  0.9773]
 [ 1.      1.     -1.    ]
 [-1.      1.     -1.    ]
 [-0.9843  0.9999  0.8826]
 [ 0.9789  0.7941  1.    ]]
```

#### ep2250 (total_updates = 426,106)
```text
[[-1.     -0.0813  0.9999]
 [ 1.      0.9997  1.    ]
 [ 1.      0.9998 -0.8572]
 [ 1.      0.992  -1.    ]
 [ 1.      0.9981 -1.    ]
 [-1.      0.9279 -1.    ]
 [-1.      1.     -1.    ]
 [-1.      0.9951  1.    ]]
```

#### ep2500 (total_updates = 476,106)
```text
[[-0.3351  0.9408  0.9974]
 [ 1.      1.      1.    ]
 [ 1.      1.     -0.4386]
 [ 1.      0.9706  0.9799]
 [ 1.      0.9991 -1.    ]
 [-1.     -0.9144 -1.    ]
 [ 1.      1.     -0.9788]
 [-0.9325  1.      1.    ]]
```

#### ep2750 (total_updates = 526,106)
```text
[[-0.9492  1.     -0.9018]
 [-0.9999  1.      1.    ]
 [ 0.9977  1.      1.    ]
 [ 1.      0.8267  1.    ]
 [ 1.      0.9999 -1.    ]
 [-1.      0.9739 -1.    ]
 [ 1.      1.     -1.    ]
 [ 0.9996  1.      1.    ]]
```

#### ep3000 (total_updates = 576,106)
```text
[[ 0.992   1.      1.    ]
 [-1.     -0.3898  1.    ]
 [ 1.      1.     -1.    ]
 [ 1.      0.9857 -0.9897]
 [ 1.      1.      0.2248]
 [-1.      1.     -1.    ]
 [ 1.      1.     -1.    ]
 [ 0.9998  1.      1.    ]]
```

#### Diff & Output Saturation Metrics Across Checkpoints
```text
=== Mean abs change between consecutive checkpoints ===
ep250 -> ep500: mean_abs_diff=0.607108, frac_outputs_saturated(|x|>0.999)=0.04
ep500 -> ep750: mean_abs_diff=0.555499, frac_outputs_saturated(|x|>0.999)=0.17
ep750 -> ep1000: mean_abs_diff=0.781736, frac_outputs_saturated(|x|>0.999)=0.42
ep1000 -> ep1250: mean_abs_diff=0.387642, frac_outputs_saturated(|x|>0.999)=0.50
ep1250 -> ep1500: mean_abs_diff=0.485878, frac_outputs_saturated(|x|>0.999)=0.50
ep1500 -> ep1750: mean_abs_diff=0.374293, frac_outputs_saturated(|x|>0.999)=0.62
ep1750 -> ep2000: mean_abs_diff=0.238083, frac_outputs_saturated(|x|>0.999)=0.62
ep2000 -> ep2250: mean_abs_diff=0.356541, frac_outputs_saturated(|x|>0.999)=0.75
ep2250 -> ep2500: mean_abs_diff=0.335285, frac_outputs_saturated(|x|>0.999)=0.62
ep2500 -> ep2750: mean_abs_diff=0.417492, frac_outputs_saturated(|x|>0.999)=0.79
ep2750 -> ep3000: mean_abs_diff=0.443127, frac_outputs_saturated(|x|>0.999)=0.79

=== INTERPRETATION ===
OK: actor output is still changing between checkpoints (mean diff = 0.4431) -- training appears active, not frozen.
```

---

### Cell 5 — Live Training Progress, Pace, and ETA Visualizer

```text
=================================================================
TRAINING PROGRESS: [███████████████░░░░░░░░░░░░░░░] 50.0% (3000/6000 eps)
=================================================================
  Pace: 10.6 min per 500 episodes (0.79 ep/s)
  Time Remaining (ETA): ~63.6 minutes (1.06 hours)
  Estimated Completion: 06:41 AM (2026-08-30)

Live Rewards: 3000 episodes logged.
  Current Reward: -50.73
  Rolling Avg (last 50): 124.65
```

---

### Cell 6 — Decisive Signal Monitor (30-Seed Behavioral Check & Q1 Spread at $Q_{\text{START}}$)

Evaluated across 30 deterministic seeds (seeds 0–29, $K=10$):

| Checkpoint | Mean Max Displacement | Frac > 50m | **Arrival Rate** | **Q1 Spread A-vs-C at $Q_{\text{START}}$** |
| :---: | :---: | :---: | :---: | :---: |
| `ep250` | 96.2 m | 13.3% | **0.0%** | **2.68%** |
| `ep500` | 91.1 m | 16.7% | **0.0%** | **9.49%** |
| `ep750` | 58.7 m | 16.7% | **0.0%** | **6.77%** |
| `ep1000` | 38.0 m | 10.0% | **0.0%** | **1.81%** |
| `ep1250` | 46.9 m | 16.7% | **0.0%** | **4.05%** |
| `ep1500` | 19.0 m | 10.0% | **0.0%** | **4.08%** |
| `ep1750` | 20.4 m | 10.0% | **0.0%** | **4.82%** |
| `ep2000` | 20.7 m | 13.3% | **0.0%** | **12.78%** |
| `ep2250` | 44.4 m | 13.3% | **0.0%** | **0.86%** |
| `ep2500` | 5.0 m | 3.3% | **0.0%** | **2.04%** |
| `ep2750` | 18.2 m | 3.3% | **0.0%** | **2.83%** |
| `ep3000` | **0.0 m** | **0.0%** | **0.0%** | **1.37%** |

---

## Key Analysis of the Episode 3,000 Check-in

1. **Arrival Rate Remains Strictly 0.0% (Decisive Metric #1):**
   - Across all 12 checkpoints evaluated over 30 deterministic seeds (360 total evaluation rollouts), the deterministic destination arrival rate is **0.0%**.
   - Even with probabilistic annealing over 20,000 transition steps, the actor has not consolidated any successful navigation behavior into its deterministic policy.

2. **Corner Displacement Collapse ($96.2\text{ m} \rightarrow 0.0\text{ m}$):**
   - In early checkpoints (`ep250`–`ep500`), the policy exhibited partial corner breakouts with mean max displacement of 91–96 m and 13–17% of seeds escaping $>50\text{ m}$.
   - As training progressed past episode 2,000, forward displacement decayed rapidly:
     - `ep2000`: 20.7 m (13.3% > 50m)
     - `ep2500`: 5.0 m (3.3% > 50m)
     - `ep2750`: 18.2 m (3.3% > 50m)
     - `ep3000`: **0.0 m (0.0% > 50m)**
   - At episode 3,000, the deterministic policy has completely collapsed into 0.0m displacement — every rollout commands an action that is immediately cancelled by the initial corner boundary on Step 1.

3. **Flat Value Surface Persists (Decisive Metric #2):**
   - The Q1 spread at $Q_{\text{START}}$ between flying toward the goal and flying into the boundary wall briefly ticked up to **12.78% at ep2000**, but immediately collapsed back to **0.86% at ep2250** and sits at **1.37% at ep3000**.
   - It remains firmly in the flat, undifferentiated regime ($0.8\% - 4.8\%$), far below the target >15–20% needed to guide gradient ascent away from boundary walls.

4. **Extreme Output Saturation ($79\%$ at ep3000):**
   - As the critic provides nearly flat Q-values at the corner, policy gradient noise drives the actor's tanh output layers into saturation:
     - `ep500`: 4% saturated
     - `ep1000`: 42% saturated
     - `ep2000`: 62% saturated
     - `ep3000`: **79% saturated**
   - The actor is pushing saturated commands against the boundary bounds.

5. **Pace & Completion Forecast:**
   - Training speed remains consistent at **0.79 ep/s** (~10.6 minutes per 500 episodes).
   - The remaining 3,000 episodes are on track to complete in ~1.06 hours (~63 minutes).


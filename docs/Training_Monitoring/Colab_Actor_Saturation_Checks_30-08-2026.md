# Colab Training, Arrival Rate & Q1 Spread Diagnostics (Run 4)

This document tracks the real-time diagnostic checks executed via [`check_actor_saturation_colab.ipynb`](../notebooks/check_actor_saturation_colab.ipynb) during the full 6,000-episode Google Colab training run with **probabilistic annealed prior-knowledge handoff** (`--r-rand 20000 --anneal-steps 20000 --checkpoint-every 250`).

Checks are performed at milestones:
- [x] **Episode 1,000** (Logged below)
- [x] **Episode 3,000 (Halfway Mark)** (Logged below)
- [x] **Episode 5,000** (Logged below)
- [x] **Episode 6,000 (Final Completion)** (Logged below)

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
| **ep3000** | 50.0% (3000/6000) | 576,106 | 0.4431 | 79% | 0.0 m | 0.0% | **0.0%** | **1.37%** | Halfway reached; policy locked at corner (0.0m disp) |
| **ep3250** | 54.2% (3250/6000) | 626,106 | 0.1138 | 88% | 0.2 m | 0.0% | **0.0%** | **0.62%** | High saturation (88%); Q1 spread flat at 0.62% |
| **ep3500** | 58.3% (3500/6000) | 676,106 | 0.0046 | 88% | 9.6 m | 6.7% | **0.0%** | **0.80%** | Minor displacement bump (9.6m); Q1 spread 0.80% |
| **ep3750** | 62.5% (3750/6000) | 726,106 | 0.5801 | 83% | 3.7 m | 3.3% | **0.0%** | **2.97%** | Active actor diff (0.5801); displacement trapped |
| **ep4000** | 66.7% (4000/6000) | 776,106 | 0.4670 | 88% | 2.5 m | 0.0% | **0.0%** | **3.43%** | High saturation (88%); Q1 spread 3.43% |
| **ep4250** | 70.8% (4250/6000) | 826,106 | 0.3347 | 92% | 5.7 m | 6.7% | **0.0%** | **2.69%** | Saturation climbs to 92% |
| **ep4500** | 75.0% (4500/6000) | 876,106 | 0.3268 | 88% | 0.0 m | 0.0% | **0.0%** | **3.14%** | Zero displacement collapse across all seeds |
| **ep4750** | 79.2% (4750/6000) | 926,106 | 0.0840 | 96% | 0.0 m | 0.0% | **0.0%** | **2.79%** | Peak saturation (96%); 0.0m displacement |
| **ep5000** | 83.3% (5000/6000) | 976,106 | 0.3294 | 83% | 0.0 m | 0.0% | **0.0%** | **2.44%** | Total corner lock-in (0.0m disp, 2.44% Q1 spread) |
| **ep5250** | 87.5% (5250/6000) | 1,026,106 | 0.1150 | 92% | 2.2 m | 3.3% | **0.0%** | **1.38%** | Saturation 92%; Q1 spread 1.38% |
| **ep5500** | 91.7% (5500/6000) | 1,076,106 | 0.1738 | 83% | 5.2 m | 6.7% | **0.0%** | **4.09%** | Minor flutter (5.2m); Q1 spread 4.09% |
| **ep5750** | 95.8% (5750/6000) | 1,126,106 | 0.1808 | 79% | 11.1 m | 6.7% | **0.0%** | **2.93%** | Approaching end; saturation 79% |
| **ep6000** | 100.0% (6000/6000) | 1,176,106 | 0.1044 | 92% | **0.0 m** | **0.0%** | **0.0%** | **1.21%** | **Run 4 Completed: Total Standstill Collapse (0.0m disp, 1.21% Q1 spread, 0% arrivals)** |

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

---

## 3. Checkpoint 5,000 Diagnostics

**Recorded:** 2026-08-30  
**Checkpoints detected on Drive:** `[250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000]` (20 checkpoints)

### Cell 4 — Actor Outputs & Saturation Check Across Checkpoints

Fixed test states (8 fixed random states, uniform $[-1, 1]^{26}$, seed 999):

#### ep3250 (total_updates = 626,106)
```text
[[ 0.8979  0.999   1.    ]
 [-1.      0.9993  0.9997]
 [ 1.      1.     -1.    ]
 [ 0.9997  0.9995 -1.    ]
 [ 1.      1.     -0.9981]
 [-1.      1.     -1.    ]
 [ 1.      1.     -1.    ]
 [ 1.      1.      1.    ]]
```

#### ep3500 (total_updates = 676,106)
```text
[[ 0.9989  1.      1.    ]
 [-0.9987  0.9967  1.    ]
 [ 1.      1.     -1.    ]
 [ 0.9991  1.     -0.9999]
 [ 1.      1.     -0.9998]
 [-1.      1.     -1.    ]
 [ 1.      1.     -1.    ]
 [ 1.      1.      1.    ]]
```

#### ep3750 (total_updates = 726,106)
```text
[[-1.      1.      1.    ]
 [-1.      1.      1.    ]
 [ 1.      1.      0.9999]
 [-0.9981  1.      0.976 ]
 [ 1.      1.      1.    ]
 [-1.      1.     -1.    ]
 [ 1.      1.      0.9464]
 [-0.9985  1.      1.    ]]
```

#### ep4000 (total_updates = 776,106)
```text
[[-1.      1.      1.    ]
 [ 1.      1.     -0.5866]
 [ 1.      1.      1.    ]
 [ 1.      1.     -1.    ]
 [ 1.      1.      1.    ]
 [-1.      1.     -0.9703]
 [ 1.      1.     -0.6731]
 [ 0.9998  1.      1.    ]]
```

#### ep4250 (total_updates = 826,106)
```text
[[-1.      1.      1.    ]
 [-0.9999  0.9995 -0.9025]
 [-1.      1.      0.9999]
 [ 1.      1.      0.9995]
 [ 1.      1.      1.    ]
 [-1.      1.     -0.9267]
 [ 1.      1.      1.    ]
 [ 1.      1.      1.    ]]
```

#### ep4500 (total_updates = 876,106)
```text
[[-1.      1.      1.    ]
 [ 0.9797  1.     -1.    ]
 [ 0.8597  1.     -1.    ]
 [ 1.      1.      1.    ]
 [ 0.9999  1.      1.    ]
 [-1.      1.     -1.    ]
 [ 1.      1.      1.    ]
 [-0.8313  1.      1.    ]]
```

#### ep4750 (total_updates = 926,106)
```text
[[-1.      1.      1.    ]
 [ 0.9998  1.     -1.    ]
 [ 1.      1.     -1.    ]
 [ 1.      1.      1.    ]
 [ 1.      1.      1.    ]
 [-1.      1.     -1.    ]
 [ 1.      1.      0.9761]
 [ 1.      1.      1.    ]]
```

#### ep5000 (total_updates = 976,106)
```text
[[ 0.9025  1.      1.    ]
 [ 1.      0.9946 -0.9648]
 [ 1.      1.      0.9856]
 [ 1.      1.      1.    ]
 [ 1.      1.      1.    ]
 [-1.      1.     -1.    ]
 [ 1.      1.     -1.    ]
 [-0.9999  1.      1.    ]]
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
ep3000 -> ep3250: mean_abs_diff=0.113830, frac_outputs_saturated(|x|>0.999)=0.88
ep3250 -> ep3500: mean_abs_diff=0.004556, frac_outputs_saturated(|x|>0.999)=0.88
ep3500 -> ep3750: mean_abs_diff=0.580059, frac_outputs_saturated(|x|>0.999)=0.83
ep3750 -> ep4000: mean_abs_diff=0.467008, frac_outputs_saturated(|x|>0.999)=0.88
ep4000 -> ep4250: mean_abs_diff=0.334701, frac_outputs_saturated(|x|>0.999)=0.92
ep4250 -> ep4500: mean_abs_diff=0.326768, frac_outputs_saturated(|x|>0.999)=0.88
ep4500 -> ep4750: mean_abs_diff=0.083987, frac_outputs_saturated(|x|>0.999)=0.96
ep4750 -> ep5000: mean_abs_diff=0.329374, frac_outputs_saturated(|x|>0.999)=0.83

=== INTERPRETATION ===
OK: actor output is still changing between checkpoints (mean diff = 0.3294) -- training appears active, not frozen.
```

---

### Cell 5 — Live Training Progress, Pace, and ETA Visualizer

```text
=================================================================
TRAINING PROGRESS: [█████████████████████████░░░░░] 83.3% (5000/6000 eps)
=================================================================
  Pace: 10.6 min per 500 episodes (0.78 ep/s)
  Time Remaining (ETA): ~21.2 minutes (0.35 hours)
  Estimated Completion: 06:40 AM (2026-08-30)

Live Rewards: 5000 episodes logged.
  Current Reward: 130.28
  Rolling Avg (last 50): 158.29
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
| `ep3000` | 0.0 m | 0.0% | **0.0%** | **1.37%** |
| `ep3250` | 0.2 m | 0.0% | **0.0%** | **0.62%** |
| `ep3500` | 9.6 m | 6.7% | **0.0%** | **0.80%** |
| `ep3750` | 3.7 m | 3.3% | **0.0%** | **2.97%** |
| `ep4000` | 2.5 m | 0.0% | **0.0%** | **3.43%** |
| `ep4250` | 5.7 m | 6.7% | **0.0%** | **2.69%** |
| `ep4500` | **0.0 m** | **0.0%** | **0.0%** | **3.14%** |
| `ep4750` | **0.0 m** | **0.0%** | **0.0%** | **2.79%** |
| `ep5000` | **0.0 m** | **0.0%** | **0.0%** | **2.44%** |

---

## Key Analysis of the Episode 5,000 Check-in

1. **Arrival Rate is Universally Zero (Decisive Metric #1):**
   - Across all 20 checkpoints evaluated over 30 deterministic seeds (600 total evaluation rollouts), **arrival rate is strictly 0.0%**.
   - Neither longer training (up to 976,106 gradient updates) nor handoff annealing has succeeded in producing a single deterministic destination arrival.

2. **Total Corner Paralysis / Displacement Collapse:**
   - Over the last 2,000 episodes, the policy has collapsed into near-total immobility at the initial corner:
     - `ep4500`: **0.0 m** (0.0% > 50m)
     - `ep4750`: **0.0 m** (0.0% > 50m)
     - `ep5000`: **0.0 m** (0.0% > 50m)
   - Every single rollout from `ep4500` through `ep5000` commands an action that is cancelled on Step 1 by the boundary wall at $Q_{\text{START}} = (0, 0, 50)$.

3. **Flat Value Surface is Invariable (Decisive Metric #2):**
   - Across the entire span from `ep3000` to `ep5000` (8 consecutive checkpoints), the Q1 spread between flying toward the goal and flying into the boundary wall has hovered steadily between **0.62% and 3.43%** (averaging ~2.4%).
   - The critic has completely failed to differentiate between moving toward the goal and colliding with the wall.

4. **Extreme Tanh Output Saturation ($83\% - 96\%$):**
   - The actor's outputs are heavily pinned to the extreme boundaries $[-1, 1]$:
     - `ep4250`: 92% saturated
     - `ep4750`: **96% saturated**
     - `ep5000`: 83% saturated
   - This confirms the flat value surface gradient amplification mechanism: without a true gradient toward success, the policy gradient pushes outputs to extreme limits that jam into the physical boundary constraints.

5. **Final Stage:**
   - Training is 83.3% complete with ~21 minutes remaining until the 6,000-episode completion.

---

## 4. Checkpoint 6,000 Diagnostics (Final Run Completion)

**Recorded:** 2026-08-30  
**Checkpoints detected on Drive:** `[250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000]` (All 24 checkpoints)

### Cell 4 — Actor Outputs & Saturation Check Across Checkpoints

Fixed test states (8 fixed random states, uniform $[-1, 1]^{26}$, seed 999):

#### ep5250 (total_updates = 1,026,106)
```text
[[-1.      1.      1.    ]
 [ 1.      1.     -1.    ]
 [ 1.      1.      1.    ]
 [ 1.      0.5653  1.    ]
 [ 1.      1.      1.    ]
 [-1.      1.     -1.    ]
 [ 1.      1.     -1.    ]
 [-0.6322  1.      1.    ]]
```

#### ep5500 (total_updates = 1,076,106)
```text
[[-0.9877  1.      1.    ]
 [ 1.      0.9999 -0.991 ]
 [ 1.      1.     -1.    ]
 [ 1.      1.      1.    ]
 [ 1.      1.      1.    ]
 [-1.      1.     -0.876 ]
 [ 1.      1.     -1.    ]
 [ 0.9601  1.      1.    ]]
```

#### ep5750 (total_updates = 1,126,106)
```text
[[-0.6975  1.      1.    ]
 [ 1.     -0.9189 -1.    ]
 [ 1.      0.9989 -1.    ]
 [ 1.      1.      1.    ]
 [ 1.      1.     -0.9522]
 [-1.      1.     -1.    ]
 [ 1.      1.     -1.    ]
 [ 0.917   1.      1.    ]]
```

#### ep6000 (total_updates = 1,176,106)
```text
[[-1.      0.9997  1.    ]
 [ 1.     -0.9904 -1.    ]
 [ 1.      0.9989 -1.    ]
 [ 1.      1.     -1.    ]
 [ 1.      1.     -1.    ]
 [-1.      1.     -1.    ]
 [ 1.      1.     -1.    ]
 [ 1.      1.      1.    ]]
```

#### Diff & Output Saturation Metrics Across All Checkpoints
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
ep3000 -> ep3250: mean_abs_diff=0.113830, frac_outputs_saturated(|x|>0.999)=0.88
ep3250 -> ep3500: mean_abs_diff=0.004556, frac_outputs_saturated(|x|>0.999)=0.88
ep3500 -> ep3750: mean_abs_diff=0.580059, frac_outputs_saturated(|x|>0.999)=0.83
ep3750 -> ep4000: mean_abs_diff=0.467008, frac_outputs_saturated(|x|>0.999)=0.88
ep4000 -> ep4250: mean_abs_diff=0.334701, frac_outputs_saturated(|x|>0.999)=0.92
ep4250 -> ep4500: mean_abs_diff=0.326768, frac_outputs_saturated(|x|>0.999)=0.88
ep4500 -> ep4750: mean_abs_diff=0.083987, frac_outputs_saturated(|x|>0.999)=0.96
ep4750 -> ep5000: mean_abs_diff=0.329374, frac_outputs_saturated(|x|>0.999)=0.83
ep5000 -> ep5250: mean_abs_diff=0.114997, frac_outputs_saturated(|x|>0.999)=0.92
ep5250 -> ep5500: mean_abs_diff=0.173849, frac_outputs_saturated(|x|>0.999)=0.83
ep5500 -> ep5750: mean_abs_diff=0.180772, frac_outputs_saturated(|x|>0.999)=0.79
ep5750 -> ep6000: mean_abs_diff=0.104381, frac_outputs_saturated(|x|>0.999)=0.92

=== INTERPRETATION ===
OK: actor output is still changing between checkpoints (mean diff = 0.1044) -- training appears active, not frozen.
```

---

### Cell 5 — Live Training Progress, Pace, and ETA Visualizer

```text
=================================================================
TRAINING PROGRESS: [██████████████████████████████] 100.0% (6000/6000 eps)
=================================================================
  Pace: 10.6 min per 500 episodes (0.78 ep/s)
  Time Remaining (ETA): ~0.0 minutes (0.00 hours)
  Estimated Completion: 06:43 AM (2026-08-30)

Live Rewards: 6000 episodes logged.
  Current Reward: -0.95
  Rolling Avg (last 50): 138.75
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
| `ep3000` | 0.0 m | 0.0% | **0.0%** | **1.37%** |
| `ep3250` | 0.2 m | 0.0% | **0.0%** | **0.62%** |
| `ep3500` | 9.6 m | 6.7% | **0.0%** | **0.80%** |
| `ep3750` | 3.7 m | 3.3% | **0.0%** | **2.97%** |
| `ep4000` | 2.5 m | 0.0% | **0.0%** | **3.43%** |
| `ep4250` | 5.7 m | 6.7% | **0.0%** | **2.69%** |
| `ep4500` | 0.0 m | 0.0% | **0.0%** | **3.14%** |
| `ep4750` | 0.0 m | 0.0% | **0.0%** | **2.79%** |
| `ep5000` | 0.0 m | 0.0% | **0.0%** | **2.44%** |
| `ep5250` | 2.2 m | 3.3% | **0.0%** | **1.38%** |
| `ep5500` | 5.2 m | 6.7% | **0.0%** | **4.09%** |
| `ep5750` | 11.1 m | 6.7% | **0.0%** | **2.93%** |
| `ep6000` | **0.0 m** | **0.0%** | **0.0%** | **1.21%** |

---

## Definitive Findings and Scientific Conclusion on Run 4

1. **Universal 0.0% Deterministic Destination Arrival Rate:**
   Across all 24 checkpoints evaluated across 30 deterministic seeds (**720 total evaluation episodes**), **not a single arrival occurred (0.0%)**.
   Even with 20,000 steps of gradual probabilistic annealing from prior knowledge to the neural actor, the policy failed to consolidate goal navigation.

2. **Total Standstill Collapse at Final Checkpoint (`ep6000`):**
   The final checkpoint `ep6000.pt` achieves **0.0 m maximum displacement** across 100% of evaluation seeds. On Step 1 of every episode, the deterministic actor commands an action that violates the spatial boundary constraints and is cancelled in place.

3. **Persistent Flat Value Surface at $Q_{\text{START}}$:**
   The Q1 spread between Action A ("Toward goal") and Action C ("Into the wall") never broke out into a discriminating signal:
   - Initial spread: **2.68%** at `ep250`
   - Peak spread: **12.78%** at `ep2000` (transient)
   - Final spread: **1.21%** at `ep6000`
   - Mean spread across 24 checkpoints: **~3.2%**
   The critic remained completely unable to distinguish between flying toward the destination and crashing into the bounding walls.

4. **Near-Total Tanh Saturation ($92\%$ at `ep6000`):**
   Outputs reached 92% saturation at `ep6000`. In the presence of an undifferentiated Q surface, the policy gradient continually pushed the unbounded actor weights toward extreme saturated states.

5. **Closing Verdict:**
   Run 4 decisively confirms that the failure of PKTD3-TD to achieve deterministic destination arrivals is **not an artifact of an abrupt hand-off switch**. It is a fundamental structural consequence of the flat value landscape at the boundary corner $Q_{\text{START}} = (0, 0, 50)$ under the paper's literal reward formulation and boundary cancellation mechanics.




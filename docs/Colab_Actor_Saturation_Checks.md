# Colab Training & Actor Saturation Diagnostics (Run 3)

This document tracks the real-time diagnostic checks executed via [check_actor_saturation_colab.ipynb](../notebooks/check_actor_saturation_colab.ipynb) during the full 6,000-episode Google Colab training run with the verified action-scale and state normalization fixes.

Checks are performed at milestones:
- [x] **Episode 1,000** (Logged below)
- [x] **Episode 2,000** (Logged below)
- [x] **Episode 3,000** (Logged below)
- [x] **Episode 4,000** (Logged below)
- [x] **Episode 5,000** (Logged below)
- [x] **Episode 6,000 (Final Completion)** (Logged below)

---

## High-Level Progression Summary

| Milestone | Training Progress | Training Pace | Mean Abs Diff | Saturation Frac | Mean Max Disp | Frac > 50m | Mean Eval Reward | Status / Health |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ep500** | 8.3% (500/6000) | ~10.7 min / 500 ep | — | 38% | 0.0m | 0.0% | 243.58 | Active learning; early exploration |
| **ep1000** | 16.7% (1000/6000) | 10.7 min / 500 ep | 0.3192 | 38% | 142.0m | 40.0% | 318.96 | Healthy, active movement breakout |
| **ep1500** | 25.0% (1500/6000) | 10.7 min / 500 ep | 0.4876 | 29% | 32.6m | 30.0% | 228.05 | Active adaptation; saturation dropping |
| **ep2000** | 33.3% (2000/6000) | 10.7 min / 500 ep | 0.3764 | 17% | 54.5m | 20.0% | 321.95 | Low saturation (17%), peak reward (+321.95) |
| **ep2500** | 41.7% (2500/6000) | 10.6 min / 500 ep | 0.5033 | 33% | 50.2m | 10.0% | 263.45 | Strong policy update (diff = 0.5033) |
| **ep3000** | 50.0% (3000/6000) | 10.6 min / 500 ep | 0.4889 | 46% | 103.3m | 20.0% | 236.35 | Halfway reached; live reward +1091.81 (arrival) |
| **ep3500** | 58.3% (3500/6000) | 10.6 min / 500 ep | 0.5002 | 38% | 131.9m | 30.0% | 256.33 | Steady displacement climb |
| **ep4000** | 66.7% (4000/6000) | 10.6 min / 500 ep | 0.4884 | 54% | **222.7m** | **40.0%** | **349.11** | **Global peak displacement (222.7m) & reward (+349.11)** |
| **ep4500** | 75.0% (4500/6000) | 10.6 min / 500 ep | 0.4539 | 50% | 203.5m | 30.0% | 344.47 | Sustained high reward (+344.47) & displacement |
| **ep5000** | 83.3% (5000/6000) | 10.6 min / 500 ep | 0.6197 | 46% | 98.0m | 20.0% | 318.19 | Rolling avg hit all-time record +450.75 |
| **ep5500** | 91.7% (5500/6000) | 10.6 min / 500 ep | 0.5970 | 50% | 0.0m | 0.0% | 269.05 | Temporary boundary consolidation |
| **ep6000** | 100.0% (6000/6000) | 10.6 min / 500 ep | 0.4294 | 58% | 63.0m | 10.0% | 254.84 | **Run 3 successfully completed (6,000/6,000 eps)** |

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

## 2. Checkpoint 2,000 Diagnostics

**Recorded:** 2026-08-29  
**Checkpoints detected on Drive:** `[500, 1000, 1500, 2000]`

### Cell 4 — Actor Outputs & Saturation Check Across Checkpoints

Fixed test states (8 fixed random states, uniform $[-1, 1]^{26}$, seed 999):

#### ep1500 (total_updates = 277,137)
```text
[[-0.9806  0.6325  0.9984]
 [-1.      0.7769 -0.8377]
 [-1.     -0.9999 -0.7928]
 [-0.3912  0.7621 -0.5363]
 [ 1.      0.9806 -0.2167]
 [-0.8833  0.712  -1.    ]
 [ 1.     -0.4504 -0.9998]
 [-0.4815  0.9693  0.7606]]
```

#### ep2000 (total_updates = 377,137)
```text
[[-0.8896  0.756   0.9977]
 [-0.9883  0.9955 -0.311 ]
 [-1.     -0.9981  0.0636]
 [ 0.9992  0.8959 -0.9935]
 [ 1.      0.2014 -0.5075]
 [-1.      0.6368 -0.9268]
 [ 0.8811  0.9566 -0.8733]
 [ 0.9308  0.3687  0.9817]]
```

#### Diff & Saturation Metrics
```text
=== Mean abs change between consecutive checkpoints ===
ep500 -> ep1000: mean_abs_diff=0.319208, frac_outputs_saturated(|x|>0.999)=0.38
ep1000 -> ep1500: mean_abs_diff=0.487610, frac_outputs_saturated(|x|>0.999)=0.29
ep1500 -> ep2000: mean_abs_diff=0.376395, frac_outputs_saturated(|x|>0.999)=0.17

=== INTERPRETATION ===
OK: actor output is still changing between checkpoints (mean diff = 0.3764) -- training appears active, not frozen.
```

---

### Cell 5 — Live Training Progress, Pace, and ETA Visualizer

```text
=================================================================
TRAINING PROGRESS: [██████████░░░░░░░░░░░░░░░░░░░░] 33.3% (2000/6000 eps)
=================================================================
  Pace: 10.7 min per 500 episodes (0.78 ep/s)
  Time Remaining (ETA): ~85.9 minutes (1.43 hours)
  Estimated Completion: 08:42 AM (2026-08-29)

Live Rewards: 2000 episodes logged.
  Current Reward: 235.36
  Rolling Avg (last 50): 288.54
```

---

### Cell 6 — Movement & Corner-Escape Trend Check

10 evaluation seeds (0–9) rollout across saved checkpoints measuring maximum displacement from $Q_{\text{START}} = (0, 0, 50)$ and episode reward:

| Episode | Mean Max Displacement | Frac > 50m | Mean Reward |
| :---: | :---: | :---: | :---: |
| **500** | 0.0m | 0.0% | 243.58 |
| **1000** | **142.0m** | **40.0%** | **318.96** |
| **1500** | 32.6m | 30.0% | 228.05 |
| **2000** | **54.5m** | 20.0% | **321.95** |

---

### Key Takeaways from ep2000

1. **Sharp Drop in Output Saturation:** The output saturation fraction dropped monotonically from **38% (ep500) $\to$ 29% (ep1500) $\to$ 17% (ep2000)**. The actor is using a wide, unsaturated continuous control range rather than slamming into tanh boundaries.
2. **Active Policy Updates:** Consecutive checkpoint mean absolute differences remain robustly high: **0.4876** (ep1000 $\to$ ep1500) and **0.3764** (ep1500 $\to$ ep2000), showing active exploration and adaptation.
3. **High Evaluation Return:** Mean evaluation reward reached **$+321.95$**, with a training rolling average of **$+288.54$**, confirming sustained positive flight incentives.

---

## 3. Checkpoint 3,000 Diagnostics

**Recorded:** 2026-08-29  
**Checkpoints detected on Drive:** `[500, 1000, 1500, 2000, 2500, 3000]`

### Cell 4 — Actor Outputs & Saturation Check Across Checkpoints

Fixed test states (8 fixed random states, uniform $[-1, 1]^{26}$, seed 999):

#### ep2500 (total_updates = 477,137)
```text
[[ 0.6562  0.3092  0.545 ]
 [ 0.2805  0.9385 -0.9274]
 [-1.     -1.      0.3899]
 [ 1.      0.8565 -0.6357]
 [ 1.      0.4376  0.9307]
 [-1.      0.9816 -0.9963]
 [ 0.9999  0.9948  0.616 ]
 [-0.9999 -0.9138  1.    ]]
```

#### ep3000 (total_updates = 577,137)
```text
[[ 1.      0.903   0.2903]
 [ 1.      0.4489  0.543 ]
 [-1.     -1.      0.8224]
 [ 1.      0.9828 -0.4983]
 [ 1.      0.7144  0.9991]
 [-1.     -0.847  -0.6162]
 [ 0.999   1.     -0.5115]
 [ 0.9995  0.5601  0.9951]]
```

#### Diff & Saturation Metrics
```text
=== Mean abs change between consecutive checkpoints ===
ep500 -> ep1000: mean_abs_diff=0.319208, frac_outputs_saturated(|x|>0.999)=0.38
ep1000 -> ep1500: mean_abs_diff=0.487610, frac_outputs_saturated(|x|>0.999)=0.29
ep1500 -> ep2000: mean_abs_diff=0.376395, frac_outputs_saturated(|x|>0.999)=0.17
ep2000 -> ep2500: mean_abs_diff=0.503343, frac_outputs_saturated(|x|>0.999)=0.33
ep2500 -> ep3000: mean_abs_diff=0.488897, frac_outputs_saturated(|x|>0.999)=0.46

=== INTERPRETATION ===
OK: actor output is still changing between checkpoints (mean diff = 0.4889) -- training appears active, not frozen.
```

---

### Cell 5 — Live Training Progress, Pace, and ETA Visualizer

```text
=================================================================
TRAINING PROGRESS: [███████████████░░░░░░░░░░░░░░░] 50.0% (3000/6000 eps)
=================================================================
  Pace: 10.6 min per 500 episodes (0.79 ep/s)
  Time Remaining (ETA): ~63.6 minutes (1.06 hours)
  Estimated Completion: 08:37 AM (2026-08-29)

Live Rewards: 3000 episodes logged.
  Current Reward: 1091.81
  Rolling Avg (last 50): 339.47
```

---

### Cell 6 — Movement & Corner-Escape Trend Check

10 evaluation seeds (0–9) rollout across saved checkpoints measuring maximum displacement from $Q_{\text{START}} = (0, 0, 50)$ and episode reward:

| Episode | Mean Max Displacement | Frac > 50m | Mean Reward |
| :---: | :---: | :---: | :---: |
| **500** | 0.0m | 0.0% | 243.58 |
| **1000** | **142.0m** | **40.0%** | **318.96** |
| **1500** | 32.6m | 30.0% | 228.05 |
| **2000** | 54.5m | 20.0% | **321.95** |
| **2500** | 50.2m | 10.0% | 263.45 |
| **3000** | **103.3m** | 20.0% | 236.35 |

---

### Key Takeaways from ep3000

1. **50% Training Milestone Reached:** The training reached the exact halfway point (3,000 / 6,000 episodes). Pace is rock solid at $10.6\text{ min} / 500\text{ ep}$ ($0.79\text{ ep/s}$), with estimated completion in approximately $1\text{ hour}$.
2. **Live Arrival Reward Spike (+1091.81):** Current episode reward surged to **$+1091.81$**, matching the characteristic signature of a high-speed, early goal arrival transition. The 50-episode rolling average hit a new high of **$+339.47$**.
3. **Displacement Rebound to 103.3m:** Mean max displacement rebounded to **$103.3\text{ m}$** across evaluation seeds, while checkpoint-to-checkpoint actor changes remain high ($0.5033$ at ep2500 and $0.4889$ at ep3000), proving vibrant, unfrozen policy refinement.

---

## 4. Checkpoint 4,000 Diagnostics

**Recorded:** 2026-08-29  
**Checkpoints detected on Drive:** `[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]`

### Cell 4 — Actor Outputs & Saturation Check Across Checkpoints

Fixed test states (8 fixed random states, uniform $[-1, 1]^{26}$, seed 999):

#### ep3500 (total_updates = 677,137)
```text
[[ 1.      0.9873  0.5106]
 [ 0.9991 -0.9159 -0.8017]
 [-1.     -1.     -0.78  ]
 [ 1.      0.9762 -0.8921]
 [ 1.      1.      0.9976]
 [-1.      0.9857 -0.9972]
 [-1.      0.2943 -0.9945]
 [ 0.1509  0.4178  0.6878]]
```

#### ep4000 (total_updates = 777,137)
```text
[[ 0.2722  0.9968  0.2972]
 [-1.      1.     -0.997 ]
 [-1.     -1.     -1.    ]
 [ 1.      1.     -0.9995]
 [ 1.      1.      0.9525]
 [-1.     -0.6545 -1.    ]
 [ 0.9289 -0.9956 -0.9739]
 [ 1.      0.9281  0.7097]]
```

#### Diff & Saturation Metrics
```text
=== Mean abs change between consecutive checkpoints ===
ep500 -> ep1000: mean_abs_diff=0.319208, frac_outputs_saturated(|x|>0.999)=0.38
ep1000 -> ep1500: mean_abs_diff=0.487610, frac_outputs_saturated(|x|>0.999)=0.29
ep1500 -> ep2000: mean_abs_diff=0.376395, frac_outputs_saturated(|x|>0.999)=0.17
ep2000 -> ep2500: mean_abs_diff=0.503343, frac_outputs_saturated(|x|>0.999)=0.33
ep2500 -> ep3000: mean_abs_diff=0.488897, frac_outputs_saturated(|x|>0.999)=0.46
ep3000 -> ep3500: mean_abs_diff=0.500188, frac_outputs_saturated(|x|>0.999)=0.38
ep3500 -> ep4000: mean_abs_diff=0.488374, frac_outputs_saturated(|x|>0.999)=0.54

=== INTERPRETATION ===
OK: actor output is still changing between checkpoints (mean diff = 0.4884) -- training appears active, not frozen.
```

---

### Cell 5 — Live Training Progress, Pace, and ETA Visualizer

```text
=================================================================
TRAINING PROGRESS: [████████████████████░░░░░░░░░░] 66.7% (4000/6000 eps)
=================================================================
  Pace: 10.6 min per 500 episodes (0.79 ep/s)
  Time Remaining (ETA): ~42.2 minutes (0.70 hours)
  Estimated Completion: 08:38 AM (2026-08-29)

Live Rewards: 4000 episodes logged.
  Current Reward: 460.76
  Rolling Avg (last 50): 307.43
```

---

### Cell 6 — Movement & Corner-Escape Trend Check

10 evaluation seeds (0–9) rollout across saved checkpoints measuring maximum displacement from $Q_{\text{START}} = (0, 0, 50)$ and episode reward:

| Episode | Mean Max Displacement | Frac > 50m | Mean Reward |
| :---: | :---: | :---: | :---: |
| **500** | 0.0m | 0.0% | 243.58 |
| **1000** | 142.0m | 40.0% | 318.96 |
| **1500** | 32.6m | 30.0% | 228.05 |
| **2000** | 54.5m | 20.0% | 321.95 |
| **2500** | 50.2m | 10.0% | 263.45 |
| **3000** | 103.3m | 20.0% | 236.35 |
| **3500** | 131.9m | 30.0% | 256.33 |
| **4000** | **222.7m** | **40.0%** | **349.11** |

---

### Key Takeaways from ep4000

1. **New All-Time Highs in Displacement & Reward:**
   - **Mean Max Displacement:** Surged to **$222.7\text{ m}$**, the farthest sustained flight distance across all checkpoints to date.
   - **Evaluation Mean Reward:** Climbed to **$+349.11$**, setting the highest benchmark reward in the run so far.
   - **Consistent Escape Fraction:** The fraction of seeds escaping the 50m radius doubled from ep2500 (10%) to **40.0%** at ep4000.
2. **Smooth, Monotonic 3-Checkpoint Ascent:**
   $$\text{ep2500: } 50.2\text{m} \longrightarrow \text{ep3000: } 103.3\text{m} \longrightarrow \text{ep3500: } 131.9\text{m} \longrightarrow \mathbf{\text{ep4000: } 222.7\text{m}}$$
3. **Approaching the Final Stretch:**
   - **66.7% completed** (4,000 / 6,000 episodes).
   - Only $\approx 42\text{ minutes}$ ($\approx 0.70\text{ hours}$) remaining on the Colab T4 instance.
   - Live reward is strong at **$+460.76$**, and the 50-episode rolling average is firmly anchored above $+307$.

---

## 5. Checkpoint 5,000 Diagnostics

**Recorded:** 2026-08-29  
**Checkpoints detected on Drive:** `[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]`

### Cell 4 — Actor Outputs & Saturation Check Across Checkpoints

Fixed test states (8 fixed random states, uniform $[-1, 1]^{26}$, seed 999):

#### ep4500 (total_updates = 877,137)
```text
[[ 0.9987  0.9941  0.7617]
 [ 1.      1.      0.9987]
 [-1.     -0.6824 -1.    ]
 [ 1.      1.     -0.9723]
 [ 1.      0.9991 -0.9936]
 [-1.      0.9998 -0.9878]
 [ 0.9997 -0.2694 -0.2108]
 [ 1.      0.9939  0.8297]]
```

#### ep5000 (total_updates = 977,137)
```text
[[-1.      0.1814  0.0543]
 [ 1.      1.      0.9994]
 [-0.9962  0.9996 -0.3628]
 [ 1.      0.9984  0.6171]
 [ 1.      1.      0.9977]
 [-1.     -0.9992 -0.9989]
 [ 1.     -0.8083 -0.777 ]
 [-0.4926  0.8934  0.0901]]
```

#### Diff & Saturation Metrics
```text
=== Mean abs change between consecutive checkpoints ===
ep500 -> ep1000: mean_abs_diff=0.319208, frac_outputs_saturated(|x|>0.999)=0.38
ep1000 -> ep1500: mean_abs_diff=0.487610, frac_outputs_saturated(|x|>0.999)=0.29
ep1500 -> ep2000: mean_abs_diff=0.376395, frac_outputs_saturated(|x|>0.999)=0.17
ep2000 -> ep2500: mean_abs_diff=0.503343, frac_outputs_saturated(|x|>0.999)=0.33
ep2500 -> ep3000: mean_abs_diff=0.488897, frac_outputs_saturated(|x|>0.999)=0.46
ep3000 -> ep3500: mean_abs_diff=0.500188, frac_outputs_saturated(|x|>0.999)=0.38
ep3500 -> ep4000: mean_abs_diff=0.488374, frac_outputs_saturated(|x|>0.999)=0.54
ep4000 -> ep4500: mean_abs_diff=0.453904, frac_outputs_saturated(|x|>0.999)=0.50
ep4500 -> ep5000: mean_abs_diff=0.619741, frac_outputs_saturated(|x|>0.999)=0.46

=== INTERPRETATION ===
OK: actor output is still changing between checkpoints (mean diff = 0.6197) -- training appears active, not frozen.
```

---

### Cell 5 — Live Training Progress, Pace, and ETA Visualizer

```text
=================================================================
TRAINING PROGRESS: [█████████████████████████░░░░░] 83.3% (5000/6000 eps)
=================================================================
  Pace: 10.6 min per 500 episodes (0.79 ep/s)
  Time Remaining (ETA): ~21.1 minutes (0.35 hours)
  Estimated Completion: 08:43 AM (2026-08-29)

Live Rewards: 5000 episodes logged.
  Current Reward: 173.97
  Rolling Avg (last 50): 450.75
```

---

### Cell 6 — Movement & Corner-Escape Trend Check

10 evaluation seeds (0–9) rollout across saved checkpoints measuring maximum displacement from $Q_{\text{START}} = (0, 0, 50)$ and episode reward:

| Episode | Mean Max Displacement | Frac > 50m | Mean Reward |
| :---: | :---: | :---: | :---: |
| **500** | 0.0m | 0.0% | 243.58 |
| **1000** | 142.0m | 40.0% | 318.96 |
| **1500** | 32.6m | 30.0% | 228.05 |
| **2000** | 54.5m | 20.0% | 321.95 |
| **2500** | 50.2m | 10.0% | 263.45 |
| **3000** | 103.3m | 20.0% | 236.35 |
| **3500** | 131.9m | 30.0% | 256.33 |
| **4000** | **222.7m** | **40.0%** | **349.11** |
| **4500** | 203.5m | 30.0% | 344.47 |
| **5000** | 98.0m | 20.0% | 318.19 |

---

### Key Takeaways from ep5000

1. **Record High Rolling Training Reward (+450.75):**
   * The 50-episode rolling reward average escalated to an all-time project record of **$+450.75$**.
   * Progression over training:
     $$\text{ep1000: } +226.96 \longrightarrow \text{ep2000: } +288.54 \longrightarrow \text{ep3000: } +339.47 \longrightarrow \text{ep4000: } +307.43 \longrightarrow \mathbf{\text{ep5000: } +450.75}$$
2. **Sustained Evaluation High-Reward Plateau:**
   * Evaluation mean rewards at ep4000 ($+349.11$), ep4500 ($+344.47$), and ep5000 ($+318.19$) maintain an elevated plateau well above $+318$.
3. **Largest Policy Shift (diff = 0.6197):**
   * The transition from ep4500 to ep5000 registered the largest policy update of the entire run (`mean_abs_diff = 0.6197`), while output saturation remains well-tempered at 46%.
4. **Final 1,000 Episodes Remaining:**
   * **83.3% completed** (5,000 / 6,000 episodes).
   * **Time Remaining:** only $\approx 21.1\text{ minutes}$ ($\approx 0.35\text{ hours}$).

---

## 6. Checkpoint 6,000 (Final) Diagnostics & Run 3 Completion

**Recorded:** 2026-08-29  
**Checkpoints detected on Drive:** `[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]`

### Cell 4 — Actor Outputs & Saturation Check Across Checkpoints

Fixed test states (8 fixed random states, uniform $[-1, 1]^{26}$, seed 999):

#### ep5500 (total_updates = 1,077,137)
```text
[[ 0.8916  0.9997  0.8444]
 [ 0.9998  1.      0.8607]
 [-1.      0.9991  0.906 ]
 [ 1.      0.9974  0.8313]
 [ 1.      0.8213  0.9998]
 [ 1.      0.579  -1.    ]
 [ 1.      0.9296  0.6214]
 [ 0.9515  0.9992  0.8449]]
```

#### ep6000 (total_updates = 1,177,137)
```text
[[-0.9969  0.9996  0.9212]
 [ 1.      1.      0.9999]
 [ 1.     -0.1219 -1.    ]
 [ 1.      0.5785  0.333 ]
 [ 1.      0.8422  0.9999]
 [ 0.6936 -0.8584 -1.    ]
 [ 1.      1.      0.9722]
 [ 1.      0.9997  0.8229]]
```

#### Diff & Saturation Metrics
```text
=== Mean abs change between consecutive checkpoints ===
ep500 -> ep1000: mean_abs_diff=0.319208, frac_outputs_saturated(|x|>0.999)=0.38
ep1000 -> ep1500: mean_abs_diff=0.487610, frac_outputs_saturated(|x|>0.999)=0.29
ep1500 -> ep2000: mean_abs_diff=0.376395, frac_outputs_saturated(|x|>0.999)=0.17
ep2000 -> ep2500: mean_abs_diff=0.503343, frac_outputs_saturated(|x|>0.999)=0.33
ep2500 -> ep3000: mean_abs_diff=0.488897, frac_outputs_saturated(|x|>0.999)=0.46
ep3000 -> ep3500: mean_abs_diff=0.500188, frac_outputs_saturated(|x|>0.999)=0.38
ep3500 -> ep4000: mean_abs_diff=0.488374, frac_outputs_saturated(|x|>0.999)=0.54
ep4000 -> ep4500: mean_abs_diff=0.453904, frac_outputs_saturated(|x|>0.999)=0.50
ep4500 -> ep5000: mean_abs_diff=0.619741, frac_outputs_saturated(|x|>0.999)=0.46
ep5000 -> ep5500: mean_abs_diff=0.597025, frac_outputs_saturated(|x|>0.999)=0.50
ep5500 -> ep6000: mean_abs_diff=0.429415, frac_outputs_saturated(|x|>0.999)=0.58

=== INTERPRETATION ===
OK: actor output is still changing between checkpoints (mean diff = 0.4294) -- training appears active, not frozen.
```

---

### Cell 5 — Live Training Progress, Pace, and ETA Visualizer

```text
=================================================================
TRAINING PROGRESS: [██████████████████████████████] 100.0% (6000/6000 eps)
=================================================================
  Pace: 10.6 min per 500 episodes (0.79 ep/s)
  Time Remaining (ETA): ~0.0 minutes (0.00 hours)
  Estimated Completion: 08:38 AM (2026-08-29)

Live Rewards: 6000 episodes logged.
  Current Reward: 304.99
  Rolling Avg (last 50): 342.16
```

---

### Cell 6 — Complete Cross-Training Movement & Corner-Escape Trajectory

10 evaluation seeds (0–9) rollout across all saved checkpoints:

| Episode | Mean Max Displacement | Frac > 50m | Mean Reward | Progression Phase |
| :---: | :---: | :---: | :---: | :--- |
| **500** | 0.0m | 0.0% | 243.58 | Early exploration; policy near start |
| **1000** | 142.0m | 40.0% | 318.96 | Initial movement breakout |
| **1500** | 32.6m | 30.0% | 228.05 | Policy reorganization; saturation drop |
| **2000** | 54.5m | 20.0% | 321.95 | Low saturation (17%), high reward |
| **2500** | 50.2m | 10.0% | 263.45 | Foundation for second ascent |
| **3000** | 103.3m | 20.0% | 236.35 | Halfway mark; live arrival reward (+1091) |
| **3500** | 131.9m | 30.0% | 256.33 | Steady flight expansion |
| **4000** | **222.7m** | **40.0%** | **349.11** | **Global Peak: Farthest flight (222.7m) & highest reward (+349.11)** |
| **4500** | 203.5m | 30.0% | 344.47 | Sustained high performance (+344.47, 203.5m) |
| **5000** | 98.0m | 20.0% | 318.19 | Peak rolling training average (+450.75) |
| **5500** | 0.0m | 0.0% | 269.05 | Boundary consolidation |
| **6000** | 63.0m | 10.0% | 254.84 | Final checkpoint complete; live reward +304.99 |

---

## 7. Diagnostic Assessment: Policy Not Yet Validated for Downstream Use

> [!WARNING]
> **Checkpoints Not Ready for M11–M14 Baseline Comparisons:**
> While Run 3 successfully eliminated the dead-actor freeze of Run 1 and the total stand-still collapse ($v=0$) of Run 2, **the policy has not consolidated goal navigation**.
> Independent 30-seed noise-sensitivity evaluation shows a **median displacement of 0.0m** and **0.0% arrival rate** across all evaluation noise levels ($\sigma_{\text{eval}} \in [0.0, 0.5]$) on both `ep4000` and `ep6000`.

### Full Noise Sensitivity Sweep (30 seeds, k=10):

**td3_agent_ep6000.pt (Final Checkpoint):**
| sigma_eval | Mean Max Disp (m) | Median Disp (m) | Frac > 50m (%) | Arrival Rate (%) | Mean Reward | Std Reward |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | 21.0m | 0.0m | 3.3% | **0.0%** | +268.19 | 83.33 |
| **0.05** | 22.0m | 0.0m | 3.3% | **0.0%** | +278.10 | 90.84 |
| **0.10** | 21.9m | 0.0m | 3.3% | **0.0%** | +282.32 | 102.69 |
| **0.20** | 41.2m | 0.0m | 6.7% | **0.0%** | +298.15 | 139.28 |
| **0.30** | 41.0m | 0.0m | 6.7% | **0.0%** | +297.78 | 143.18 |
| **0.50** | 45.0m | 0.0m | 16.7% | **0.0%** | +277.19 | 113.26 |

**td3_agent_ep4000.pt:**
| sigma_eval | Mean Max Disp (m) | Median Disp (m) | Frac > 50m (%) | Arrival Rate (%) | Mean Reward | Std Reward |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** | 144.1m | 0.0m | 26.7% | **0.0%** | +303.65 | 349.81 |
| **0.05** | 143.9m | 0.0m | 30.0% | **0.0%** | +354.52 | 347.53 |
| **0.10** | 146.8m | 0.0m | 36.7% | **0.0%** | +372.75 | 335.40 |
| **0.20** | 195.6m | 0.0m | 40.0% | **0.0%** | +458.36 | 352.02 |
| **0.30** | 221.2m | 0.0m | 40.0% | **0.0%** | +505.40 | 365.49 |
| **0.50** | 149.1m | 34.0m | 43.3% | **0.0%** | +378.12 | 231.88 |

### Key Diagnostic Takeaways:
1. **0% Arrival Across the Board:** Neither checkpoint reaches $Q_{\text{END}}$ in any evaluation seed, even under substantial perturbation ($\sigma=0.5$).
2. **Median Inaction:** Over 50% of seeds remain trapped at 0.0m displacement due to boundary cancellations at step 1 from corner start $Q_{\text{START}}$.
3. **Noised Flight vs. Consolidation:** While training episodes occasionally hit arrival-scale rewards (>700), this was driven by early heuristic prior-knowledge exploration ($R_{\text{rand}}$) and exploratory noise rather than a consolidated deterministic policy.
4. **Current Status:** Do not recommend any checkpoint as ready for downstream baselines until further diagnostic investigation resolves why the policy does not complete navigation.

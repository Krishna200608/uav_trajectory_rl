# Colab Training, Arrival Rate & Q1 Spread Diagnostics (Run 4)

This document tracks the real-time diagnostic checks executed via [`check_actor_saturation_colab.ipynb`](../notebooks/check_actor_saturation_colab.ipynb) during the full 6,000-episode Google Colab training run with **probabilistic annealed prior-knowledge handoff** (`--r-rand 20000 --anneal-steps 20000 --checkpoint-every 250`).

Checks are performed at milestones:
- [x] **Episode 1,000** (Logged below)
- [ ] **Episode 3,000** (Pending)
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

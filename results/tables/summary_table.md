# Consolidated Comparison Table (All 5 Methods, k=10, 10 Seeds)

| Method | Arrival Rate | Mean Steps | Min-Dist-to-Goal (m) | LoS Probability | Throughput (Mbps) | Energy (J/step) | DTE | Behavior |
|---|---|---|---|---|---|---|---|---|
| TDPK | 10/10 (100%) | 81.80 ± 5.83 | 2.85 ± 1.41 | 0.30 ± 0.06 | 103.91 ± 5.79 | 184.57 ± 3.63 | 4.58 ± 0.29 | Direct-line geometric flight toward destination; ignores ground users entirely |
| Greedy | 10/10 (100%) | 200.00 ± 0.00 | 2.63 ± 1.17 | 0.18 ± 0.05 | 102.86 ± 22.78 | 59.49 ± 2.55 | 4.94 ± 1.15 | 1-step lookahead; stalls near destination to avoid early-termination penalty until step 199 |
| DuelingDQL | 0/10 (0%) | 200.00 ± 0.00 | 320.14 ± 77.91 | 0.95 ± 0.01 | 182.11 ± 10.27 | 435.23 ± 104.90 | 7.65 ± 0.72 | Discrete Q-learning; settles mid-field to maximize throughput without reaching destination |
| PPO | 0/10 (0%) | 200.00 ± 0.00 | 233.85 ± 74.99 | 0.94 ± 0.01 | 180.35 ± 9.21 | 152.36 ± 44.11 | 8.51 ± 0.40 | Continuous actor-critic; progresses ~700m+ but freezes at eastern boundary wall from azimuth bias |
| PKTD3-TD | 0/10 (0%) | 200.00 ± 0.00 | 848.53 ± 0.00 | 0.05 ± 0.01 | 53.39 ± 9.28 | 84.19 ± 128.75 | 2.39 ± 0.45 | Documented non-convergence (flat value surface); actor saturates and remains locked near Q_START |

> **Note on PKTD3-TD's energy variance:** its high energy std (relative to mean) reflects the non-convergent actor saturating toward either steep-climb (`λ≈0`) or steep-descent (`λ≈π`) commands depending on per-seed state, producing large sign-flipped climb-power terms even though the UAV's position never changes in any seed (energy is charged for commanded motion, not realized displacement). This is a documented consequence of the already-closed training-instability investigation, not a new defect.

> **Note on sample size:** this table uses the same 10-seed sample as all other M14 figures for internal consistency across the evaluation suite. This differs from M11/M12's original 30-seed deterministic evaluation (310.6m/263.1m for Dueling DQL/PPO respectively); both samples are independently verified and consistent with each other, the difference reflects sample size, not a discrepancy.

# PKTD3-TD Implementation Tracker — DMMSN Group 5

**Paper:** M. Li, M. Dong, H. Wang, H. Wang, "3-D Trajectory Design Based on Deep Reinforcement Learning for UAV-Assisted Communication Networks," *IEEE Trans. Network Science and Engineering*, vol. 13, 2026.
**Team:** Krishna Sikheriya, Kushagra Gupta, Aditya Pankaj Sharma, Vaibhav Sharma
**Supervisor:** Nayanjit Talukdar (PhD Scholar, WSN Lab, Dept. of IT, IIIT Allahabad)
**Toolchain:** Google Antigravity (Gemini Flash 3.7 high) executes code; Claude architects modules, writes prompts, reviews output against the paper.

---

## 1. Ground-truth parameters (extracted directly from the paper's tables/text)

### Table II — Reward setting
| Param | Value | Param | Value |
|---|---|---|---|
| c_th | 1/(2×10⁷) | c_h | 0.5 |
| c_ar | 1 | c_en | 1/300 |
| c_nr | 20 | c_near | 0.5 |
| c_ac | 0.5 | c_lack | 1 |

### Table III — Algorithm parameters
| Param | Value | Param | Value |
|---|---|---|---|
| M (max episodes) | 6000 | C (air resistance coeff.) | 0.2 |
| R (replay buffer size) | 10⁵ | W (UAV weight) | 20 |
| M_b (mini-batch size) | 128 | S_FP (fuselage flat-plate area) | 0.0151 |
| γ (discount factor) | 0.96 | P0 (blade profile power, hover) | 80.182 |
| τ (target soft-update) | 0.005 | A (rotor disc area) | 0.503 |
| c (action clip bound) | 1 | s (rotor solidity) | 0.05 |
| d (actor/target update delay) | 2 | ρ (air density) | 1.23 |
| λ_pk (prior-knowledge polar angle) | 0.5π | d0 (fuselage drag ratio) | 1.23 |
| ρ_pk (prior-knowledge max azimuth) | 0.5π | U_tip (rotor tip speed) | 120 |
| R_rand (# prior-knowledge steps) | 20000 | σ1, σ2, σ3, σ̃ | 1, 0.25π, 0.1, 0.2 |

`σ1,σ2` = std devs of the Gaussian-Markov user-mobility noise terms ψ_k, φ_k (eq. 4–5). `σ3` = exploration noise ε added to actor output (eq. 31). `σ̃` = target-policy smoothing noise (eq. 38).

### Section V-A simulation setup
- Service area 600 m × 600 m; z_min = 50 m, z_max = 200 m
- q_s = [0,0,50], q_e = [600,600,50]; T = 200 s, δ = 1 s → N = 200 slots
- v0 = 0 m/s, v_max = 20 m/s, ac_max = 5 m/s²
- p_k = 10 dBm (per-user tx power), B_u = 20 MHz
- b1 = 9.61, b2 = 0.16, η_LoS = 1 dB, η_NLoS = 20 dB
- w1 = 1/(2×10⁷) [numerically == c_th], w2 = 1/300 [numerically == c_en]
- Learning rate = 0.0001 (actor & critic, Adam); networks = 2 hidden layers × 256 neurons; actor activation tanh, critic activation ReLU

### ⚠️ Parameters the paper does NOT give numeric values for (flagged, not invented)
- **f_c** (carrier frequency) — appears in the free-space path-loss term LF = 20log(r) + 20log(f_c) + 20log(4π/v_c), eq. (10)-(11), but no number is stated in the text, Table II, or Table III. *(SUPERSEDED: initial 2.0 GHz placeholder revised to 2.4 GHz ISM band per Channel Calibration Revision in Review notes below).*
- **N0** (noise power spectral density) — appears in the transmission-rate denominator, eq. (13). Not numerically specified. *(Preserved at standard thermal noise floor -174 dBm/Hz).*
- **ω** (Gaussian-Markov tuning parameter, "0 ≤ ω ≤ 1") — controls how much of next-slot user velocity/direction comes from memory vs. drift vs. noise, eq. (4)-(5). No numeric value given.
These three will need documented placeholder values (config.py flags them clearly as `# ASSUMPTION, not from paper`) unless the team can pull them from the cited references [116]/[117]/[115].

### ⚠️ Discrepancy found between eq. (29) and Algorithm 1 pseudocode
Eq. (29) defines `r_n = r_n,1 + r_n,2 + r_n,3 + r_n,4 + r_n,5 + r_n,6` (six terms, includes the height-constraint penalty r_n,6).
Algorithm 1, line 15, lists `r_n = r_n,1+r_n,2+r_n,3+r_n,4+r_n,5` — **omits r_n,6**.
**Decision for this project: implement all six terms (eq. 29 is the complete, formal definition); treat the pseudocode line as a likely typo.** Flag this to the supervisor if asked.

### Table IV — Robustness to user position error (σ_loc)
| σ_loc | 0 | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 |
|---|---|---|---|---|---|---|
| LoS prob. | 0.88 | 0.87 | 0.87 | 0.86 | 0.83 | 0.83 |
| Throughput (×10⁷ bits) | 2.74 | 2.71 | 2.71 | 2.72 | 2.70 | 2.70 |
| Energy (J) | 133.83 | 136.59 | 134.10 | 138.92 | 144.78 | 146.71 |

### Table V — Robustness to sensing latency (slots)
| Latency | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| LoS prob. | 0.88 | 0.89 | 0.84 | 0.86 | 0.83 | 0.82 |
| Throughput (×10⁷ bits) | 2.74 | 2.76 | 2.71 | 2.72 | 2.69 | 2.67 |
| Energy (J) | 133.83 | 136.02 | 139.22 | 137.59 | 142.88 | 148.84 |

### Table VI — Robustness to mobility mutation
| Magnitude | 0.2 | 0.2 | 0.6 | 0.6 | 1.0 | 1.0 |
|---|---|---|---|---|---|---|
| Probability | 0.02 | 0.06 | 0.02 | 0.06 | 0.02 | 0.06 |
| LoS prob. | 0.86 | 0.82 | 0.85 | 0.80 | 0.82 | 0.78 |
| Throughput (×10⁷) | 2.74 | 2.70 | 2.73 | 2.69 | 2.71 | 2.66 |
| Energy (J) | 131.39 | 135.51 | 133.92 | 136.51 | 137.85 | 146.15 |

### Baselines (Section V-A) to reproduce for comparison
- **TDPK**: prior-knowledge-only policy — fly straight toward destination, per-slot speed randomly generated.
- **Dueling DQL** [47]: discrete-action baseline; balances flight time vs. throughput via a scaling factor.
- **PPO** [44]: same state/reward design as PKTD3-TD, continuous actions.
- **Greedy** [83]: picks the action maximizing the *immediate* objective each slot (myopic).

---

## 2. Module tracker

| ID | Module | Depends on | Status |
|---|---|---|---|
| M0 | Shared config / constants | — | **Done — reviewed, approved** |
| M1 | UAV kinematics (eq. 1–3) | M0 | **Done — reviewed, approved** |
| M2 | User mobility — Gaussian-Markov (eq. 4–7) | M0 | **Done — reviewed, approved** |
| M3 | Channel model — LoS/path-loss/rate (eq. 8–14) | M0 | **Done — reviewed, approved (eq. 13 fix hand-verified: log2(SNR), no +1)** |
| M4 | UAV energy/propulsion model (eq. 15–16) | M0 | **Done — reviewed, approved (hand-verified numerically)** |
| M5 | MDP env wrapper: state/action/reward/step (eq. 17–29) | M1–M4 | **Done — reviewed, approved (Claude hand-verified all 6 reward terms exactly, incl. xy-cancellation and terminal-arrival edge cases; state normalization added per CRITICAL FIX)** |
| M6 | Prior-knowledge exploration policy (eq. 30–31) | M5 | **Done — reviewed, approved (un-normalization boundary cases hand-verified, incl. R_ex==R_rand edge)** |
| M7 | TD3 networks + replay buffer | M0 | **Done — reviewed, approved (shapes, q1_forward consistency, and circular-buffer overwrite hand-verified)** |
| M8 | TD3 update rules: clipped double-Q, delayed update, target smoothing (eq. 32–38) | M7 | **Done — reviewed, approved (delayed-update cadence and terminal-target zeroing hand-verified; gradient clipping added per CRITICAL FIX)** |
| M9 | Training loop / full Algorithm 1 | M5, M6, M7, M8 | **Done (code verified; Run 3 completed but unvalidated: 0% arrival rate on noise sweep, under active diagnosis)** |
| M10 | Baseline: TDPK | M5 | **Done — reviewed, approved (geometry hand-verified: diagonal, vertical, and degenerate same-point cases all match spec exactly)** |
| M11 | Baseline: Dueling DQL | M5 | Not started |
| M12 | Baseline: PPO | M5 | Not started |
| M13 | Baseline: Greedy | M5 | Not started |
| M14 | Evaluation & plotting suite (Figs. 4–12, Tables IV–VI) | M9–M13 | Not started |

**Suggested parallelization:** once M0 lands, M1/M2/M3/M4 are mutually independent — good four-way split across the team.

---

## 3. Review notes (log of mismatches found during code review)

### M3 — channel_model.py — BUG: eq. (13) incorrectly includes "+1" in the log
Original M3 prompt (drafted by Claude) instructed `log2(1 + SNR)`, defaulting to the textbook Shannon-capacity form from general knowledge. Pixel-level inspection of the actual typeset equation on page 6 confirms the paper's literal eq. (13) is `log2(SNR)` — no "+1". This is a Claude drafting error, not a Gemini implementation error; Gemini correctly implemented what it was asked to implement.
**Fix:** drop the "+1"; `rate_bps = bandwidth_per_user * log2(snr)`. This can legitimately go negative when SNR < 1 — do not clip. Flagged for awareness at M9 (training loop / reward): a negative throughput term is expected paper-faithful behavior, not a bug, when links are weak.
**Status:** fix sent back to team, awaiting re-verification.

### M3 — eq. (8) LoS probability — ambiguous PDF typesetting, resolved, no code change needed
Pixel-level zoom on eq. (8) appears to show `exp(-b2*θ - b1)` rather than the standard Al-Hourani `exp(-b2*(θ-b1))` that the code implements. Numerically checked both readings across θ=0°–90°: the "literal" reading gives P_LoS ≈ 0.999+ at *every* angle (physically implausible, and inconsistent with the paper's own reported LoS values of 0.78–0.89 in Tables IV–VI). The standard sigmoid reading (what the code currently does) produces a believable 0.02→0.9998 sweep and matches the paper's own reported numbers. Conclusion: keep the code as-is; likely a typesetting/rendering quirk in the source PDF, not a real alternate formula.

### Config — σ1, σ2, σ3, σ̃ interpretation (std dev vs. variance)
The paper's Notation section defines `N(μ, σ²)` with the second argument as *variance*, but eq. (4)-(5), (31), (38) write `N(0, σ1)`, `N(0, σ3)` etc. — the bare symbol, not squared. Taken at face value against the paper's own notation rule, this would mean σ1/σ2/σ3/σ̃ (Table III values) are variances, and the actual std dev fed into a Gaussian sampler should be their square roots. Current code (config + user_mobility.py) uses the Table III values directly as `scale=` (std dev) in `np.random.normal`, matching standard TD3-literature convention (exploration/smoothing noise is conventionally parameterized by std dev directly) rather than the paper's own stated notation rule. Decision: keep as implemented (std dev = table value directly) for consistency with TD3 convention; documented here as a known ambiguity in case results look off and this needs revisiting.

### M5 — mdp_environment.py — eq. (26) corrected: q_s to q_e
The paper's typeset eq. (26) for d_near_n uses Q_START (q_s), but its own prose description ("decrease in distance to the destination") and the reward's purpose both require Q_END (q_e). Implemented using Q_END; documented as a corrected typo in the source paper.

### M5 — r_n,3 terminal-condition interpretation
Eq. (23)'s condition "N = T/delta" is interpreted as "this is the terminal step of the episode" (which can occur before N_SLOTS via early arrival, per Algorithm 1's loop-termination rule), not a literal equality against the constant N_SLOTS — otherwise early-arrival episodes would never receive this reward term, defeating its purpose.

### M5 — new assumption: ARRIVAL_THRESHOLD_M = 5.0 m
Paper gives no numeric tolerance for constraint C6 (exact arrival q_N = q_e). Added as a documented assumption in config.py.

### M6 — prior_knowledge_policy.py — un-normalization mapping is a Claude design decision
The paper specifies eq. (31)'s clipping happens in the actor's normalized
[-c,c] output space but never specifies how that maps to the physical
action ranges (which aren't all symmetric about zero). Implemented as an
explicit affine mapping (see module docstring for exact formulas) --
flagged here since this is an engineering decision, not a value taken
from the paper.

### M9 design note — total_updates vs. Algorithm 1's slot index n
TD3Agent.train_step's policy_delay gating counts CALLS to train_step, not
the raw environment time-slot n from Algorithm 1. Since train_step is only
called once R_ex > R_rand (matching Algorithm 1 Line 17's gate), the
RELATIVE cadence (actor+target update every policy_delay training calls)
is correct and is what matters for TD3 stability, but it is not
phase-aligned with the paper's absolute slot index n. Inconsequential to
correctness; noted for completeness.

### M4 fix — energy_model.py — pitch angle singularity clamp (±70°)
In eq. (15), τ_c = π/2 - λ_n approaches ±π/2 during near-vertical climb (λ_n -> 0)
or descent (λ_n -> π). The term cos(τ_c) in the denominator and tan(τ_c) in climb
power previously caused division by near-zero (1e-7), producing unphysical
power of 2.5 trillion Watts and blowing episode rewards to -8.3e11 once the
neural network exploration branch took actions. Pitch angle τ_c is now aerodynamically
bounded to [-70°, +70°] (matching physical multicopter flight envelopes) and total
power is floored at 0.0 W, stabilizing rewards into the healthy +200 to +800 range.

### M7 fix — td3_networks.py — automatic device alignment on CUDA
When running on GPU accelerators (e.g. Google Colab T4), Actor and TwinCritic
networks reside on cuda:0. Forward passes now automatically ensure input tensors
reside on the network's active device (self.device) before executing layers,
resolving CPU/CUDA device mismatch errors in tests and inference.

### CRITICAL FIX — actor saturation traced to unnormalized state input
The full 6000-episode run1 training (checkpoints/run1) produced a DEAD
actor: verified by loading checkpoints at episodes 500/1000/3000/6000 and
confirming bit-identical, tanh-saturated (+-1) outputs on fixed test states
across all of them -- the actor stopped learning almost immediately and ran
the remaining ~5500 episodes as a fixed, state-independent controller. This
fully explains the flat ~-270 reward plateau reported after training.
ROOT CAUSE: the state vector fed to the actor/critic networks mixed raw
physical quantities of very different scales (positions to 600, distances
to ~848, time to 200, speed to 20) with no normalization, a well-known
cause of saturated pre-activations in bounded-output (tanh) actors.
FIX: state normalization added to UAVTrajectoryEnv._build_state() (all
components now roughly in [-1,1] or [0,1]), plus gradient norm clipping
(max_norm=10.0) added to TD3Agent.train_step as an additional stability
safeguard. Both are documented engineering design decisions, not paper-
specified values.
checkpoints/run1 IS INVALID and must not be used for baseline comparisons
or M14 evaluation plots -- retain it in the repo for the record (do not
delete), but treat it as a documented failed run, not a completed result.

### Run 2 (checkpoints/run2) — INVALID: Stand-still policy collapse (v=0 at every step)
checkpoints/run2 IS INVALID: the actor collapsed to a stationary v=0 policy; confirmed via direct
critic inspection, not just reward trend.
Detailed findings:
1. Direct rollout verification across 5 different seeds and checkpoints spanning ep500 through ep6000
   revealed that the trained actor outputs normalized action a_norm = [-1.0, 1.0, -1.0], which
   maps directly to commanded speed v = 0.0 m/s at every single step. The UAV never moves a single
   meter from Q_START = (0, 0, 50).
2. Direct critic inspection confirms why: Q1(state, action) evaluated on the initial state s_0
   decreases monotonically from ~367 at v = 0 m/s down to ~347 at v = 20 m/s across all flight
   directions tested (holding directions fixed). The critic learned that "standing still" is optimal,
   and the actor faithfully follows that gradient.
3. Retain checkpoints/run2 in the repository as a documented failed run (do not delete), but do
   NOT use for baseline comparisons or M14 evaluation.

### Diagnostic Study: Cancelled-move energy charging hypothesis
HYPOTHESIS:
In mdp_environment.py, energy (r_n,2) was originally charged on commanded speed even when the UAV's
position update was cancelled due to spatial boundary violations (xy area or z bounds) -- an undocumented
M5 engineering design decision. Because the UAV starts at the corner Q_START = (0, 0, 50), early random/
exploratory actions frequently violated boundaries and were cancelled, expending full energy for 0 progress
while standing still incurred no boundary penalties. This was hypothesized to create an incentive for inaction.

METHOD:
Added a configurable toggle `charge_energy_on_cancelled_move: bool = True` to UAVTrajectoryEnv and
`--no-charge-on-cancel` flag to scripts/train.py. Two identical 800-episode runs (seed 0) were executed:
- Run A (Baseline, charge ON): default behavior.
- Run B (Hypothesis, charge OFF): when an action is cancelled, v_n = 0.0 is used for energy calculation.

RESULTS (10-seed rollout evaluation at ep800):
- Run A (Charge ON):
  - Final/Max displacement: 0.0m / 0.0m across ALL 10 seeds (0/10 exceeded 50m).
  - Actor output: a_norm = [1.0, 0.9999, 0.9896] -> commanded v = 20 m/s directly into the floor/wall
    (lam = pi, rho = pi), getting cancelled at 100% of steps.
  - Critic Q1(s0, action): slightly higher at v = 20 m/s (61.05) vs v = 0 m/s (60.58).
- Run B (Charge OFF):
  - Final/Max displacement: 0.0m / 0.0m in 9 out of 10 seeds (Seeds 0-8). Seed 9 escaped and achieved
    227.9m final displacement (231.6m max) with reward +392.84.
  - Actor output: a_norm = [-0.9998, 0.9676, 0.9985] -> commanded v = 0.0 m/s (stationary).
  - Critic Q1(s0, action): Q1 at v = 0 m/s (71.67) remains HIGHER than v = 20 m/s (71.27) in forward
    directions, and higher into boundaries (74.59 vs 73.35). The "lower speed = higher Q" preference persists.

CONCLUSION & ASSESSMENT:
The hypothesis that charging energy on cancelled moves was the primary driver of stand-still collapse is
RULED OUT (or at best only weakly partial). While removing the penalty allowed 1 of 10 seeds to escape
and move, 9 of 10 seeds still collapsed to 0 displacement, and the critic still exhibits a higher Q-value
for standing still than for moving forward. The root incentive problem lies deeper in the reward structure
(e.g., terminal penalty r_n,3 vs per-step throughput and energy trade-offs, or the proximity reward r_n,4).

### Reward-baseline diagnostic (hover vs TDPK vs prior-knowledge policy)
Investigated whether the reward function intrinsically favors hovering over genuine movement,
and whether the replay buffer ever contained successful arrivals during the prior-knowledge (PK) phase.

1. THREE-WAY POLICY COMPARISON (20 Seeds, K=10, 200 steps):
   - Always Hover: Mean reward = 216.07 +/- 79.55 | Arrival rate = 0.0% (0/20)
   - TDPK (Direct Flight): Mean reward = 345.26 +/- 39.39 | Arrival rate = 100.0% (20/20) | Avg steps to arrival = 89.1
   - Prior-Knowledge Policy: Mean reward = 653.46 +/- 261.97 | Arrival rate = 50.0% (10/20)
   TDPK beats Always Hover by +129.19 points on average (beating hover in 17 of 20 seeds).
   Finding: The reward function itself does NOT intrinsically prefer hovering; a policy that actively flies
   to the destination earns substantial positive throughput and avoids non-arrival penalties.

2. PK-PHASE ARRIVAL RATE AUDIT (150 Episodes, K=10):
   - Using the exact `generate_prior_knowledge_action` heuristic from eq. (30):
     - Total arrivals: 60 / 150 (40.00% arrival rate).
     - Steps on arrival: Min = 85, Max = 192, Mean = 121.1 steps.
     - Distance to goal at termination: Mean = 27.01m, Min = 0.83m.
   Finding: The replay buffer during the R_rand = 20,000 exploration phase (100 episodes) received ~40
   episodes of successful arrivals, disproving the hypothesis that the critic lacked grounding arrival data.

3. ROOT CAUSE & INTERPRETATION:
   - Neither the reward balance (hover vs TDPK) nor the lack of arrival examples during exploration explains
     the stand-still collapse.
   - Decisive architectural insight: An action-space representation mismatch exists in the TD3 training loop.
     In scripts/train.py, un-normalized physical actions `(v, lam, rho) in [0, 20] x [0, pi] x [-pi, pi]`
     were stored into `replay_buffer`, so the critic was trained on physical actions. However, the actor target
     and actor update loss evaluate the critic using normalized actions `[-1, 1]^3`. When the actor outputs
     an action in `[-1, 1]`, the critic interprets it in physical units (e.g., speed <= 1.0 m/s), severely
     distorting policy gradients and causing the actor to collapse to the lower bound `v_norm = -1.0` (0 m/s).

### Action-Scale Mismatch Resolution & Replay Buffer Normalization
TRUE ROOT CAUSE OF RUN 1 & RUN 2 PATHOLOGY:
The fundamental training pathology across both run1 and run2 was an action-space representation
mismatch between the replay buffer and the TD3 networks, independent of (and in addition to) the
earlier energy-singularity and state-normalization issues:
1. `scripts/train.py` was storing raw physical actions `(v, lam, rho) in [0, 20] x [0, pi] x [-pi, pi]`
   into the replay buffer. Consequently, the critic Q1/Q2 networks were trained via MSE on physical-scale
   actions where speed spanned 0 to 20 m/s.
2. However, TD3Agent._compute_target (`critic_target(next_states, next_action)`) and the actor loss
   (`-critic.q1_forward(states, actor(states))`) feed the actor's live output, which is bounded in
   normalized space `[-c, c]^3 = [-1, 1]^3` (Actor.forward computes tanh * max_action).
3. The critic was therefore queried during target computation and policy gradient updates in a scale it
   was never trained on. A normalized speed output of +1.0 was perceived by the critic as a crawling
   physical speed of 1.0 m/s, while -1.0 was perceived as -1.0 m/s.

EXPLICIT CORRECTION TO EARLIER "CRITIC PREFERS STANDING STILL" DIAGNOSTIC:
The earlier diagnostic note concluded that the critic preferred standing still because querying
critic.q1_forward with raw inputs in [-1, 1] showed lower Q-values for positive values. That diagnostic
made the exact same scale error: it probed the critic in [-1, 1], which represented near-hovering
speeds (<= 1 m/s) in physical space. When the trained critic is correctly probed across genuine physical
scales (v = 0 to 20 m/s), Q1 increases monotonically with speed toward the destination (347 at v=0 up to
355 at v=20). The critic actually learned the correct physical relationship; the networks were merely
disconnected by the action representation mismatch.

FIX IMPLEMENTED:
1. Added `normalize_action()` to `prior_knowledge_policy.py`, the exact algebraic inverse of
   `unnormalize_action()`. Verified with 1,000 random vectors in `tests/test_prior_knowledge_policy.py`
   round-tripping to machine precision (atol=1e-9).
2. Updated `scripts/train.py` so that `replay_buffer.add()` stores `normalize_action(action)` in `[-c, c]^3`,
   unifying the replay buffer, target network smoothing, critic evaluation, and actor gradients in the
   identical normalized `[-1, 1]^3` domain.

800-EPISODE DIAGNOSTIC EVALUATION (`checkpoints/diag_actionscale_fix`):
1. CRITIC SANITY & CONSISTENCY CHECK (td3_agent_final.pt on initial state s0 heading toward goal):
   - Q1 evaluated via `normalize_action(phys)` vs direct normalized tensor matches IDENTICALLY across all speeds:
     - v = 0.0 m/s (v_norm = -1.00) -> Q1 = 48.0553 (Exact Match: YES)
     - v = 5.0 m/s (v_norm = -0.50) -> Q1 = 49.1367 (Exact Match: YES)
     - v = 10.0 m/s (v_norm = 0.00) -> Q1 = 50.3100 (Exact Match: YES)
     - v = 15.0 m/s (v_norm = +0.50) -> Q1 = 50.9506 (Exact Match: YES)
     - v = 20.0 m/s (v_norm = +1.00) -> Q1 = 51.6965 (Exact Match: YES)
   - The critic shows a clean, strictly monotonic increase with speed toward the goal.
2. BEHAVIORAL DISPLACEMENT ACROSS CHECKPOINTS (10 seeds each):
   - ep200: Mean final disp = 0.0m | Mean reward = 274.26
   - ep400: Mean final disp = 18.1m (Max dist up to 142.6m; Seed 3 = 142.6m, Seed 8 = 58.7m; 2/10 exceed 50m) | Mean reward = 263.98
   - ep600: Mean final disp = 6.8m (Seed 4 = 68.0m; 1/10 exceed 50m) | Mean reward = 230.94
   - ep800/final: Mean final disp = 20.5m (Seed 4 = 80.7m, Seed 5 = 124.7m; 2/10 exceed 50m) | Mean reward = 148.30
3. Training rewards remained consistently positive (+200 to +300) across all 800 episodes, proving active,
   healthy policy exploration and breaking the stand-still stagnation of run1 and run2.

### Extended 2,500-Episode Action-Scale Diagnostic Trend (`checkpoints/diag_actionscale_2500`)
Extended the diagnostic training run to 2,500 episodes (saving every 250 episodes) with the normalized
action fix to evaluate whether the moving-seed fraction exhibits an upward learning trend across training.

EVALUATION PROTOCOL:
- 10 evaluation seeds (0-9) tested per checkpoint against a fresh environment.
- Metrics recorded: Mean Max Displacement from Q_START, Fraction of Seeds Exceeding 50m, Mean Reward.

SUMMARY TREND TABLE ACROSS CHECKPOINTS:
| Checkpoint | Mean Max Disp (m) | Frac Exceeding 50m | Mean Reward | Seeds Exceeding 50m (Max Distance in m) |
|---|:---:|:---:|:---:|---|
| **ep250**  | 0.0m  | 0.0%  | +247.82 | None |
| **ep500**  | 28.4m | 30.0% | +255.92 | Seed 3 (109m), Seed 4 (56m), Seed 8 (109m) |
| **ep750**  | 6.8m  | 10.0% | +214.09 | Seed 4 (66m) |
| **ep1000** | 6.2m  | 10.0% | +202.52 | Seed 8 (61m) |
| **ep1250** | 64.2m | 30.0% | +222.97 | Seed 3 (104m), Seed 8 (283m), Seed 9 (240m) |
| **ep1500** | 0.3m  | 0.0%  | +164.25 | None |
| **ep1750** | 61.1m | 20.0% | +193.28 | Seed 0 (452m), Seed 1 (159m) |
| **ep2000** | 8.5m  | 10.0% | +114.45 | Seed 8 (85m) |
| **ep2250** | 27.1m | 20.0% | +74.05  | Seed 0 (60m), Seed 9 (211m) |
| **ep2500 / Final** | 0.0m | 0.0% | +91.54 | None (all seeds collapsed to 0m) |

ANALYSIS & INTERPRETATION:
1. Trend Read: FLAT / NOISY, NOT TRENDING UPWARD.
   - The fraction of seeds moving meaningfully fluctuates between 0% and 30% without sustained improvement:
     `0% -> 30% -> 10% -> 10% -> 30% -> 0% -> 20% -> 10% -> 20% -> 0%`.
   - Mean reward across evaluation seeds steadily degrades from ~+255 down to ~+91.
2. Root Cause of Continued Inaction in 70-100% of Seeds:
   - Direct inspection of the actor output vectors reveals that the actor is NOT commanding v=0 (as in run2).
     Instead, the actor commands high speeds ($v = 15 \dots 20\text{ m/s}$), but commands directions pointing
     downward into the ground ($\lambda \approx 160^\circ \dots 180^\circ$) or backward into the boundaries
     ($\rho \approx -120^\circ \dots 180^\circ$).
   - Because $Q_{\text{START}} = (0, 0, 50)$ is located at the exact lower-left bottom corner of the 3D domain
     $[0, 600] \times [0, 600] \times [50, 100]$, only $1/8$ (12.5%) of spherical angles point into the valid flight
     volume ($\lambda \in [0, \pi/2], \rho \in [0, \pi/2]$). The remaining $7/8$ (87.5%) of directions immediately
     violate spatial boundaries on step 1, causing 100% boundary cancellations.
   - Charging energy on these cancelled moves drains reward while yielding 0 progress, and the actor gets trapped
     in boundary cancellation attractors.

### Run 3 Noise-Sensitivity Diagnostic & Evaluation Assessment (`checkpoints/run3/`)
Full training run executed on Google Colab Tesla T4 GPU across 6,000 episodes (1,177,137 updates) with action-scale and state normalization fixes. Checkpoints synced to `checkpoints/run3/`.

While Run 3 eliminated the dead-actor freeze of Run 1 and the total stand-still collapse ($v=0$) of Run 2, **the policy is NOT yet validated for downstream use (M11–M14)**. Independent 30-seed evaluation revealed that the deterministic policy yields a median displacement of 0.0m and 0% arrival rate on both `ep4000` and `ep6000`, despite 7–14% of training episodes (with exploration noise) hitting arrival-scale rewards (>700).

To determine whether exploration noise was driving navigation and whether the critic's value landscape supports navigation, we executed a noise-sensitivity sweep (`scripts/diagnose_noise_sensitivity.py`) across 30 seeds (0–29, k=10) with $\sigma_{\text{eval}} \in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]$ added to actor output before clipping:

#### Full Sweep Results:

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

#### Honest Diagnostic Interpretation:
1. **Arrival Remains at 0.0% Across All Noise Levels:**
   - Even under strong exploratory perturbation up to $\sigma_{\text{eval}} = 0.5$, neither checkpoint achieves a single arrival out of 30 seeds.
   - This proves that the arrival-scale rewards (>700) observed during training were driven by the early prior-knowledge exploration regime ($R_{\text{ex}} \le R_{\text{rand}} = 20,000$ steps) and occasional lucky noise chains during training, rather than a learned, reproducible navigation policy.
   - The failure to arrive cannot be attributed solely to the lack of test-time noise; something more fundamental is preventing the policy from completing navigation across the service area.
2. **Median Displacement Remains 0.0m for >50% of Seeds:**
   - For both checkpoints across nearly all noise levels, the median displacement is exactly 0.0m. Over half of the evaluation seeds never break out of the initial corner position at all due to boundary cancellation traps at step 1.
3. **Checkpoint Comparison (`ep4000` vs `ep6000`):**
   - `ep4000` is markedly more responsive to noise: its mean max displacement reaches up to 221.2m at $\sigma=0.30$, and 40% of seeds escape past 50m with mean reward reaching $+505.40$.
   - `ep6000` regressed significantly toward the corner: max displacement is only 21–45m, with only 3–17% escaping past 50m.
   - However, because arrival rate remains strictly 0.0% for both, neither checkpoint is ready or recommended for benchmark comparisons.
4. **Conclusion:**
   - The earlier "Pathology Resolution Confirmed" / "ep4000 champion checkpoint" framing is explicitly retracted.
   - Run 3 is an improvement in training stability, but the deterministic navigation capability is NOT consolidated. Further investigation is required before proceeding to M11–M14.

### Reward Component Balance: TDPK vs Hover (`scripts/diagnose_reward_balance.py`)
Investigated whether the throughput-vs-energy reward balance for a full journey provides a large, robust learning signal over hovering, or if our assumed (non-paper-specified) parameters (`FC_HZ`, `N0_DBM_HZ`, `OMEGA`) create a narrow, fragile margin.

Ran 30 seeds ($k=10$) for both TDPK (direct-to-destination flight, arriving in 100% of seeds) and Always-Hover ($v=0$ for the full 200 slots), decomposing cumulative episode reward into its exact 6 constituent terms ($r_1$ through $r_6$).

#### Component Breakdown Table:
| Metric / Component | TDPK (Arrived: 30/30) | Always Hover (30/30) | Delta (TDPK − Hover) | Operational Role |
| :--- | :---: | :---: | :---: | :--- |
| **Mean Steps Taken** | **89.2** / 200 | **200.0** / 200 | **−110.8 steps** | Faster arrival truncates episode |
| **r1 (Throughput)** | **+509.95** | **+641.07** | **−131.12** | **Hover earns MORE total throughput over 200 slots** |
| **r2 (Energy Cost)** | **−54.24** | **−160.66** | **+106.42** | TDPK spends less total energy by terminating earlier |
| **r3 (Terminal Penalty/Bonus)**| **−20.00** | **−62.43** | **+42.43** | TDPK avoids end-of-episode distance penalty $d_{\text{re}}$ |
| **r4 (Proximity / Lack)** | **−65.79** | **−200.00** | **+134.21** | TDPK earns $d_{\text{near}}$ progress and avoids 111 steps of $t_{\text{lack}}$ |
| **r5 (Acceleration Penalty)** | **−24.08** | **+0.00** | **−24.08** | Random speed changes incur acceleration penalty |
| **r6 (Height Penalty)** | **+0.00** | **+0.00** | **+0.00** | Zero altitude boundary violations |
| **Total Cumulative Reward** | **+345.84** | **+217.98** | **+127.85** | **Net Advantage: 1.59x (+58.7%)** |

#### Fraction of Throughput Eaten by Energy Cost:
- **TDPK Full Journey:** Energy cost ($|r_2| = 54.24$) consumes **10.6%** of gross throughput reward ($r_1 = 509.95$). Total negative penalties ($|r_2 \dots r_6| = 164.11$) consume **32.2%** of throughput.
- **Always Hover:** Energy cost ($|r_2| = 160.66$) consumes **25.1%** of gross throughput reward ($r_1 = 641.07$). Total negative penalties ($|r_2 \dots r_6| = 423.09$) consume **66.0%** of throughput.

#### Crucial Findings on the Reward-Balance Hypothesis:
1. **The Net Advantage is NARROW (+58.7%, 1.59x), NOT Large/Robust (not 3–5x):**
   - Completing a full journey yields only a $+58.7\%$ premium over simply hovering stationary at $Q_{\text{START}}$.
   - This narrow margin confirms the hypothesis: under our assumed wireless channel parameters (`FC_HZ`, `N0_DBM_HZ`, `OMEGA`), hovering captures a remarkably strong, risk-free baseline return ($+217.98$).
2. **Surprising Discovery: Hover Earns MORE Total Throughput Than Completing the Mission:**
   - Always-Hover earns **$+641.07$** in throughput vs TDPK's **$+509.95$** ($\Delta r_1 = -131.12$).
   - Because TDPK reaches the destination at step 89.2, it is immediately truncated from accumulating transmission throughput for the remaining 110.8 slots. Hovering sits in place for all 200 slots collecting continuous data from passing users.
3. **Where Does TDPK's Entire Advantage Actually Come From?**
   - TDPK's advantage does **NOT** come from collecting more communication throughput.
   - It comes entirely from **time truncation**: because the episode ends at step 89, the UAV stops paying the per-slot hover energy ($+106.42$), stops paying the per-slot $t_{\text{lack}}$ penalty ($+134.21$), and avoids the end-of-episode distance penalty ($+42.43$).
4. **Implications for Reinforcement Learning (Credit Assignment Failure):**
   - For an exploring policy during training, moving is dangerous: $87.5\%$ of 3D directions from $Q_{\text{START}}$ hit boundaries. If an agent flies 100m–200m into the field but fails to reach $Q_{\text{END}}$ before $N=200$, it pays moving energy, incurs acceleration penalties, and suffers the full $-c_{\text{nr}} d_{\text{re}}$ distance penalty, resulting in a reward **substantially WORSE than hovering**.
   - Hovering acts as an immediate local maximum with zero risk and guaranteed $+217.98$ reward, creating a severe credit assignment barrier against discovering the complete 89-step trajectory without persistent heuristic guidance.

### Channel Calibration Revision: FC_HZ / N0_DBM_HZ (`scripts/diagnose_channel_calibration.py`)
Investigated whether recalibrating our assumed wireless channel parameters (`FC_HZ`, `N0_DBM_HZ`) resolves the narrow margin between full-journey flight (TDPK) and stationary lingering (Always-Hover).

Executed a 12-combination parameter grid sweep over $f_c \in [2.0, 2.4, 5.0, 28.0]\text{ GHz}$ and $N_0 \in [-174, -169, -164]\text{ dBm/Hz}$ across 30 seeds ($k=10$):

#### 12-Combination Sweep Results:
| FC (GHz) | N0 (dBm/Hz) | Ref Rate @300m | TDPK r1 | TDPK Total | Hover r1 | Hover Total | Delta (TDPK − Hover) | TDPK/Hover Margin |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2.0 GHz | -174 | 12.86 Mbps | +510.0 | +345.8 | +641.1 | +218.0 | +127.9 | **1.59x** (original baseline) |
| 2.0 GHz | -169 | 9.54 Mbps | +361.7 | +197.6 | +308.9 | -114.2 | +311.8 | -1.73x (Hover net negative) |
| 2.0 GHz | -164 | 6.21 Mbps | +213.5 | +49.4 | -23.3 | -446.4 | +495.8 | -0.11x (Hover net negative) |
| **2.4 GHz** | **-174** | **11.81 Mbps** | **+463.0** | **+298.9** | **+535.9** | **+112.8** | **+186.1** | **2.65x (+165.0%) [ADOPTED]** |
| 2.4 GHz | -169 | 8.48 Mbps | +314.8 | +150.7 | +203.7 | -219.4 | +370.1 | -0.69x (Hover net negative) |
| 2.4 GHz | -164 | 5.16 Mbps | +166.6 | +2.5 | -128.5 | -551.6 | +554.1 | -0.00x (Hover net negative) |
| 5.0 GHz | -174 | 7.57 Mbps | +274.0 | +109.9 | +112.3 | -310.8 | +420.7 | -0.35x (Hover net negative) |
| 5.0 GHz | -169 | 4.25 Mbps | +125.8 | -38.3 | -219.9 | -643.0 | +604.7 | 0.06x |
| 5.0 GHz | -164 | 0.93 Mbps | -22.4 | -186.5 | -552.1 | -975.2 | +788.7 | 0.19x |
| 28.0 GHz | -174 | -2.37 Mbps | -169.5 | -333.6 | -881.9 | -1305.0 | +971.3 | 0.26x (SNR < 1, all rates negative) |
| 28.0 GHz | -169 | -5.69 Mbps | -317.7 | -481.9 | -1214.1 | -1637.2 | +1155.3 | 0.29x (all rates negative) |
| 28.0 GHz | -164 | -9.02 Mbps | -466.0 | -630.1 | -1546.3 | -1969.3 | +1339.3 | 0.32x (all rates negative) |

#### Selected Calibration & Engineering Justification:
Adopted **`FC_HZ = 2.4e9` (2.4 GHz)** and **`N0_DBM_HZ = -174.0 dBm/Hz`**:
1. **Physical Plausibility:** 2.4 GHz is the universal ISM band used for Wi-Fi / LTE-U UAV telemetry and communications links (IEEE 802.11b/g/n) in 20 MHz channels with 100 mW ($20\text{ dBm}$) transmit power. It replaces the arbitrary 2.0 GHz placeholder. $N_0 = -174\text{ dBm/Hz}$ is standard theoretical thermal noise floor ($k_B T_0$ at 290 K). Reference rate at 300m is a realistic **11.81 Mbps** ($\approx 1.18\text{ Mbps/user}$ for $K=10$).
2. **Substantially Strengthened Learning Margin:** The TDPK / Always-Hover total reward ratio jumps from **1.59x to 2.65x (+165.0% advantage)**, with the net advantage growing from $+127.9$ to **$+186.1$**.
3. **Preservation of Healthy Positive Value Landscape:** Always-Hover total reward is deflated by nearly half (from $+218.0$ down to $+112.8$), compressing the attractiveness of the stationary corner trap while avoiding negative reward inversions that destabilize early critic bootstrapping.

#### Downstream Synchronizations & End-to-End Re-Verification:
- **`config.py`:** Updated `FC_HZ = 2.4e9` with documented calibration comment.
- **`tests/test_channel_model.py`:** Recomputed path losses and rates under 2.4 GHz (`pl_los = 82.15`, `pl_nlos = 101.15`, `pl_avg = 82.20`, single-user `rate = 191303475.88`, `sum_rate = 198223227.00`).
- **`tests/test_mdp_environment.py`:** Updated `test_env_20_random_steps` cumulative reward to `11.3603`.
- **Unit Tests:** All 40 unit tests pass (`40 passed in 13.54s`).
- **Confirmed End-to-End Diagnostic Re-Run (`scripts/diagnose_reward_balance.py`):**
  - TDPK Total Reward: **$+298.89$**
  - Always Hover Total Reward: **$+112.77$**
  - Net Delta: **$+186.12$**
#### 800-Episode Local Diagnostic under Recalibrated Channel (`checkpoints/diag_channelfix/`):
Executed an 800-episode diagnostic run (`scripts/train.py --episodes 800 --seed 0 --checkpoint-dir checkpoints/diag_channelfix --checkpoint-every 200`) under the updated 2.4 GHz channel calibration.
Evaluated all checkpoints (`ep200`, `ep400`, `ep600`, `final`) across 20 seeds (0–19, $k=10$) with the deterministic actor (`scripts/diagnose_channelfix_eval.py`):

| Checkpoint | Mean Max Disp (m) | Median Disp (m) | Frac > 50m (%) | Arrival Rate (%) | Mean Reward |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `td3_agent_ep200.pt` | 1.0 m | 0.0 m | 0.0% | **0.0%** | +135.87 |
| `td3_agent_ep400.pt` | 51.1 m | 0.0 m | 10.0% | **0.0%** | +190.27 |
| `td3_agent_ep600.pt` | 18.5 m | 0.0 m | 5.0% | **0.0%** | +90.81 |
| `td3_agent_final.pt` (ep800) | 26.1 m | 0.0 m | 5.0% | **0.0%** | +125.11 |

##### Honest Read on Diagnostic Findings:
1. **Arrival Rate Did NOT Move Off Zero (STILL STRICTLY 0.0%):**
   - In 80 total deterministic evaluation rollouts (4 checkpoints $\times$ 20 seeds), exactly **0 episodes arrived at $Q_{\text{END}}$**.
   - Arrival rate did not budge from 0.0%, conclusively demonstrating that channel recalibration alone does not resolve the navigation consolidation bottleneck.
2. **Median Inaction Persists (0.0 m):**
   - Across all 4 checkpoints, the median maximum displacement remains exactly 0.0 m. Over 50% of evaluation seeds never leave $Q_{\text{START}}$ due to immediate boundary cancellations on step 1.
3. **Contrast with Training Behavior:**
   - During training, episodes with exploratory perturbation ($\sigma_3 = 0.1$) regularly hit arrival-scale returns ($+962$ to $+1352$). However, the deterministic policy gradient does not consolidate these exploratory trajectories into a reproducible policy.
4. **Conclusion:**
   - Channel recalibration successfully widened the theoretical and empirical TDPK-vs-Hover margin (2.65x), but the policy navigation/credit assignment deficit is NOT solved by channel parameter changes alone.
   - Do NOT start a full 6,000-episode Colab run yet. Further diagnosis of the actor gradient / action-space boundary behavior is required.

### Effective-Horizon (Gamma Ablation) & Replay-Buffer Imbalance Diagnostics (`checkpoints/diag_gamma099/`)
Investigated two targeted hypotheses for why the deterministic actor policy fails to consolidate goal arrival:
1. **Replay-Buffer Composition Imbalance:** Once heuristic guidance ends ($R_{\text{ex}} > R_{\text{rand}}$), do non-arrived timeout episodes drown out arrival transitions in the replay buffer?
2. **Effective Horizon Shortfall:** Does Table III's discount factor $\gamma = 0.96$ (effective horizon $1/(1 - \gamma) = 25$ steps) decay Bellman credit too aggressively across an 89-step journey ($\gamma^{89} \approx 0.026$)?

#### Step 1: Measured Replay Buffer Composition (Direct Measurement)
Instrumented `scripts/train.py` to record episode steps, arrival flags, and phase switches:
- **Total Network-Phase Episodes ($R_{\text{ex}} > 20,000$):** **685 episodes** (episodes 116–800)
- **Arrived Network-Phase Episodes:** **0 (0.0%)**
- **Mean Steps (Arrived Network Episodes):** **0.0**
- **Mean Steps (Non-Arrived Network Episodes):** **200.0 steps**
- **Total Transitions Added During Network Phase:** **137,000**
  - **From Arrived Episodes:** **0 (0.00%)**
  - **From Non-Arrived Episodes:** **137,000 (100.00%)**

> [!CRITICAL]
> **Severe Replay-Buffer Starvation Discovered:**
> Every single one of the 685 network-phase episodes timed out at 200 steps without arriving.
> Because `REPLAY_SIZE = 100,000`, the circular buffer completely overwrote and discarded all initial prior-knowledge arrival transitions by step 120,000 (~episode 600).
> **From episode 600 onwards, the replay buffer contained literally 0 arrival transitions.** The TD3 critic was computing Bellman updates exclusively on non-arrival data.

> [!NOTE]
> **Causal Record Correction:**
> Replay-buffer purging (evicting the initial ~20k transitions once cumulative steps exceed the 100k buffer capacity at ~ep600) cannot be the *original trigger* of the 0% arrival failure. Checkpoint `ep200.pt` ($R_{\text{ex}} \approx 37,000$) already exhibits a strict 0.0% arrival rate and 0.0m median displacement, long before any PK-phase data could have been overwritten.
> Replay buffer purging entrenches the failure and prevents mid-to-late recovery, but the initial breakdown occurs directly at the **PK-to-network handoff itself** (episodes 115–150).
> **Revised Priority:** Extend $R_{\text{rand}}$ (e.g. to 60,000, 3x the default, within the paper's Fig. 10 explored sensitivity regime) to test whether substantially more PK-phase seeding prevents the handoff collapse in the first place.

#### Step 2: Diagnostic Ablation — Higher Horizon ($\gamma = 0.99$)
Executed `scripts/train.py --episodes 800 --seed 0 --gamma 0.99 --checkpoint-dir checkpoints/diag_gamma099 --checkpoint-every 200` (CLI override wired to `TD3Agent`, keeping `config.GAMMA = 0.96` default intact).
Evaluated all checkpoints across 20 deterministic seeds (0–19, $k=10$) side-by-side with $\gamma = 0.96$:

| Checkpoint | Metric | $\gamma = 0.96$ (Table III Default) | $\gamma = 0.99$ (Horizon Ablation) | Effect of Longer Horizon |
| :---: | :---: | :---: | :---: | :--- |
| **`td3_agent_ep200.pt`** | Mean Max Disp<br>Median Disp<br>Frac > 50m<br>Arrival Rate<br>Mean Reward | 1.0 m<br>0.0 m<br>0.0%<br>**0.0%**<br>+135.87 | 17.0 m<br>0.0 m<br>5.0%<br>**0.0%**<br>+162.29 | Immediate mobility increase at ep200 |
| **`td3_agent_ep400.pt`** | Mean Max Disp<br>Median Disp<br>Frac > 50m<br>Arrival Rate<br>Mean Reward | 51.1 m<br>0.0 m<br>10.0%<br>**0.0%**<br>+190.27 | **140.0 m**<br>0.0 m<br>**25.0%**<br>**0.0%**<br>**+246.83** | **$2.7\times$ higher displacement, $2.5\times$ more escapes, $+30\%$ reward** |
| **`td3_agent_ep600.pt`** | Mean Max Disp<br>Median Disp<br>Frac > 50m<br>Arrival Rate<br>Mean Reward | 18.5 m<br>0.0 m<br>5.0%<br>**0.0%**<br>+90.81 | 75.1 m<br>0.0 m<br>20.0%<br>**0.0%**<br>+136.47 | $4\times$ higher displacement sustained |
| **`td3_agent_final.pt` (ep800)**| Mean Max Disp<br>Median Disp<br>Frac > 50m<br>Arrival Rate<br>Mean Reward | 26.1 m<br>0.0 m<br>5.0%<br>**0.0%**<br>+125.11 | 75.1 m<br>0.0 m<br>20.0%<br>**0.0%**<br>+148.44 | $3\times$ higher displacement sustained |

#### Step 3: Synthesis & Honest Engineering Read
1. **Did Arrival Rate Move Off Zero?**
   - **NO. Arrival rate remains strictly 0.0% across all checkpoints for both $\gamma = 0.96$ and $\gamma = 0.99$.**
2. **Strong Evidence Supporting the Horizon Hypothesis:**
   - Increasing $\gamma$ from 0.96 to 0.99 dramatically improved forward flight: peak mean displacement jumped from $51.1\text{ m}$ to **$140.0\text{ m}$**, and corner escape rate ($>50\text{ m}$) jumped from $10\%$ to **$25\%$**.
   - This proves that a longer effective horizon allows the actor to learn sustained flight deeper into the service volume.
3. **Why $\gamma = 0.99$ Still Failed to Achieve Arrivals:**
   - The buffer composition numbers explain why: because zero network-phase episodes arrived, the replay buffer suffered complete arrival starvation once circular overwrite purged the first 20,000 steps.
   - With 0% arrival transitions in the buffer, no value of $\gamma$ can propagate an arrival signal that does not exist in the training data.
4. **Current Recommendation:**
   - Do NOT start a full Colab run yet.
   - The root pathology is now isolated: (1) the hard cutoff at $R_{\text{rand}} = 20,000$ leaves the actor with zero ongoing arrival demonstrations, and (2) standard circular buffer overwrite purges all positive arrival transitions.

### R_RAND Extension Ablation: PK-to-Network Handoff Diagnostic (`checkpoints/diag_rrand60k/`)
Tested whether tripling the prior-knowledge seeding phase ($R_{\text{rand}} = 60,000$, $\sim 343$ PK episodes, 3x the paper's default of 20,000 and the upper sensitivity range explored in Fig. 10) prevents the policy collapse observed at the PK-to-network handoff.

#### Replay Buffer Composition Comparison (Network Phase Only):
| Metric | Run A ($R_{\text{rand}} = 20,000$, Default) | Run B ($R_{\text{rand}} = 60,000$, 3x Extension) |
| :--- | :---: | :---: |
| **PK-Phase Duration** | Episodes 1–115 (20,000 steps) | Episodes 1–343 (60,000 steps) |
| **Network-Phase Episodes** | 685 episodes (eps 116–800) | 457 episodes (eps 344–800) |
| **Arrived Network Episodes** | **0 (0.0%)** | **0 (0.0%)** |
| **Mean Steps (Arrived / Non-Arrived)** | 0.0 / 200.0 steps | 0.0 / 200.0 steps |
| **Total Transitions in Network Phase** | 137,000 transitions | 91,400 transitions |
| **Transitions from Arrived Episodes** | **0 (0.00%)** | **0 (0.00%)** |
| **Transitions from Non-Arrived Episodes** | **137,000 (100.00%)** | **91,400 (100.00%)** |

#### Deterministic Evaluation Table (20 Seeds, $k=10$, Run B: $R_{\text{rand}} = 60,000$):
| Checkpoint | Total Updates at Ckpt | Mean Max Disp (m) | Median Disp (m) | Frac > 50m (%) | **Arrival Rate (%)** | Mean Reward |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `td3_agent_ep200.pt` | 0 (Untrained random init) | 208.0 m | 0.0 m | 35.0% | **0.0%** | +225.03 |
| `td3_agent_ep400.pt` | 11,501 updates | 24.2 m | 0.0 m | 10.0% | **0.0%** | +129.51 |
| `td3_agent_ep600.pt` | 51,501 updates | 51.8 m | 0.0 m | 25.0% | **0.0%** | +162.98 |
| `td3_agent_final.pt` (ep800) | 91,501 updates | 106.9 m | 0.0 m | 35.0% | **0.0%** | +287.10 |

#### Honest Engineering Read: Was R_RAND the Missing Piece?
1. **Did Arrival Rate Move Off Zero?**
   - **NO. Arrival rate is STILL STRICTLY 0.0% across all checkpoints (0 arrivals out of 80 deterministic rollouts).**
   - Post-handoff arrival rate during training was also **0 out of 457 episodes (0.0%)**.
2. **Definitive Conclusion on Data Quantity:**
   - Tripling the prior-knowledge buffer seeding (60,000 transitions from 343 arrival trajectories) did **not** enable the actor to navigate to the goal upon handoff.
   - This proves conclusively that the handoff failure is **NOT a data-quantity problem**. Giving the critic 3x more arrival demonstrations does not prevent the actor network from collapsing into the corner once handed control.
3. **The Collapse Phenomenon at Handoff:**
   - At `td3_agent_ep200.pt` (0 updates, random initial weights), mean displacement was 208.0m.
   - Once TD3 updates began at episode 344, the actor collapsed at `ep400` down to 24.2m displacement and only 10% escapes. The critic updates actively drove the actor policy *into* the corner.
4. **Next Diagnostic Priority:**
   - Examine the **actor's specific action output and critic gradient $\nabla_a Q(s, a)$ at the exact corner state $(0,0,50)$ right at the handoff moment**. Why does the actor gradient push the actor into the boundary walls at $Q_{\text{START}}$ rather than reproducing the prior-knowledge velocity vector?

---
*This file is a living reference — update the Status column as modules are completed/reviewed, and log any new mismatches found during review under this "Review notes" section.*
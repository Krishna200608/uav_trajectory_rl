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
- **f_c** (carrier frequency) — appears in the free-space path-loss term LF = 20log(r) + 20log(f_c) + 20log(4π/v_c), eq. (10)-(11), but no number is stated in the text, Table II, or Table III.
- **N0** (noise power spectral density) — appears in the transmission-rate denominator, eq. (13). Not numerically specified.
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
| M9 | Training loop / full Algorithm 1 | M5, M6, M7, M8 | **Done — reviewed, approved (smoke test verified independently in a clean venv; network-driven branch confirmed via total_updates>0 after loading the saved checkpoint)** |
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

---
*This file is a living reference — update the Status column as modules are completed/reviewed, and log any new mismatches found during review under this "Review notes" section.*
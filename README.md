# Grover Solver for Deployment Configuration

Grover search (Qiskit + Aer) over three deployment features:

| feature | meaning    | qubit |
|---------|------------|-------|
| `P`     | public API | `q0`  |
| `A`     | auth       | `q1`  |
| `W`     | worker     | `q2`  |

Requirements: **R1** = `P or W`, **R2** = `if P then A`, **R3** = `not (P and W)`.

## Bit order (read this first)

**q0=P, q1=A, q2=W; Qiskit prints W A P.** Qiskit is little-endian: qubit 0 is the
*right-most* character of a counts key, so key `"011"` means W=0, A=1, P=1, which is `{P:1,A:1,W:0}`.
Basis-state index is `i = P + 2A + 4W`. All decoding goes through
`src/classical.py::bitstring_to_config`, and every result is printed with feature names.

| config          | P-A-W reading | Qiskit key (W-A-P) |
|-----------------|---------------|--------------------|
| `{P:1,A:1,W:0}` | `110`         | `011`              |
| `{P:0,A:0,W:1}` | `001`         | `100`              |
| `{P:0,A:1,W:1}` | `011`         | `110`              |

K = 3 of N = 8.

## Setup

```bash
cd grover_deployment_solver
python3 -m venv .venv            # Debian w/o python3-venv: python3 -m venv --without-pip .venv + get-pip.py
.venv/bin/pip install -r requirements.txt
```
Tested with Python 3.13, qiskit 2.6.2, qiskit-aer 0.18.2, numpy 2.5.4, matplotlib 3.11.2.

## Run

```bash
.venv/bin/python run_all.py      # everything, ~20 s, fixed seed 1234, 8192 shots
.venv/bin/python -m pytest -q    # 62 tests
```
Outputs go to `outputs/`. `results.txt` holds the full log, `01..07_*.txt` hold the individual sections, and the `*.png` files are the figures.

## Layout

| file | content |
|------|---------|
| `src/classical.py` | checker R1–R3, exhaustive enumeration, truth table, bit-order helpers |
| `src/manual_oracle.py` | oracle V1 / V2 / BROKEN from X, CX, CCX only; per-basis-state verification; gate counts |
| `src/grover.py` | H prep, diffuser (H X CCZ X H), iteration rule, AerSimulator runs, classical re-check |
| `src/policy.py` | safe `ast` policy parser, `PhaseOracleGate` oracle with diagonal check, K=0 handling |
| `src/theory.py` | amplitude algebra, sin²((2t+1)θ), Wilson CIs and 3σ flags, ablation analysis, cost text |
| `src/linalg.py` | Grover as explicit linear algebra: column vectors, gate matrices, O, D = 2\|s⟩⟨s\| − I, G = D·O, 2×2 rotation, cross-checks vs Qiskit |
| `src/plots.py` | amplitude bars, success curves, rotation geometry, histograms, ablation, gate-matrix and G = D·O figures |

## Linear-algebra view (`src/linalg.py`)

Everything is also written as plain NumPy matrices and checked against the Qiskit circuits
(`outputs/08_linear_algebra.txt`, `outputs/gate_matrices.png`, `outputs/linear_algebra.png`):

* **Variables are column vectors:** |0⟩ = [1,0]ᵀ, |1⟩ = [0,1]ᵀ, and |W A P⟩ = |W⟩ ⊗ |A⟩ ⊗ |P⟩ is 8×1, with a single 1 at row
  P + 2A + 4W.
* **Gates are matrices** (X, H, Z, CX 4×4, CCX 8×8, CCZ), and **a circuit is their product**, M = U_last ⋯ U_1.
  The kick qubit |−⟩ = HX|0⟩ is an eigenvector of X with eigenvalue −1.
* **The oracle** is O = diag(1,1,1,−1,−1,1,−1,1). The product of V1's 31 gate matrices (256×256) and of V2's 15 (32×32)
  both reduce to O on the helpers-|0⟩ block, and the V1 product equals Qiskit's `Operator`.
* **The diffuser** is D = 2|s⟩⟨s| − I (−0.75 on the diagonal, 0.25 elsewhere). The gate product H X CCZ X H gives −D, which differs only by a global phase.
* **One iteration** is G = D·O, and G|s⟩ has valid entries 0.530 and invalid entries −0.177, so P = 0.84375. In the {|U⟩,|M⟩} basis G is
  the rotation matrix [[cos2θ, −sin2θ],[sin2θ, cos2θ]] with eigenvalues e^(±i2θ).
* Gᵗ|s⟩ matches sin²((2t+1)θ) and Qiskit's statevector for t = 0…4 and K = 3, 2, 1 (17/17 checks pass).
* Without uncomputation, the oracle's helpers-|0⟩ block is the zero matrix.

## Oracles (Task A.2)

* **V1** (8 qubits): helpers `r1 = P|W` (De Morgan), `r2 = ~P|A`, `r3 = ~(P&W)`, an AND helper
  `r1&r2`, and a kick qubit prepared as in the lecture, `|0> -X-> |1> -H-> |->`.
  `CCX(and, r3, kick)` kicks the phase back, then every compute gate is replayed in reverse.
* **V2** (5 qubits): `CX(P→W)` turns W into `P xor W`, which equals `R1 & R3`. One helper holds R2, and a
  single CCX does the kick before everything is uncomputed.
* Verification checks all 8 basis inputs with an exact statevector for three things: (a) phase −1 exactly on valid inputs,
  (b) input bits unchanged, (c) every helper and the kick qubit back in |0>.

| oracle | qubits | X | CX | CCX | H | total | CX after decomposition |
|--------|-------:|--:|---:|----:|--:|------:|-----------------------:|
| V1     | 8 | 20 | 0 | 9 | 2 | 31 | 54 |
| V2     | 5 |  8 | 2 | 3 | 2 | 15 | 20 |

## Policies (Task B.4)

A policy uses `& | ^ ~ ( )` over `P, A, W`. It is parsed with `ast` against a node whitelist and never passed to `eval`.
Errors are specific: `use '&' / '|' instead of 'and' / 'or'`, `unknown variable 'X'`, and so on.
The policy is ANDed with R1–R3, and **K is counted classically first**. When K = 0 the solver reports
"no configuration exists" and offers the exhaustive 8-row truth table as proof. Grover is not run in that case.

**`PhaseOracleGate` behaviour in Qiskit 2.6.** I checked this against the installed version and found two things.
First, the gate only gets qubits for variables that *appear* in the expression. Second, gate qubit k is
`gate.boolean_expression.args[k]`, which follows the *order of first appearance* unless `var_order` is given.
Without explicit wiring, `"W & ~P"` would put W on q0.
`build_phase_oracle` passes `var_order=['P','A','W']` and wires each gate qubit to its feature's qubit.
It then compares the full 8-entry diagonal with the classical truth table. That check is a test, `test_phase_oracle_gate_appearance_order_trap`.

| policy   | K | t | measured success | answer |
|----------|---|---|------------------|--------|
| `~W`     | 1 | 2 | 0.942 | `{P:1,A:1,W:0}` ✓ |
| `A`      | 2 | 1 | 1.000 | `{P:1,A:1,W:0}` ✓ (or `{P:0,A:1,W:1}`) |
| `P & ~A` | 0 | n/a | n/a | no configuration exists (truth-table proof) |

## Iteration strategy

K is counted classically, then

```
t = max(1, round(pi/(4*theta) - 1/2)),   theta = asin(sqrt(K/N))
```
→ K=3: t=1 (P=0.844), K=2: t=1 (P=1.000), K=1: t=2 (P=0.945).

**Limitation:** this solver *relies on the classical count*. If K were unknown you would need the
randomized exponential-search schedule (Boyer–Brassard–Høyer–Tapp) or quantum counting. At N=8,
counting K classically already solves the whole problem.

## Theory vs measurement

Amplitude walk-through for K=3 (scaled by √8; normalised in brackets):
initial 1 (0.354) → phase-marked valid −1, invalid +1 → mean μ = 0.25 (0.088) →
after diffuser valid **1.5 (0.530)**, invalid −0.5 (−0.177) → success 3·0.530² = **0.84375**.

I measured t = 0..4 at 8192 shots for K = 3, 2, 1 (15 points). All 15 agree with sin²((2t+1)θ), and none deviates by more than 3σ.
The 95% Wilson intervals are in `outputs/05_theory.txt`.

**About the "overshoot" for K=3.** The first overshoot happens at t=2 (P=0.023). At t=3 the state has
rotated about 264°, and the success is actually *higher* (0.990) than at t=1. t=1 is still preferred for three reasons:
* it is the first approach to |M>;
* the expected oracle calls per success are 1/0.844 = 1.18 for t=1 versus 3/0.990 = 3.03 for t=3;
* a 3× deeper circuit is noisier on real hardware and more sensitive to errors in θ.

## Uncomputation ablation

The BROKEN oracle skips the uncompute step but keeps the kick, so its phases are still correct but its helpers stay dirty.
That leaves the workspace entangled with (P,A,W): the search-register purity falls from 1.00 to 0.25.
The diffuser can then only interfere states that share the same garbage pattern.
Success at t=1: predicted 0.844, correct oracle 0.845, **broken oracle 0.183** (exact value 0.1875).
That is worse than measuring with no Grover iteration at all (0.375). See `outputs/ablation.png`.

## Cost: is there an advantage?

At N=8, **no**. Classical enumeration costs 8 checker calls at O(L) each, and it returns K, every solution, and a
proof of infeasibility. Grover at t=1 needs one oracle call, but the full circuit decomposes to 60 CX + 129 single-qubit gates.
On top of that come shots, a classical re-check, and the K we had to count classically anyway.
Grover's ~(π/4)·√(N/K) queries against ~N/K classical checks is a quadratic gain that only matters for large N.
The simulator also stores 2^q amplitudes (256 for V1), which is exponentially more work than enumerating 8 cases.

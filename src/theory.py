"""Amplitude algebra, success predictions, and prediction-vs-measurement.

With N = 8 states, K of them marked, uniform start amplitude a0 = 1/sqrt(N):

  after oracle    valid  -> -a0          invalid -> +a0
  mean            mu = ((N-K)a0 - K a0) / N = (N-2K)/N * a0
  after diffuser  a  -> 2 mu - a   (inversion about the mean)
                  valid   = 2mu + a0     invalid = 2mu - a0
  success after t iterations   P(t) = sin^2((2t+1) theta),  theta = asin(sqrt(K/N))
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from qiskit.quantum_info import Statevector

from .classical import N_STATES, is_valid
from .grover import SEED, SHOTS, choose_iterations, grover_circuit, run_counts, success_fraction, theta_for

T_RANGE = range(0, 5)


# --------------------------------------------------------- amplitude algebra
@dataclass
class AmplitudeStep:
    K: int
    N: int
    a0: float
    marked_valid: float
    marked_invalid: float
    mean: float
    after_valid: float
    after_invalid: float

    @property
    def success_one_iter(self) -> float:
        return self.K * self.after_valid ** 2


def amplitude_step(K: int, N: int = N_STATES) -> AmplitudeStep:
    a0 = 1 / math.sqrt(N)
    mu = (N - 2 * K) / N * a0
    return AmplitudeStep(K, N, a0, -a0, a0, mu, 2 * mu + a0, 2 * mu - a0)


def amplitude_table(K: int, N: int = N_STATES) -> str:
    s = amplitude_step(K, N)
    sc = math.sqrt(N)                           # scale factor: scaled = normalized * sqrt(N)
    rows = [("initial", s.a0, s.a0),
            ("after phase mark (oracle)", s.marked_valid, s.marked_invalid),
            ("mean amplitude mu", s.mean, s.mean),
            ("after diffuser (2mu - a)", s.after_valid, s.after_invalid)]
    head = f"{'step':<28} | {'valid scaled':>12} | {'invalid scaled':>14} | {'valid norm':>10} | {'invalid norm':>12}"
    out = [f"Amplitude walk-through for K={K}, N={N}  (scaled = amplitude x sqrt(N), so the start is 1)",
           head, "-" * len(head)]
    for name, v, i in rows:
        out.append(f"{name:<28} | {v*sc:>12.4f} | {i*sc:>14.4f} | {v:>10.4f} | {i:>12.4f}")
    out.append(f"check normalisation after diffuser: K*v^2 + (N-K)*u^2 = "
               f"{K*s.after_valid**2 + (N-K)*s.after_invalid**2:.6f}")
    out.append(f"predicted success after 1 iteration: K * valid^2 = {s.success_one_iter:.6f}"
               f"  (= sin^2(3 theta) = {predicted_success(K, 1):.6f})")
    return "\n".join(out)


def predicted_success(K: int, t: int, N: int = N_STATES) -> float:
    return math.sin((2 * t + 1) * theta_for(K, N)) ** 2


def prediction_table(K: int, N: int = N_STATES) -> str:
    th = theta_for(K, N)
    out = [f"K={K}: theta = asin(sqrt({K}/{N})) = {math.degrees(th):.3f} deg;  "
           f"pi/(4 theta) - 1/2 = {math.pi/(4*th) - 0.5:.3f}  ->  t* = {choose_iterations(K, N)}"]
    for t in T_RANGE:
        out.append(f"   t={t}: angle (2t+1)theta = {math.degrees((2*t+1)*th):7.2f} deg   "
                   f"P = sin^2 = {predicted_success(K, t, N):.4f}")
    return "\n".join(out)


# ------------------------------------------------------------- statistics
def wilson_ci(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


@dataclass
class Measurement:
    K: int
    t: int
    shots: int
    hits: int
    predicted: float

    @property
    def p_hat(self):
        return self.hits / self.shots

    @property
    def ci(self):
        return wilson_ci(self.hits, self.shots)

    @property
    def sigma(self):
        # binomial s.d. under the prediction; floor avoids /0 when P is exactly 0 or 1
        return math.sqrt(max(self.predicted * (1 - self.predicted), 1.0 / self.shots) / self.shots)

    @property
    def z(self):
        return (self.p_hat - self.predicted) / self.sigma

    @property
    def flagged(self):
        return abs(self.z) > 3


def measure_sweep(oracle, K: int, pred: Callable = is_valid, shots: int = SHOTS, seed: int = SEED) -> List[Measurement]:
    out = []
    for t in T_RANGE:
        counts = run_counts(grover_circuit(oracle, t), shots=shots, seed=seed + 100 * K + t)
        hits = round(success_fraction(counts, pred) * shots)
        out.append(Measurement(K, t, shots, hits, predicted_success(K, t)))
    return out


def sweep_table(ms: List[Measurement], label: str) -> str:
    head = f"{'t':>2} | {'predicted':>9} | {'measured':>8} | {'95% CI (Wilson)':^17} | {'z':>6} | flag"
    out = [f"Prediction vs measurement: {label}", head, "-" * len(head)]
    for m in ms:
        lo, hi = m.ci
        out.append(f"{m.t:>2} | {m.predicted:9.4f} | {m.p_hat:8.4f} | [{lo:.4f}, {hi:.4f}] | {m.z:+6.2f} | "
                   f"{'>3 sigma!' if m.flagged else 'ok'}")
    out.append(f"deviations beyond 3 sigma: {sum(m.flagged for m in ms)}")
    return "\n".join(out)


# --------------------------------------------------------------- ablation
def exact_success(oracle, t: int, pred: Callable = is_valid) -> float:
    """Exact P(valid) on the search register from the full statevector."""
    sv = Statevector(grover_circuit(oracle, t, measure=False))
    probs = sv.probabilities_dict(qargs=[0, 1, 2])
    from .classical import bitstring_to_config
    return sum(p for k, p in probs.items() if pred(bitstring_to_config(k)))


def search_register_purity(oracle) -> float:
    """Tr(rho^2) of the 3 search qubits right after one oracle call on |s>."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import partial_trace
    qc = QuantumCircuit(oracle.num_qubits)
    qc.h(range(3))
    qc.compose(oracle, inplace=True)
    sv = Statevector(qc)
    rest = list(range(3, oracle.num_qubits))
    rho = partial_trace(sv, rest) if rest else sv.to_operator()
    return float(np.real(rho.purity()))


# -------------------------------------------------------------- narrative
OVERSHOOT_NOTE = """\
Overshoot / periodicity for K=3 (theta = 37.76 deg):
  t=1 -> (3 theta)  = 113.3 deg, P = 0.844   (just past 90 deg: a slight overshoot already)
  t=2 -> (5 theta)  = 188.8 deg, P = 0.023   (rotated far PAST |M>: overshoot, near-total failure)
  t=3 -> (7 theta)  = 264.3 deg, P = 0.990   (the state has swung round to the -|M> side;
                                               measuring only sees sin^2, so it looks great)
  t=4 -> (9 theta)  = 339.8 deg, P = 0.119
Honest reading: t=3 really does give a HIGHER single-shot success (0.990) than t=1 (0.844).
t=1 is still the preferred choice because
  * it is what the formula t = round(pi/(4 theta) - 1/2) gives: the FIRST time the state comes
    close to |M>; t=3 is the second lap of the same rotation;
  * cost per verified answer: expected oracle calls = t / P(t) -> t=1: 1/0.844 = 1.18,
    t=3: 3/0.990 = 3.03 (2.6x more), and every answer is re-checked classically anyway,
    so a rare miss just costs one more cheap shot;
  * 3x deeper circuits mean 3x more noise on real hardware, and the 'lucky' second lap
    depends on theta being known exactly: a small error in K or theta is amplified by (2t+1).
"""


def preferred_t_summary() -> str:
    out = ["Preferred iteration count  t = max(1, round(pi/(4 theta) - 1/2)),  theta = asin(sqrt(K/N))"]
    for K in (3, 2, 1):
        t = choose_iterations(K)
        out.append(f"  K={K}: theta={math.degrees(theta_for(K)):6.2f} deg  t={t}  P(t)={predicted_success(K, t):.4f}")
    return "\n".join(out)


def cost_report(oracle_gate_counts: dict, grover_counts: dict) -> str:
    L = 3
    rows = []
    for n in (3, 10, 20, 30, 40):
        N = 2 ** n
        rows.append(f"  n={n:>2}  N={N:>16,}  classical exhaustive = {N:>16,} checks   "
                    f"Grover (K=1) ~ (pi/4)sqrt(N) = {math.pi/4*math.sqrt(N):>12,.0f} oracle calls")
    return f"""COST DISCUSSION (honest)
Classical: exhaustive enumeration = N = 8 checker calls, each O(L) with L = {L} rules
  (~ 7 Boolean literal operations) -> ~ 8 x O(L) elementary operations, and it gives K, every
  solution, AND a proof when K = 0.
Grover at t=1 (K=3): 1 oracle call + circuit overhead:
  oracle V1 alone : {oracle_gate_counts}
  full t=1 circuit: {grover_counts}
  plus state prep, measurement, repeated shots, and a classical re-check of the answer.
  And the solver needed K first -- which we obtained by the classical enumeration itself.
=> At N = 8 there is NO practical advantage: one circuit is already more elementary gates
   than the whole classical search.
Scaling: Grover needs ~ (pi/4) sqrt(N/K) oracle calls versus ~N/K classical random checks
  (N for exhaustive). Quadratic, not exponential, speed-up; it only pays off when N is huge and
  the oracle is cheap relative to the classical check:
{chr(10).join(rows)}
Simulator caveat: AerSimulator stores 2^q complex amplitudes for q qubits (here q = 8 for
  oracle V1 -> 256 amplitudes; q = 5 for V2 -> 32). Simulating Grover classically costs
  O(2^q) memory and time per gate -- exponentially WORSE than just enumerating the 2^3
  configurations. Any speed-up claim refers to real quantum hardware, not this simulation.
Limitation: this solver uses the classical count K to pick t. With K unknown one would use
  the randomized exponential schedule of Boyer-Brassard-Hoyer-Tapp (pick t uniformly in
  [0, m), grow m by 6/5 on failure; expected O(sqrt(N/K)) calls) or quantum counting
  (phase estimation on the Grover operator) to estimate K first.
"""

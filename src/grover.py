"""Grover search over the 3 feature qubits (q0=P, q1=A, q2=W).

Any oracle is accepted as a QuantumCircuit whose first 3 qubits are the search
register; extra qubits are ancillas that start (and must end) in |0>.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, Optional

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .classical import (
    N_STATES, bitstring_to_config, config_to_index, fmt_config, index_to_config, is_valid,
)

SEED = 1234
SHOTS = 8192
N_SEARCH = 3


# ------------------------------------------------------------ iterations
def theta_for(K: int, N: int = N_STATES) -> float:
    return math.asin(math.sqrt(K / N))


def choose_iterations(K: int, N: int = N_STATES) -> int:
    """t = max(1, round(pi/(4*theta) - 1/2)),  theta = asin(sqrt(K/N)).

    Requires the classical solution count K (0 < K < N).  If K were unknown
    we would need either the randomized exponential-search schedule
    (Boyer-Brassard-Hoyer-Tapp) or quantum counting first.
    """
    if not 0 < K < N:
        raise ValueError(f"need 0 < K < N, got K={K}, N={N}")
    return max(1, round(math.pi / (4 * theta_for(K, N)) - 0.5))


# -------------------------------------------------------------- circuits
def diffuser(n: int = N_SEARCH) -> QuantumCircuit:
    """H X CCZ X H on the 3 search qubits (CCZ = H . CCX . H on the target).

    Implements -(2|s><s| - I) ; the global sign is unobservable.
    """
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.ccx(0, 1, n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover_circuit(oracle: QuantumCircuit, t: int, measure: bool = True) -> QuantumCircuit:
    n = oracle.num_qubits
    qc = QuantumCircuit(n, name=f"Grover t={t}")
    qc.h(range(N_SEARCH))                               # |s> on the search register
    D = diffuser()
    for _ in range(t):
        qc.compose(oracle, qubits=range(n), inplace=True)
        qc.compose(D, qubits=range(N_SEARCH), inplace=True)
    if measure:
        cr = ClassicalRegister(N_SEARCH, "c")
        qc.add_register(cr)
        qc.measure(range(N_SEARCH), cr)                 # c0=P, c1=A, c2=W
    return qc


_SIM_CACHE: Dict[int, AerSimulator] = {}


def run_counts(qc: QuantumCircuit, shots: int = SHOTS, seed: int = SEED) -> Dict[str, int]:
    sim = AerSimulator(seed_simulator=seed)
    tqc = transpile(qc, sim, seed_transpiler=seed, optimization_level=1)
    return dict(sim.run(tqc, shots=shots).result().get_counts())


def success_fraction(counts: Dict[str, int], pred=is_valid) -> float:
    tot = sum(counts.values())
    return sum(v for k, v in counts.items() if pred(bitstring_to_config(k))) / tot


def decode_counts(counts: Dict[str, int]) -> Counter:
    """Qiskit keys ('W A P') -> integer basis index (P=bit0)."""
    out = Counter()
    for k, v in counts.items():
        out[config_to_index(bitstring_to_config(k))] += v
    return out


def counts_table(counts: Dict[str, int], pred=is_valid) -> str:
    tot = sum(counts.values())
    lines = [f"{'Qiskit key(WAP)':^15} | {'config':^15} | {'count':>6} | {'freq':>6} | valid?"]
    for k in sorted(counts, key=lambda s: -counts[s]):
        c = bitstring_to_config(k)
        lines.append(f"{k:^15} | {fmt_config(c):^15} | {counts[k]:>6} | {counts[k]/tot:6.3f} | "
                     f"{'YES' if pred(c) else 'no'}")
    return "\n".join(lines)


def solve(oracle: QuantumCircuit, K: int, pred=is_valid, shots: int = SHOTS, seed: int = SEED,
          t: Optional[int] = None):
    """Run Grover, take the most frequent outcome, RE-CHECK it classically."""
    t = choose_iterations(K) if t is None else t
    counts = run_counts(grover_circuit(oracle, t), shots=shots, seed=seed)
    best = max(counts, key=counts.get)
    cfg = bitstring_to_config(best)
    return {
        "t": t, "counts": counts, "best_key": best, "best_config": cfg,
        "verified": bool(pred(cfg)), "success": success_fraction(counts, pred),
    }


def report(oracle: QuantumCircuit, K: int, label: str) -> str:
    r = solve(oracle, K)
    lines = [
        f"Grover with {label}: K={K}, N={N_STATES}, theta={math.degrees(theta_for(K)):.2f} deg, "
        f"t={r['t']}, shots={SHOTS}, seed={SEED}",
        counts_table(r["counts"]),
        f"measured success fraction = {r['success']:.4f}",
        f"most frequent key '{r['best_key']}' (W A P) -> {fmt_config(r['best_config'])}",
        f"classical re-check of that configuration: {'VALID' if r['verified'] else 'INVALID'}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from .manual_oracle import build_oracle_v1
    from .classical import count_solutions
    print(report(build_oracle_v1(), count_solutions(), "manual oracle V1"))

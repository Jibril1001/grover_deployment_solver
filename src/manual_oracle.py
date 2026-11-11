"""Hand-built phase oracles using ONLY X, CX, CCX  (+ H to prepare |-> on the
kick-back qubit, exactly as in the lecture:  |0> --X--> |1> --H--> |->).

No oracle generators are used here.

Qubit layout
------------
Every oracle has the three search qubits first:  q0=P, q1=A, q2=W.
Ancillas follow.  They start in |0> and (for a correct oracle) end in |0>.

Version 1  (one helper per requirement, 8 qubits)
    q3 r1  = P | W            (De Morgan: ~(~P & ~W))
    q4 r2  = ~P | A           (~(P & ~A))
    q5 r3  = ~(P & W)
    q6 and = r1 & r2          (AND helper)
    q7 kick in |->            CCX(and, r3, kick) kicks phase (-1)^(r1&r2&r3)
    ... then every compute gate is applied again in REVERSE order (uncompute).

Version 2  (minimal, 5 qubits)  uses  R1 & R3 == P xor W:
    CX(P -> W)         W' = P ^ W            (in place, reversible)
    h = P & ~A         (= NOT R2)
    phase on W' & ~h   -> (P^W) & R2 = R1 & R2 & R3
    uncompute h, then CX(P -> W) restores W.

Broken  (ablation): Version 1 WITHOUT the uncompute block.  The phase is still
right, but the helpers keep a copy of information about (P,A,W).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector

from .classical import (
    FEATURES, N_STATES, fmt_config, index_to_bitstring, index_to_config, is_valid,
)

Gate = Tuple[str, Tuple[int, ...]]   # ("x", (q,)) / ("cx", (c,t)) / ("ccx", (c1,c2,t))


def _apply(qc: QuantumCircuit, ops: List[Gate]) -> None:
    for name, qs in ops:
        getattr(qc, name)(*qs)       # only x / cx / ccx ever appear in ops


def _search_regs() -> List[QuantumRegister]:
    return [QuantumRegister(1, n) for n in FEATURES]


# ------------------------------------------------------------- version 1
P, A, W = 0, 1, 2
V1_R1, V1_R2, V1_R3, V1_AND, V1_KICK = 3, 4, 5, 6, 7

V1_COMPUTE: List[Gate] = [
    # r1 = P | W  = NOT(~P & ~W)
    ("x", (P,)), ("x", (W,)), ("ccx", (P, W, V1_R1)), ("x", (P,)), ("x", (W,)), ("x", (V1_R1,)),
    # r2 = ~P | A = NOT(P & ~A)
    ("x", (A,)), ("ccx", (P, A, V1_R2)), ("x", (A,)), ("x", (V1_R2,)),
    # r3 = NOT(P & W)
    ("ccx", (P, W, V1_R3)), ("x", (V1_R3,)),
    # AND helper = r1 & r2
    ("ccx", (V1_R1, V1_R2, V1_AND)),
]


def _kick_prep(qc, k):          # lecture recipe |0> -X-> |1> -H-> |->
    qc.x(k)
    qc.h(k)


def _kick_unprep(qc, k):        # |-> -H-> |1> -X-> |0>
    qc.h(k)
    qc.x(k)


def build_oracle_v1(uncompute: bool = True) -> QuantumCircuit:
    regs = _search_regs() + [QuantumRegister(1, n) for n in ("r1", "r2", "r3", "and", "kick")]
    qc = QuantumCircuit(*regs, name="Oracle V1" if uncompute else "Oracle BROKEN")
    _kick_prep(qc, V1_KICK)
    qc.barrier(label="compute")
    _apply(qc, V1_COMPUTE)
    qc.barrier(label="kick")
    qc.ccx(V1_AND, V1_R3, V1_KICK)            # phase (-1)^(r1 & r2 & r3)
    if uncompute:
        qc.barrier(label="uncompute")
        _apply(qc, list(reversed(V1_COMPUTE)))  # explicit reverse uncomputation
    else:
        qc.barrier(label="NO uncompute")
    _kick_unprep(qc, V1_KICK)
    return qc


def build_oracle_broken() -> QuantumCircuit:
    return build_oracle_v1(uncompute=False)


# ------------------------------------------------------------- version 2
V2_H, V2_KICK = 3, 4
V2_COMPUTE: List[Gate] = [
    ("cx", (P, W)),                                   # W' = P ^ W   (= R1 & R3)
    ("x", (A,)), ("ccx", (P, A, V2_H)), ("x", (A,)),  # h = P & ~A   (= NOT R2)
    ("x", (V2_H,)),                                   # h = R2
]


def build_oracle_v2() -> QuantumCircuit:
    regs = _search_regs() + [QuantumRegister(1, "r2"), QuantumRegister(1, "kick")]
    qc = QuantumCircuit(*regs, name="Oracle V2")
    _kick_prep(qc, V2_KICK)
    qc.barrier(label="compute")
    _apply(qc, V2_COMPUTE)
    qc.barrier(label="kick")
    qc.ccx(W, V2_H, V2_KICK)                          # phase (-1)^((P^W) & R2)
    qc.barrier(label="uncompute")
    _apply(qc, list(reversed(V2_COMPUTE)))
    _kick_unprep(qc, V2_KICK)
    return qc


# ----------------------------------------------------------- verification
@dataclass
class CheckRow:
    index: int
    config: dict
    expected_valid: bool
    phase: float
    input_preserved: bool
    helpers_clean: bool

    @property
    def phase_ok(self) -> bool:
        return np.isclose(self.phase, -1.0 if self.expected_valid else 1.0)

    @property
    def passed(self) -> bool:
        return self.phase_ok and self.input_preserved and self.helpers_clean


def verify_oracle(oracle: QuantumCircuit, n_search: int = 3, pred=is_valid) -> List[CheckRow]:
    """Run the oracle on every basis input |x>|0..0> with an exact statevector.

    Checks (a) phase is -1 exactly on valid x and +1 otherwise,
           (b) the search register still holds x,
           (c) every ancilla (helpers + kick) is back in |0>.
    """
    rows = []
    n = oracle.num_qubits
    for x in range(2 ** n_search):
        qc = QuantumCircuit(n)
        for k in range(n_search):
            if (x >> k) & 1:
                qc.x(k)
        qc.compose(oracle, qubits=range(n), inplace=True)
        amps = Statevector(qc).data
        out = int(np.argmax(np.abs(amps)))
        pure_basis = np.isclose(abs(amps[out]), 1.0)
        rows.append(CheckRow(
            index=x,
            config=index_to_config(x),
            expected_valid=pred(index_to_config(x)),
            phase=float(np.real(amps[out])) if pure_basis else float("nan"),
            input_preserved=pure_basis and (out & (2 ** n_search - 1)) == x,
            helpers_clean=pure_basis and (out >> n_search) == 0,
        ))
    return rows


def check_table(name: str, rows: List[CheckRow]) -> str:
    head = f"{'Qiskit(WAP)':^11} | {'config':^15} | valid | phase | input kept | helpers |0> | result"
    lines = [f"--- {name} ---", head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{index_to_bitstring(r.index):^11} | {fmt_config(r.config):^15} | {'YES' if r.expected_valid else ' no':^5} | "
            f"{r.phase:+5.1f} | {'yes' if r.input_preserved else 'NO':^10} | {'yes' if r.helpers_clean else 'NO':^11} | "
            f"{'PASS' if r.passed else 'FAIL'}"
        )
    lines.append(f"overall: {'PASS' if all(r.passed for r in rows) else 'FAIL'} "
                 f"({sum(r.passed for r in rows)}/{len(rows)} rows)")
    return "\n".join(lines)


def resource_counts(qc: QuantumCircuit) -> dict:
    ops = {k: v for k, v in qc.count_ops().items() if k != "barrier"}
    flat = qc.copy()
    flat.data = [d for d in flat.data if d.operation.name != "barrier"]
    t = transpile(flat, basis_gates=["cx", "u"], optimization_level=0)
    return {
        "qubits": qc.num_qubits,
        "ancillas": qc.num_qubits - 3,
        "x": ops.get("x", 0), "cx": ops.get("cx", 0), "ccx": ops.get("ccx", 0), "h": ops.get("h", 0),
        "total_gates": sum(ops.values()),
        "depth": flat.depth(),
        "cx_after_decomp": t.count_ops().get("cx", 0),
    }


def compare_table(circs: dict) -> str:
    keys = ["qubits", "ancillas", "x", "cx", "ccx", "h", "total_gates", "depth", "cx_after_decomp"]
    head = f"{'oracle':<14}" + "".join(f"{k:>16}" for k in keys)
    lines = [head, "-" * len(head)]
    for name, qc in circs.items():
        rc = resource_counts(qc)
        lines.append(f"{name:<14}" + "".join(f"{rc[k]:>16}" for k in keys))
    lines.append("(h = the 2 H gates that prepare/unprepare the |-> kick qubit; cx_after_decomp = CX count "
                 "after decomposing every CCX into {CX,U}, 6 CX per CCX)")
    return "\n".join(lines)


if __name__ == "__main__":
    for nm, b in (("V1", build_oracle_v1), ("V2", build_oracle_v2), ("BROKEN", build_oracle_broken)):
        print(check_table(nm, verify_oracle(b())))

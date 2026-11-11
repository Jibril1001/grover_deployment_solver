"""Classical reference model for the deployment-configuration problem.

Features
--------
P = public API, A = auth, W = worker.

Requirements
------------
R1 = P or W
R2 = if P then A          (== ~P | A)
R3 = not (P and W)

Qubit / bit order (IMPORTANT)
-----------------------------
q0 = P, q1 = A, q2 = W.

Qiskit is little-endian: qubit 0 is the RIGHT-most character of a printed
bitstring.  A counts key such as ``"011"`` is therefore read  W A P:
W=0, A=1, P=1  ->  {P:1,A:1,W:0}.

The integer basis-state index is ``i = P*1 + A*2 + W*4``.  Every function in
this project goes through :func:`index_to_config` / :func:`bitstring_to_config`
so that this convention lives in exactly one place.
"""
from __future__ import annotations

from typing import Callable, Dict, List

FEATURES = ("P", "A", "W")          # FEATURES[k] lives on qubit k
N_QUBITS = len(FEATURES)
N_STATES = 2 ** N_QUBITS            # N = 8

Config = Dict[str, int]
Predicate = Callable[[Config], bool]


# ---------------------------------------------------------------- bit order
def index_to_config(i: int) -> Config:
    """Basis-state index -> {P,A,W}.  Bit k of i is qubit k."""
    return {name: (i >> k) & 1 for k, name in enumerate(FEATURES)}


def config_to_index(cfg: Config) -> int:
    return sum(int(cfg[name]) << k for k, name in enumerate(FEATURES))


def index_to_bitstring(i: int) -> str:
    """Qiskit-printed bitstring (q2 q1 q0 = W A P)."""
    return format(i, f"0{N_QUBITS}b")


def bitstring_to_config(bits: str) -> Config:
    """Decode a Qiskit counts key ('W A P' order) into feature names."""
    bits = bits.replace(" ", "")
    if len(bits) != N_QUBITS or set(bits) - {"0", "1"}:
        raise ValueError(f"bad bitstring {bits!r}")
    return index_to_config(int(bits, 2))


def config_to_pap(cfg: Config) -> str:
    """Human 'P A W' order string, e.g. '110' for {P:1,A:1,W:0}."""
    return "".join(str(cfg[n]) for n in FEATURES)


def fmt_config(cfg: Config) -> str:
    return "{" + ",".join(f"{n}:{cfg[n]}" for n in FEATURES) + "}"


# ---------------------------------------------------------- requirements
def r1(c: Config) -> bool:
    return bool(c["P"] or c["W"])


def r2(c: Config) -> bool:
    return bool((not c["P"]) or c["A"])


def r3(c: Config) -> bool:
    return not (c["P"] and c["W"])


REQUIREMENTS = (("R1", r1), ("R2", r2), ("R3", r3))


def is_valid(c: Config) -> bool:
    """The checker.  Cost O(L) with L = number of literals (here 3 rules)."""
    return r1(c) and r2(c) and r3(c)


# ----------------------------------------------------------- enumeration
class CountingChecker:
    """Wraps a predicate and counts how many times it is called."""

    def __init__(self, pred: Predicate):
        self.pred = pred
        self.calls = 0

    def __call__(self, c: Config) -> bool:
        self.calls += 1
        return self.pred(c)


def enumerate_solutions(pred: Predicate = is_valid) -> List[int]:
    """Exhaustive search over all N=8 configurations; returns valid indices."""
    return [i for i in range(N_STATES) if pred(index_to_config(i))]


def count_solutions(pred: Predicate = is_valid) -> int:
    return len(enumerate_solutions(pred))


def truth_table_rows(pred: Predicate = is_valid, extra: Dict[str, Predicate] | None = None):
    rows = []
    for i in range(N_STATES):
        c = index_to_config(i)
        row = {"index": i, "qiskit": index_to_bitstring(i), "paw": config_to_pap(c), **c}
        for name, f in REQUIREMENTS:
            row[name] = int(f(c))
        for name, f in (extra or {}).items():
            row[name] = int(f(c))
        row["valid"] = int(pred(c))
        rows.append(row)
    return rows


def truth_table_text(pred: Predicate = is_valid, extra: Dict[str, Predicate] | None = None) -> str:
    extra = extra or {}
    cols = ["R1", "R2", "R3", *extra.keys()]
    head = f"{'idx':>3} | {'Qiskit(WAP)':^11} | {'P A W':^5} | " + " ".join(f"{c:>6}" for c in cols) + " | valid | config"
    lines = [head, "-" * len(head)]
    for r in truth_table_rows(pred, extra):
        cfg = {n: r[n] for n in FEATURES}
        lines.append(
            f"{r['index']:>3} | {r['qiskit']:^11} | {r['P']} {r['A']} {r['W']} | "
            + " ".join(f"{r[c]:>6}" for c in cols)
            + f" | {'  YES' if r['valid'] else '   no'} | {fmt_config(cfg)}"
        )
    return "\n".join(lines)


def report() -> str:
    sols = enumerate_solutions()
    out = [
        "TASK A.1  Classical checker + exhaustive enumeration",
        "Qubit map: q0=P, q1=A, q2=W.  Qiskit prints bitstrings as  W A P  (q2 q1 q0).",
        "",
        truth_table_text(),
        "",
        f"K = {len(sols)} valid configurations out of N = {N_STATES}:",
    ]
    for i in sols:
        c = index_to_config(i)
        out.append(f"  {fmt_config(c)}   P-A-W order '{config_to_pap(c)}'   Qiskit key '{index_to_bitstring(i)}'")
    out.append("  -> in P-A-W reading order the valid set is {001, 011, 110};")
    out.append("     the SAME states as Qiskit keys (W-A-P) are {100, 110, 011}.")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())

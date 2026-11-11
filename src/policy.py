"""TASK B: an extra Boolean policy over P, A, W.

Grammar (enforced with an ``ast`` whitelist, never ``eval``):
    expr := P | A | W | ~expr | expr & expr | expr | expr | expr ^ expr | ( expr )

Pipeline
--------
1. parse + validate the policy (clear PolicyError on bad input / unknown names)
2. count K classically over all 8 configurations of  R1 & R2 & R3 & policy
3. K == 0  -> report "no configuration exists", PROOF = exhaustive truth table;
             Grover is NOT run.
4. K  > 0  -> build the oracle with Qiskit's PhaseOracleGate, verify its
             diagonal equals (-1)^f(x) for every x (this also verifies the bit
             order), choose t from K, run Grover, re-check the answer classically.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from .classical import (
    FEATURES, N_STATES, fmt_config, index_to_bitstring, index_to_config, is_valid, truth_table_text,
)
from .grover import SEED, SHOTS, choose_iterations, counts_table, solve, theta_for

MAX_LEN = 200
MAX_DEPTH = 30
BASE_EXPR = "(P | W) & (~P | A) & ~(P & W)"      # R1 & R2 & R3


class PolicyError(ValueError):
    """Raised for any policy string that is not in the whitelisted grammar."""


_BINOPS = {ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^"}
_HINTS = {
    ast.BoolOp: "use '&' / '|' instead of 'and' / 'or'",
    ast.Not: "use '~' instead of 'not'",
    ast.Constant: "constants are not allowed; use only the variables P, A, W",
    ast.Call: "function calls are not allowed",
    ast.Attribute: "attribute access is not allowed",
    ast.Compare: "comparisons are not allowed; use & | ^ ~",
    ast.Subscript: "indexing is not allowed",
}


# --------------------------------------------------------------- parsing
def parse_policy(text: str) -> ast.Expression:
    if not isinstance(text, str):
        raise PolicyError("policy must be a string")
    if not text.strip():
        raise PolicyError("policy is empty")
    if len(text) > MAX_LEN:
        raise PolicyError(f"policy too long ({len(text)} > {MAX_LEN} characters)")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        raise PolicyError(f"invalid syntax in {text!r} at column {e.offset}: {e.msg}") from None
    _validate(tree.body, 0)
    return tree


def _validate(node: ast.AST, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise PolicyError("policy nested too deeply")
    if isinstance(node, ast.Name):
        if node.id not in FEATURES:
            raise PolicyError(f"unknown variable {node.id!r}; allowed variables are P, A, W (case-sensitive)")
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.Invert):
            raise PolicyError(_HINTS.get(type(node.op), f"unary operator {type(node.op).__name__} not allowed; only '~'"))
        return _validate(node.operand, depth + 1)
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise PolicyError(f"operator {type(node.op).__name__} not allowed; only & | ^ ~ and parentheses")
        _validate(node.left, depth + 1)
        _validate(node.right, depth + 1)
        return
    raise PolicyError(_HINTS.get(type(node), f"{type(node).__name__} is not allowed in a policy"))


def evaluate(tree: ast.Expression | ast.AST, cfg: dict) -> bool:
    """Interpret a validated tree on a configuration (no eval)."""
    node = tree.body if isinstance(tree, ast.Expression) else tree
    if isinstance(node, ast.Name):
        return bool(cfg[node.id])
    if isinstance(node, ast.UnaryOp):
        return not evaluate(node.operand, cfg)
    a, b = evaluate(node.left, cfg), evaluate(node.right, cfg)
    op = type(node.op)
    return (a and b) if op is ast.BitAnd else (a or b) if op is ast.BitOr else (a != b)


def canonical(tree: ast.Expression) -> str:
    return ast.unparse(tree)


# --------------------------------------------------------- oracle (lib)
def build_phase_oracle(expression: str) -> QuantumCircuit:
    """PhaseOracleGate on exactly 3 qubits with q0=P, q1=A, q2=W.

    Checked against the installed Qiskit 2.6:  ``qiskit.circuit.library.PhaseOracleGate``
    (native synthesis, no tweedledum).  Observed behaviour:
      * the gate has one qubit per variable that APPEARS in the expression
        (``"P & ~A"`` -> 2 qubits, even with var_order=['P','A','W']);
      * gate qubit k  <->  ``gate.boolean_expression.args[k]``, which is var_order
        filtered to the present variables (default: order of first appearance,
        so ``"W & ~P"`` would put W on gate qubit 0!).
    We therefore pass var_order=['P','A','W'] AND wire each gate qubit to the
    project qubit of the variable it represents, then VERIFY the full 8-entry
    diagonal against the classical truth table.
    """
    from qiskit.circuit.library import PhaseOracleGate
    gate = PhaseOracleGate(expression, var_order=list(FEATURES))
    wires = [FEATURES.index(v) for v in gate.boolean_expression.args]
    qc = QuantumCircuit(len(FEATURES), name="PhaseOracle")
    qc.append(gate, wires)
    return qc


def oracle_diagonal(qc: QuantumCircuit) -> np.ndarray:
    U = Operator(qc).data
    off = U - np.diag(np.diag(U))
    if not np.allclose(off, 0):
        raise AssertionError("oracle is not diagonal")
    return np.real_if_close(np.diag(U))


def diagonal_matches(qc: QuantumCircuit, pred) -> bool:
    d = oracle_diagonal(qc)
    expected = np.array([-1.0 if pred(index_to_config(i)) else 1.0 for i in range(N_STATES)])
    return bool(np.allclose(d, expected))


# --------------------------------------------------------------- solver
@dataclass
class PolicyResult:
    policy: str
    expression: str
    K: int
    solutions: list
    table: str
    exists: bool
    t: Optional[int] = None
    diagonal_ok: Optional[bool] = None
    grover: dict = field(default_factory=dict)
    oracle: Optional[QuantumCircuit] = None


def make_predicate(tree):
    return lambda c: is_valid(c) and evaluate(tree, c)


def solve_policy(policy: str, run_grover: bool = True, shots: int = SHOTS, seed: int = SEED) -> PolicyResult:
    tree = parse_policy(policy)
    pol = canonical(tree)
    pred = make_predicate(tree)
    table = truth_table_text(pred, extra={"policy": lambda c: evaluate(tree, c)})
    sols = [i for i in range(N_STATES) if pred(index_to_config(i))]
    expr = f"{BASE_EXPR} & ({pol})"
    res = PolicyResult(policy=pol, expression=expr, K=len(sols), solutions=sols, table=table, exists=bool(sols))
    if not sols:
        return res                                       # do NOT run Grover
    oracle = build_phase_oracle(expr)
    res.oracle = oracle
    res.diagonal_ok = diagonal_matches(oracle, pred)
    if not res.diagonal_ok:
        raise AssertionError("PhaseOracleGate diagonal does not match the classical truth table")
    res.t = choose_iterations(res.K)
    if run_grover:
        res.grover = solve(oracle, res.K, pred=pred, shots=shots, seed=seed)
    return res


def report(policy: str) -> str:
    try:
        r = solve_policy(policy)
    except PolicyError as e:
        return f"Policy {policy!r}: REJECTED -> {e}"
    lines = [f"=== policy {policy!r}  (canonical: {r.policy}) ===",
             f"combined expression: {r.expression}",
             "exhaustive classical truth table (R1 & R2 & R3 & policy):", r.table,
             f"K = {r.K}"]
    if not r.exists:
        lines += ["RESULT: no configuration exists.",
                  "Proof: the truth table above enumerates all 8 = 2^3 assignments of (P,A,W) and the",
                  "'valid' column is 0 on every row.  Grover was NOT run (nothing to amplify).",
                  ]
        return "\n".join(lines)
    lines += [f"solutions: " + ", ".join(fmt_config(index_to_config(i)) + f" [key {index_to_bitstring(i)}]" for i in r.solutions),
              f"PhaseOracleGate diagonal vs classical (-1)^f, all 8 entries, q0=P q1=A q2=W: "
              f"{'MATCH' if r.diagonal_ok else 'MISMATCH'}",
              f"diagonal = {np.round(oracle_diagonal(r.oracle).real, 6).tolist()}  (index i = P + 2A + 4W)",
              f"theta = {np.degrees(theta_for(r.K)):.2f} deg  ->  t = {r.t}",
              counts_table(r.grover['counts'], make_predicate(parse_policy(policy))),
              f"measured success = {r.grover['success']:.4f}",
              f"answer: {fmt_config(r.grover['best_config'])}  classical re-check: "
              f"{'VALID' if r.grover['verified'] else 'INVALID'}"]
    return "\n".join(lines)


if __name__ == "__main__":
    for p in ("~W", "A", "P & ~A", "P and A", "X | P"):
        print(report(p), "\n")

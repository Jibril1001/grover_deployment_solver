import pytest

from src.classical import fmt_config, index_to_config
from src.policy import PolicyError, build_phase_oracle, diagonal_matches, make_predicate, parse_policy, solve_policy


@pytest.mark.parametrize("bad", [
    "", "   ", "P and A", "P or W", "not P", "X", "p", "P + A", "P == A", "1", "True",
    "__import__('os').system('ls')", "P.real", "f(P)", "(P | A", "P &", "P[0]", "-P", "P << A",
    "lambda: P", "P if A else W", "P" * 300,
])
def test_parser_rejects(bad):
    with pytest.raises(PolicyError):
        parse_policy(bad)


def test_unknown_name_message():
    with pytest.raises(PolicyError, match="unknown variable 'X'"):
        parse_policy("X | P")


def test_and_hint():
    with pytest.raises(PolicyError, match="use '&'"):
        parse_policy("P and A")


@pytest.mark.parametrize("ok", ["P", "~W", "(P | A) & ~(W ^ A)", "~~P", "P^A^W"])
def test_parser_accepts(ok):
    parse_policy(ok)


def _expected(policy):
    pred = make_predicate(parse_policy(policy))
    return {fmt_config(index_to_config(i)) for i in range(8) if pred(index_to_config(i))}


def test_exactly_one_solution():
    assert _expected("~W") == {"{P:1,A:1,W:0}"}
    r = solve_policy("~W", shots=2048)
    assert r.K == 1 and r.t == 2 and r.diagonal_ok
    assert fmt_config(r.grover["best_config"]) == "{P:1,A:1,W:0}" and r.grover["verified"]


def test_more_than_one_solution():
    assert _expected("A") == {"{P:1,A:1,W:0}", "{P:0,A:1,W:1}"}
    r = solve_policy("A", shots=2048)
    assert r.K == 2 and r.t == 1 and r.diagonal_ok
    assert r.grover["verified"] and r.grover["success"] == pytest.approx(1.0)


def test_no_solution_does_not_run_grover():
    assert _expected("P & ~A") == set()
    r = solve_policy("P & ~A")
    assert r.K == 0 and not r.exists
    assert r.grover == {} and r.oracle is None and r.t is None
    assert r.table.count("   no |") == 8          # proof: every one of the 8 rows is invalid


@pytest.mark.parametrize("expr", ["P", "A", "W", "P & ~A", "(P | W) & ~(P & W)", "P ^ A"])
def test_phase_oracle_gate_bit_order(expr):
    pred = make_predicate(parse_policy(expr))
    # check the gate alone against the policy (without R1-R3) for a stronger bit-order test
    tree = parse_policy(expr)
    from src.policy import evaluate
    assert diagonal_matches(build_phase_oracle(expr), lambda c: evaluate(tree, c))


def test_phase_oracle_gate_appearance_order_trap():
    # 'W & ~P' lists W first; without explicit wiring W would land on q0 (=P).
    assert diagonal_matches(build_phase_oracle("W & ~P"), lambda c: c["W"] and not c["P"])

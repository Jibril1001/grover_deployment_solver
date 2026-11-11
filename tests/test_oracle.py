import pytest

from src.classical import count_solutions, is_valid
from src.grover import solve
from src.manual_oracle import build_oracle_broken, build_oracle_v1, build_oracle_v2, verify_oracle

ALLOWED = {"x", "cx", "ccx", "h", "barrier"}


@pytest.mark.parametrize("builder", [build_oracle_v1, build_oracle_v2])
def test_oracle_phase_inputs_and_cleanup(builder):
    rows = verify_oracle(builder())
    assert len(rows) == 8
    for r in rows:
        assert r.phase == pytest.approx(-1.0 if is_valid(r.config) else 1.0)
        assert r.input_preserved
        assert r.helpers_clean


@pytest.mark.parametrize("builder", [build_oracle_v1, build_oracle_v2, build_oracle_broken])
def test_only_allowed_gates(builder):
    qc = builder()
    assert set(qc.count_ops()) <= ALLOWED
    assert qc.count_ops().get("h", 0) == 2      # only the |-> kick prep/unprep


def test_v2_is_smaller():
    assert build_oracle_v2().num_qubits < build_oracle_v1().num_qubits
    assert sum(build_oracle_v2().count_ops().values()) < sum(build_oracle_v1().count_ops().values())


def test_broken_oracle_keeps_phase_but_leaves_garbage():
    rows = verify_oracle(build_oracle_broken())
    assert all(r.phase_ok and r.input_preserved for r in rows)
    assert not all(r.helpers_clean for r in rows)


@pytest.mark.parametrize("builder", [build_oracle_v1, build_oracle_v2])
def test_grover_finds_valid(builder):
    r = solve(builder(), count_solutions(), shots=2048)
    assert r["t"] == 1 and r["verified"]
    assert abs(r["success"] - 0.84375) < 0.04

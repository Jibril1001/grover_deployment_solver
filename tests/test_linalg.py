import math

import numpy as np
import pytest

from src import linalg as la


def test_column_vectors_and_index():
    v = la.basis_ket(P=1, A=1, W=0)
    assert v.shape == (8, 1) and v[3, 0] == 1 and v.sum() == 1      # index = P + 2A + 4W


def test_uniform_state():
    assert np.allclose(la.uniform_state(), 1 / math.sqrt(8))


def test_one_iteration_amplitudes():
    after = la.state_after(1)
    assert after[3, 0] == pytest.approx(1.5 / math.sqrt(8))
    assert after[0, 0] == pytest.approx(-0.5 / math.sqrt(8))
    assert la.success(after) == pytest.approx(0.84375)


def test_reduced_matrix_is_rotation_by_2theta():
    th = math.asin(math.sqrt(3 / 8))
    R = np.array([[math.cos(2 * th), -math.sin(2 * th)], [math.sin(2 * th), math.cos(2 * th)]])
    assert np.allclose(la.reduced_grover(), R)


def test_matrices_match_qiskit_circuits():
    assert all(la.cross_checks().values())


def test_gate_matrices():
    assert np.allclose(la.CX @ la.CX, np.eye(4))            # self-inverse
    assert np.allclose(la.CCX @ la.CCX, np.eye(8))
    minus = la.H @ la.X @ la.KET0                             # |-> = H X |0>
    assert np.allclose(la.X @ minus, -minus)                  # kick-back eigenvalue -1


def test_oracle_is_product_of_gate_matrices():
    gp = la.gate_products()
    assert np.allclose(gp["V1"]["block"], la.oracle_matrix())
    assert np.allclose(gp["V2"]["block"], la.oracle_matrix())
    assert np.allclose(gp["D"], -la.diffuser_matrix())
    assert np.allclose(gp["BROKEN"]["block"], 0)

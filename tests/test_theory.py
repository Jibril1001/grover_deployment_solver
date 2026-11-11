import pytest

from src.grover import choose_iterations
from src.theory import amplitude_step, predicted_success, wilson_ci


def test_amplitudes_K3():
    s = amplitude_step(3)
    sc = 8 ** 0.5
    assert s.mean * sc == pytest.approx(0.25)
    assert s.after_valid * sc == pytest.approx(1.5)
    assert s.after_invalid * sc == pytest.approx(-0.5)
    assert s.success_one_iter == pytest.approx(0.84375)


@pytest.mark.parametrize("K,t,p", [(3, 1, 0.84375), (2, 1, 1.0), (1, 2, 0.9453125)])
def test_preferred_t(K, t, p):
    assert choose_iterations(K) == t
    assert predicted_success(K, t) == pytest.approx(p)


def test_wilson():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi

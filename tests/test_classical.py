from src.classical import (bitstring_to_config, config_to_pap, count_solutions, enumerate_solutions,
                           index_to_bitstring, index_to_config)


def test_valid_set_and_K():
    sols = {config_to_pap(index_to_config(i)) for i in enumerate_solutions()}
    assert sols == {"001", "011", "110"}          # P-A-W reading order
    assert {index_to_bitstring(i) for i in enumerate_solutions()} == {"100", "110", "011"}  # Qiskit W-A-P
    assert count_solutions() == 3


def test_little_endian_decode():
    assert bitstring_to_config("011") == {"P": 1, "A": 1, "W": 0}
    assert bitstring_to_config("100") == {"P": 0, "A": 0, "W": 1}

"""Grover as explicit linear algebra (NumPy only), cross-checked against Qiskit.

Column vectors for one variable:   |0> = [1, 0]^T      |1> = [0, 1]^T
Three variables via the Kronecker (tensor) product, written in Qiskit order
(most significant first = q2 q1 q0 = W A P):

    |W A P> = |W> (x) |A> (x) |P>         ->  an 8x1 column vector

so the 1 sits at row  i = P + 2A + 4W  (the same index Qiskit uses).

Matrices (8x8):
    H3 = H (x) H (x) H                  |s> = H3 |000>   (all entries 1/sqrt(8))
    O  = diag((-1)^f(i))                phase oracle
    D  = 2|s><s| - I                    diffuser (inversion about the mean)
    G  = D O                            one Grover iteration
In the 2-D basis {|U>, |M>} (normalised sums of invalid / valid states) G is the
rotation matrix  [[cos 2θ, -sin 2θ], [sin 2θ, cos 2θ]].
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from qiskit.quantum_info import Operator, Statevector

from .classical import N_STATES, fmt_config, index_to_bitstring, index_to_config, is_valid

KET0 = np.array([[1.0], [0.0]])
KET1 = np.array([[0.0], [1.0]])
H = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2)
I8 = np.eye(N_STATES)


# ----------------------------------------------------------------- vectors
def ket(bit: int) -> np.ndarray:
    return KET1 if bit else KET0


def basis_ket(P: int, A: int, W: int) -> np.ndarray:
    """|W A P> = |W> (x) |A> (x) |P>   (8x1 column vector)."""
    return np.kron(np.kron(ket(W), ket(A)), ket(P))


def uniform_state() -> np.ndarray:
    H3 = np.kron(np.kron(H, H), H)
    return H3 @ basis_ket(0, 0, 0)


# ---------------------------------------------------------------- matrices
def oracle_matrix(pred: Callable = is_valid) -> np.ndarray:
    return np.diag([-1.0 if pred(index_to_config(i)) else 1.0 for i in range(N_STATES)])


def diffuser_matrix() -> np.ndarray:
    s = uniform_state()
    return 2 * (s @ s.T) - I8


def grover_matrix(pred: Callable = is_valid) -> np.ndarray:
    return diffuser_matrix() @ oracle_matrix(pred)


def state_after(t: int, pred: Callable = is_valid) -> np.ndarray:
    return np.linalg.matrix_power(grover_matrix(pred), t) @ uniform_state()


def success(vec: np.ndarray, pred: Callable = is_valid) -> float:
    return float(sum(vec[i, 0] ** 2 for i in range(N_STATES) if pred(index_to_config(i))))


def two_d_basis(pred: Callable = is_valid):
    valid = np.array([[1.0 if pred(index_to_config(i)) else 0.0] for i in range(N_STATES)])
    K = int(valid.sum())
    M = valid / math.sqrt(K)
    U = (1 - valid) / math.sqrt(N_STATES - K)
    return U, M


def reduced_grover(pred: Callable = is_valid) -> np.ndarray:
    """2x2 matrix of G in the basis {|U>, |M>}:  B^T G B with B = [U M]."""
    U, M = two_d_basis(pred)
    B = np.hstack([U, M])
    return B.T @ grover_matrix(pred) @ B


# ------------------------------------------------------- gates as matrices
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
I2 = np.eye(2)
# multi-qubit gates in their own little-endian basis (control(s) = low qubits, target = highest)
CX = np.array([[1, 0, 0, 0],          # basis |t c>: 00, 01, 10, 11
               [0, 0, 0, 1],          # |01> (c=1,t=0) -> |11>
               [0, 0, 1, 0],
               [0, 1, 0, 0]], dtype=float)
CCX = np.eye(8)[:, [0, 1, 2, 7, 4, 5, 6, 3]]   # swaps |011> <-> |111>  (flip t iff c1=c2=1)
CCZ = np.diag([1.0, 1, 1, 1, 1, 1, 1, -1])     # -1 only on |111>
GATES = {"x": X, "h": H, "z": Z, "cx": CX, "ccx": CCX, "ccz": CCZ}


def op_on(U: np.ndarray, qubits, n: int) -> np.ndarray:
    """Embed a k-qubit gate U acting on `qubits` (U's qubit j = qubits[j]) into a 2^n x 2^n matrix."""
    k = len(qubits)
    full = np.zeros((2 ** n, 2 ** n))
    for col in range(2 ** n):
        local_in = sum(((col >> q) & 1) << j for j, q in enumerate(qubits))
        rest = col & ~sum(1 << q for q in qubits)
        for local_out in range(2 ** k):
            amp = U[local_out, local_in]
            if amp:
                row = rest | sum(((local_out >> j) & 1) << q for j, q in enumerate(qubits))
                full[row, col] += amp
    return full


def circuit_matrix(ops, n: int) -> np.ndarray:
    """Matrix of a gate list.  Circuits run left->right, so  M = U_last ... U_2 U_1."""
    M = np.eye(2 ** n)
    for name, qs in ops:
        M = op_on(GATES[name], qs, n) @ M
    return M


def oracle_ops(version: int):
    """The exact gate lists used by src/manual_oracle.py, including the kick prep."""
    from . import manual_oracle as mo
    if version == 1:
        k, comp, kick = mo.V1_KICK, mo.V1_COMPUTE, [("ccx", (mo.V1_AND, mo.V1_R3, mo.V1_KICK))]
        n = 8
    else:
        k, comp, kick = mo.V2_KICK, mo.V2_COMPUTE, [("ccx", (mo.W, mo.V2_H, mo.V2_KICK))]
        n = 5
    ops = [("x", (k,)), ("h", (k,))] + comp + kick + list(reversed(comp)) + [("h", (k,)), ("x", (k,))]
    return ops, n


def diffuser_ops():
    layer = lambda g: [(g, (q,)) for q in range(3)]
    return layer("h") + layer("x") + [("ccz", (0, 1, 2))] + layer("x") + layer("h")


def gate_products() -> dict:
    """Oracle and diffuser built ONLY from gate-matrix products (no Qiskit)."""
    out = {}
    for v in (1, 2):
        ops, n = oracle_ops(v)
        M = circuit_matrix(ops, n)
        out[f"V{v}"] = {"n": n, "gates": len(ops), "M": M, "block": M[:N_STATES, :N_STATES]}
    from . import manual_oracle as mo
    ops1, _ = oracle_ops(1)
    k = mo.V1_KICK
    broken = [("x", (k,)), ("h", (k,))] + mo.V1_COMPUTE + [("ccx", (mo.V1_AND, mo.V1_R3, k))] + [("h", (k,)), ("x", (k,))]
    Mb = circuit_matrix(broken, 8)
    out["BROKEN"] = {"M": Mb, "block": Mb[:N_STATES, :N_STATES]}
    out["D"] = circuit_matrix(diffuser_ops(), 3)
    # CCZ from CCX:  (I(x)I(x)H) CCX (I(x)I(x)H)
    Hq2 = op_on(H, [2], 3)
    out["CCZ_from_CCX"] = Hq2 @ CCX @ Hq2
    return out


# ------------------------------------------------- cross-check vs Qiskit
def circuit_oracle_block(oracle_qc) -> np.ndarray:
    """8x8 block of the oracle's unitary with every ancilla in |0> (in and out)."""
    return np.real_if_close(Operator(oracle_qc).data[:N_STATES, :N_STATES])


def cross_checks() -> dict:
    from .grover import diffuser, grover_circuit
    from .manual_oracle import build_oracle_v1, build_oracle_v2
    O = oracle_matrix()
    out = {
        "H3|000> = |s> (all 1/sqrt8)": np.allclose(uniform_state(), np.full((8, 1), 1 / math.sqrt(8))),
        "manual oracle V1 block == O": np.allclose(circuit_oracle_block(build_oracle_v1()), O),
        "manual oracle V2 block == O": np.allclose(circuit_oracle_block(build_oracle_v2()), O),
        # circuit H X CCZ X H implements I - 2|s><s| = -D (global phase -1)
        "diffuser circuit == -(2|s><s| - I)": np.allclose(Operator(diffuser()).data, -diffuser_matrix()),
        "G is unitary (G^T G = I)": np.allclose(grover_matrix().T @ grover_matrix(), I8),
    }
    gp = gate_products()
    out["V1 gate-matrix product (256x256) block == O, helpers return to |0>"] = (
        np.allclose(gp["V1"]["block"], O) and np.allclose(np.abs(gp["V1"]["M"][:N_STATES, :N_STATES]), I8))
    out["V2 gate-matrix product (32x32) block == O"] = np.allclose(gp["V2"]["block"], O)
    out["V1 gate-matrix product == Qiskit Operator(V1)"] = np.allclose(gp["V1"]["M"], Operator(build_oracle_v1()).data)
    out["H X CCZ X H gate product == -(2|s><s| - I)"] = np.allclose(gp["D"], -diffuser_matrix())
    out["BROKEN oracle block == 0 matrix (garbage leaves the helpers-|0> subspace)"] = np.allclose(gp["BROKEN"]["block"], 0)
    out["H.CCX.H == CCZ"] = np.allclose(gp["CCZ_from_CCX"], CCZ)
    out["X|-> = -|->  (kick-back eigenvalue)"] = np.allclose(X @ (H @ X @ KET0), -(H @ X @ KET0))
    for t in range(5):
        sv = Statevector(grover_circuit(build_oracle_v2(), t, measure=False)).data[:N_STATES]
        out[f"Qiskit statevector t={t} == G^t|s> (up to global sign)"] = np.allclose(
            np.abs(sv), np.abs(state_after(t)[:, 0])) and np.allclose(sv, (-1) ** t * state_after(t)[:, 0])
    return out


# ------------------------------------------------------------------ report
def _vec(v: np.ndarray, pred=is_valid) -> str:
    rows = []
    for i in range(N_STATES):
        tag = "valid" if pred(index_to_config(i)) else ""
        rows.append(f"    [{v[i, 0]:+.4f}]   |{index_to_bitstring(i)}>  {fmt_config(index_to_config(i))}  {tag}")
    return "\n".join(rows)


def _mat(m: np.ndarray, fmt="{:+.3f}") -> str:
    return "\n".join("    [" + " ".join(fmt.format(x) for x in row) + "]" for row in m)


def report() -> str:
    s = uniform_state()
    O, D = oracle_matrix(), diffuser_matrix()
    marked = O @ s
    after = D @ marked
    th = math.asin(math.sqrt(3 / 8))
    R = reduced_grover()
    eig = np.linalg.eigvals(R)
    lines = [
        "Column vectors for one variable:  |0> = [1, 0]^T,  |1> = [0, 1]^T",
        "Example: {P:1,A:1,W:0} = |W>(x)|A>(x)|P> = |0>(x)|1>(x)|1> = |011>  -> 8x1 vector with a 1 at row 3:",
        "    " + str(basis_ket(1, 1, 0).T.astype(int)[0].tolist()) + "^T",
        "",
        "H (2x2) =", _mat(H, "{:+.4f}"),
        "|s> = (H(x)H(x)H)|000>  (8x1):", _vec(s),
        "",
        "GATES AS MATRICES",
        "X =", _mat(X, "{:+.0f}"), "Z =", _mat(Z, "{:+.0f}"),
        "CX (4x4, basis |target control>: 00,01,10,11) =", _mat(CX, "{:+.0f}"),
        "CCX (8x8 permutation, swaps |011> and |111>) =", _mat(CCX, "{:+.0f}"),
        "CCZ = diag(1,1,1,1,1,1,1,-1);  check  H.CCX.H on the target == CCZ",
        "Kick qubit: |-> = H X |0> = " + str(np.round((H @ X @ KET0).T[0], 4).tolist())
        + "^T,   X|-> = " + str(np.round((X @ H @ X @ KET0).T[0], 4).tolist()) + "^T = -|->",
        "A gate on some qubits of an n-qubit register is embedded as a 2^n x 2^n matrix (tensor with I);",
        "a circuit is the PRODUCT of its gate matrices, applied right-to-left:  M = U_last ... U_2 U_1.",
        "",
        "ORACLE FROM GATE MATRICES ONLY (no Qiskit):",
    ]
    gp = gate_products()
    for v in (1, 2):
        g = gp[f"V{v}"]
        lines.append(f"    V{v}: product of {g['gates']} gate matrices, each {2**g['n']}x{2**g['n']}; "
                     f"8x8 block with all helpers |0> has diag = {np.diag(g['block']).round(6).astype(int).tolist()}")
    lines += [
        "    BROKEN (no uncompute): its 8x8 helpers-|0> block is the ZERO matrix "
        f"(max |entry| = {np.abs(gp['BROKEN']['block']).max():.0f}): every input column lands on a row with "
        "non-zero helper bits, i.e. the oracle is no longer a diagonal +-1 matrix on the search space.",
        "DIFFUSER FROM GATE MATRICES:  (H(x)3)(X(x)3) CCZ (X(x)3)(H(x)3) = I - 2|s><s| = -D  (global sign -1, unobservable)",
        "",
        "Oracle O = diag((-1)^f(i)), i = 0..7 (index = P + 2A + 4W):",
        "    diag = " + str(np.diag(O).astype(int).tolist()),
        "O|s>  (phase-marked):", _vec(marked),
        "",
        "Diffuser D = 2|s><s| - I  (every entry of |s><s| is 1/8, so D = (1/4)J - I):",
        _mat(D),
        "D O|s>  (after one Grover iteration):", _vec(after),
        f"success = sum of squares of valid entries = {success(after):.5f}",
        "",
        "Same thing in the 2-D basis {|U>, |M>}  (|M> = sum of valid / sqrt3, |U> = sum of invalid / sqrt5):",
        f"    |s> = cos(theta)|U> + sin(theta)|M>,  theta = asin(sqrt(3/8)) = {math.degrees(th):.2f} deg",
        "    G restricted to that plane  (B^T G B, B = [U M]):", _mat(R, "{:+.4f}"),
        f"    = rotation by 2theta = {math.degrees(2*th):.2f} deg:  [[cos2θ, -sin2θ], [sin2θ, cos2θ]] = "
        f"[[{math.cos(2*th):+.4f}, {-math.sin(2*th):+.4f}], [{math.sin(2*th):+.4f}, {math.cos(2*th):+.4f}]]",
        f"    eigenvalues = {np.round(eig, 4).tolist()}  (= e^(±i 2θ), |λ| = 1)",
        "",
        "Success after t iterations from the matrix power  G^t|s>  vs the formula sin^2((2t+1)θ):",
    ]
    for K, pred, label in _cases():
        thk = math.asin(math.sqrt(K / 8))
        vals = " ".join(f"t={t}: {success(state_after(t, pred), pred):.4f}/{math.sin((2*t+1)*thk)**2:.4f}" for t in range(5))
        lines.append(f"    K={K} ({label}): {vals}")
    lines += ["", "Cross-checks against the real Qiskit circuits:"]
    for k, v in cross_checks().items():
        lines.append(f"    [{'PASS' if v else 'FAIL'}] {k}")
    return "\n".join(lines)


def _cases():
    from .policy import make_predicate, parse_policy
    return [(3, is_valid, "original"), (2, make_predicate(parse_policy("A")), "policy A"),
            (1, make_predicate(parse_policy("~W")), "policy ~W")]


if __name__ == "__main__":
    print(report())

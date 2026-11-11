"""Reproduce every result, table and figure:   python run_all.py

Fixed seed (SEED=1234 in src/grover.py) for the simulator and transpiler.
Text goes to stdout AND outputs/*.txt, figures to outputs/*.png.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

from src import classical, grover, manual_oracle as mo, plots, policy, theory  # noqa: E402
from src.classical import count_solutions, is_valid  # noqa: E402
from src.grover import SEED, SHOTS  # noqa: E402

random.seed(SEED)
np.random.seed(SEED)

_ALL: list[str] = []


def section(fname: str, title: str, body: str) -> None:
    text = f"{'=' * 100}\n{title}\n{'=' * 100}\n{body}\n"
    print(text)
    (OUT / fname).write_text(text)
    _ALL.append(text)


def draw(qc, name):
    qc.draw("mpl", filename=str(OUT / name), fold=-1, style={"backgroundcolor": "#fcfcfb"})
    import matplotlib.pyplot as plt
    plt.close("all")


def main():
    import qiskit, qiskit_aer
    header = (f"Grover Solver for Deployment Configuration\n"
              f"qiskit {qiskit.__version__} | qiskit-aer {qiskit_aer.__version__} | numpy {np.__version__} | "
              f"seed {SEED} | shots {SHOTS}\n"
              f"Variable/bit order: q0=P, q1=A, q2=W; Qiskit prints bitstrings as W A P (q2 q1 q0).")
    print(header)
    _ALL.append(header + "\n")

    # ---------------------------------------------------------------- A.1
    section("01_classical.txt", "TASK A.1  CLASSICAL CHECKER", classical.report())
    K3 = count_solutions()

    # ---------------------------------------------------------------- A.2
    v1, v2, bad = mo.build_oracle_v1(), mo.build_oracle_v2(), mo.build_oracle_broken()
    body = "\n\n".join([
        mo.check_table("Oracle V1 (helpers r1,r2,r3 + AND helper + |-> kick, explicit reverse uncompute)",
                       mo.verify_oracle(v1)),
        mo.check_table("Oracle V2 (minimal: W' = P xor W = R1&R3, helper r2)", mo.verify_oracle(v2)),
        mo.check_table("BROKEN oracle (V1 with the uncompute step skipped) -- expected to FAIL cleanup",
                       mo.verify_oracle(bad)),
        "Resource comparison:\n" + mo.compare_table({"V1": v1, "V2": v2, "BROKEN": bad}),
    ])
    section("02_oracle_checks.txt", "TASK A.2  MANUAL PHASE ORACLES (X, CX, CCX only)", body)
    draw(v1, "oracle_v1.png")
    draw(v2, "oracle_v2.png")
    draw(bad, "oracle_broken.png")
    draw(grover.diffuser(), "diffuser.png")
    draw(grover.grover_circuit(v2, 1), "grover_v2_t1.png")

    # ---------------------------------------------------------------- A.3
    t3 = grover.choose_iterations(K3)
    r_v1 = grover.solve(v1, K3)
    body = "\n\n".join([grover.report(v1, K3, "manual oracle V1"),
                        grover.report(v2, K3, "manual oracle V2")])
    section("03_grover.txt", "TASK A.3  GROVER SEARCH (AerSimulator)", body)
    plots.plot_histogram(r_v1["counts"], is_valid,
                         f"K=3 (original requirements), oracle V1, t={t3}, {SHOTS} shots — success "
                         f"{r_v1['success']:.3f}", OUT / "hist_K3.png")

    # ---------------------------------------------------------------- B.4
    reports = [policy.report(p) for p in ("~W", "A", "P & ~A")]
    reports.append("Parser rejection demos:\n" + "\n".join(
        "  " + policy.report(p) for p in ("P and A", "not W", "X | P", "P +  A", "__import__('os')",
                                          "P & ", "(P | A", "1", "P.real", "", "p & a")))
    section("04_policy.txt", "TASK B.4  EXTRA POLICY + PhaseOracleGate", "\n\n".join(reports))
    pol = {p: policy.solve_policy(p) for p in ("A", "~W")}
    for p, r in pol.items():
        pred = policy.make_predicate(policy.parse_policy(p))
        plots.plot_histogram(r.grover["counts"], pred,
                             f"K={r.K} (policy '{p}'), PhaseOracleGate, t={r.t}, {SHOTS} shots — "
                             f"success {r.grover['success']:.3f}", OUT / f"hist_K{r.K}.png")

    # ------------------------------------------------------------- theory
    cases = {3: (v1, is_valid, "K=3 original requirements, oracle V1"),
             2: (pol["A"].oracle, policy.make_predicate(policy.parse_policy("A")), "K=2 policy 'A'"),
             1: (pol["~W"].oracle, policy.make_predicate(policy.parse_policy("~W")), "K=1 policy '~W'")}
    parts, sweeps = [], {}
    for K, (orc, pred, label) in cases.items():
        parts.append(theory.amplitude_table(K))
        parts.append(theory.prediction_table(K))
        sweeps[K] = theory.measure_sweep(orc, K, pred)
        parts.append(theory.sweep_table(sweeps[K], label))
        plots.plot_amplitudes(K, pred, OUT / f"amplitudes_K{K}.png")
    parts.append(theory.preferred_t_summary())
    parts.append(theory.OVERSHOOT_NOTE)
    parts.append("""Iteration strategy: K is counted CLASSICALLY first, then
  t = max(1, round(pi/(4*theta) - 1/2)),  theta = asin(sqrt(K/N)).
  If K were unknown we would need a randomized (BBHT exponential search) or a quantum-counting
  approach; this solver relies on the classical count, which at N=8 already solves the problem.""")
    section("05_theory.txt", "THEORY + EXPERIMENTS (5, 6, E)", "\n\n".join(parts))
    plots.plot_success_curves(sweeps, OUT / "success_vs_iterations.png")
    plots.plot_rotation(OUT / "rotation_geometry.png")

    # ----------------------------------------------------------- ablation
    ok = grover.solve(v1, K3, t=1, seed=SEED + 7)
    br = grover.solve(bad, K3, t=1, seed=SEED + 7)
    pred1 = theory.predicted_success(K3, 1)
    ex_ok, ex_bad = theory.exact_success(v1, 1), theory.exact_success(bad, 1)
    pur_ok, pur_bad = theory.search_register_purity(v1), theory.search_register_purity(bad)
    m_bad = theory.Measurement(3, 1, SHOTS, round(br["success"] * SHOTS), pred1)
    body = f"""{mo.check_table('BROKEN oracle check table', mo.verify_oracle(bad))}

                         correct V1      broken (no uncompute)
helpers back to |0>?     yes             NO
purity Tr(rho^2) of the
 search register after
 one oracle call on |s>  {pur_ok:.4f}          {pur_bad:.4f}
exact P(valid), t=1      {ex_ok:.4f}          {ex_bad:.4f}
measured P(valid), t=1   {ok['success']:.4f}          {br['success']:.4f}   (95% CI [{m_bad.ci[0]:.4f}, {m_bad.ci[1]:.4f}])
predicted (ideal)        {pred1:.4f}
broken vs prediction:    z = {m_bad.z:+.1f} sigma  -> {'FAR below the predicted 0.844' if br['success'] < pred1 else 'unexpected'}

Why: without uncompute the state after the oracle is  sum_x (-1)^f(x) |x>|g(x)>,  where the
workspace g(x) = (r1, r2, r3, and) is a function of x.  The workspace now holds which-path
information, i.e. it is ENTANGLED with the search register (purity {pur_bad:.3f} < 1: the search
register alone is a mixed state).  The diffuser acts only on the search qubits, so it can only
make amplitudes interfere inside each group of x that share the same g(x).
Sectors here: all 3 valid x share g=(1,1,1,1); the invalid ones split into {{000,010}}, {{100}},
{{101}}, {{111}} (P-A-W order).  Inversion about the mean inside each tiny sector cannot pump
amplitude into the valid states: the exact success drops to {ex_bad:.4f} -- even below the
no-Grover baseline 3/8 = 0.375.  Uncomputation is what disentangles the workspace and makes
the oracle a pure phase  |x> -> (-1)^f(x)|x>.
"""
    section("06_ablation.txt", "ADDITION D  UNCOMPUTATION ABLATION", body)
    plots.plot_ablation(ok["counts"], br["counts"], ok["success"], br["success"], pred1, OUT / "ablation.png")

    # ----------------------------------------------------- linear algebra
    from src import linalg
    section("08_linear_algebra.txt", "LINEAR-ALGEBRA VIEW: column vectors, O, D = 2|s><s| - I, G = D O",
            linalg.report())
    plots.plot_linear_algebra(OUT / "linear_algebra.png")
    plots.plot_gate_matrices(OUT / "gate_matrices.png")

    # --------------------------------------------------------------- cost
    oc = mo.resource_counts(v1)
    full = grover.grover_circuit(v1, 1)
    from qiskit import transpile
    tf = transpile(full, basis_gates=["cx", "u", "measure"], optimization_level=0)
    gc = {"qubits": full.num_qubits, "ops_after_decomp": dict(tf.count_ops()), "depth": tf.depth()}
    section("07_cost.txt", "COST DISCUSSION (8, G)", theory.cost_report(
        {k: oc[k] for k in ("qubits", "x", "cx", "ccx", "h", "total_gates", "cx_after_decomp")}, gc))

    (OUT / "results.txt").write_text("\n".join(_ALL))
    print(f"\nAll outputs written to {OUT}")
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()

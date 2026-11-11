"""All figures. Static PNGs, one consistent palette:
valid = blue, invalid = neutral gray, reference/mean/broken = orange."""
from __future__ import annotations

import math
from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .classical import N_STATES, bitstring_to_config, fmt_config, index_to_bitstring, index_to_config, is_valid  # noqa: E402
from .grover import theta_for  # noqa: E402
from .theory import amplitude_step, predicted_success  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"      # valid / series 1
ORANGE = "#eb6834"    # mean line, broken oracle / series 2
AQUA = "#1baf7a"      # series 3
GRAY = "#a9a8a2"      # invalid
K_COLORS = {3: BLUE, 2: ORANGE, 1: AQUA}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.axisbelow": True,
    })


_style()


def _labels():
    return [f"{fmt_config(index_to_config(i))}\n{index_to_bitstring(i)}" for i in range(N_STATES)]


def _bar_colors(pred):
    return [BLUE if pred(index_to_config(i)) else GRAY for i in range(N_STATES)]


def _legend_valid(ax, loc="upper right"):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="valid"), Patch(color=GRAY, label="invalid")],
              frameon=False, loc=loc, fontsize=9, ncol=2)


# ------------------------------------------------------------ amplitudes
def plot_amplitudes(K: int, pred: Callable, path: str):
    s = amplitude_step(K)
    valid = np.array([pred(index_to_config(i)) for i in range(N_STATES)])
    stages = [
        ("1. initial  |s>", np.full(N_STATES, s.a0), None),
        ("2. after oracle (phase mark)", np.where(valid, s.marked_valid, s.marked_invalid), s.mean),
        ("3. after diffuser (2μ − a)", np.where(valid, s.after_valid, s.after_invalid), None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    x = np.arange(N_STATES)
    for ax, (title, amps, mean) in zip(axes, stages):
        ax.bar(x, amps, color=_bar_colors(pred), width=0.7, edgecolor=SURFACE, linewidth=2)
        ax.axhline(0, color=INK2, lw=0.8)
        if mean is not None:
            ax.axhline(mean, color=ORANGE, lw=2, ls="--")
            ax.text(N_STATES - 0.6, mean, f" mean μ = {mean:.3f}", color=INK, va="bottom", ha="right", fontsize=9)
        ax.set_title(title, loc="left")
        ax.set_xticks(x, _labels(), rotation=60, fontsize=7.5)
        for xi, a in zip(x, amps):
            if valid[xi]:
                ax.text(xi, a + (0.02 if a >= 0 else -0.02), f"{a:+.3f}", ha="center",
                        va="bottom" if a >= 0 else "top", fontsize=8, color=INK)
    axes[0].set_ylabel("amplitude (normalised)")
    _legend_valid(axes[0], loc="lower left")
    succ = K * s.after_valid ** 2
    fig.suptitle(f"Amplitudes for K={K}, N=8 — success after one iteration = K·v² = {succ:.5f}",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------- success curves
def plot_success_curves(sweeps: Dict[int, list], path: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    tt = np.linspace(0, 4.3, 400)
    for K, ms in sweeps.items():
        c = K_COLORS[K]
        th = theta_for(K)
        ax.plot(tt, np.sin((2 * tt + 1) * th) ** 2, color=c, lw=2, alpha=0.9)
        ts = [m.t for m in ms]
        ph = [m.p_hat for m in ms]
        lo = [m.p_hat - m.ci[0] for m in ms]
        hi = [m.ci[1] - m.p_hat for m in ms]
        ax.errorbar(ts, ph, yerr=[lo, hi], fmt="o", ms=8, color=c, mec=SURFACE, mew=2, capsize=4,
                    label=f"K={K}  (θ={math.degrees(th):.1f}°)")
        ax.text(4.32, math.sin(9.6 * th) ** 2, f"K={K}", color=INK, va="center", fontsize=9)
    ax.set_xlabel("Grover iterations t")
    ax.set_ylabel("P(measure a valid configuration)")
    ax.set_xticks(range(5))
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.2, 4.6)
    ax.set_title("Success vs iterations: sin²((2t+1)θ) (lines) vs 8192-shot measurement (95% Wilson CI)", loc="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------- geometry
def plot_rotation(path: str, cases=((3, 3), (2, 2), (1, 3))):
    fig, axes = plt.subplots(1, len(cases), figsize=(15, 5.2))
    for ax, (K, tmax) in zip(axes, cases):
        th = theta_for(K)
        phi = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(phi), np.sin(phi), color=GRID, lw=1.2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.text(1.42, 0.04, "|U⟩", fontsize=11, color=INK, ha="right")
        ax.text(0.04, 1.42, "|M⟩", fontsize=11, color=INK, va="top")
        # oracle reflection of the start state
        ax.annotate("", xy=(math.cos(-th), math.sin(-th)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6, ls="--"))
        ax.text(math.cos(-th) * 1.05, math.sin(-th) * 1.05 - 0.08, "O|s⟩ (−θ)", color=INK2, fontsize=8)
        shades = np.linspace(0.35, 1.0, tmax + 1)
        for t in range(tmax + 1):
            ang = (2 * t + 1) * th
            ax.annotate("", xy=(math.cos(ang), math.sin(ang)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2, alpha=shades[t]))
            ax.text(1.14 * math.cos(ang), 1.14 * math.sin(ang),
                    f"t={t}\n{np.degrees(ang):.0f}°, P={predicted_success(K, t):.2f}",
                    ha="center", va="center", fontsize=7.5, color=INK)
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"K={K}: θ={np.degrees(th):.1f}°, each iteration rotates by 2θ={np.degrees(2*th):.1f}°",
                     loc="left", fontsize=10)
    fig.suptitle("Grover as rotation in the |U⟩ (invalid) – |M⟩ (valid) plane;  P(t) = sin²((2t+1)θ)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ histograms
def _hist(ax, counts: Dict[str, int], pred: Callable, title: str):
    tot = sum(counts.values())
    freq = np.zeros(N_STATES)
    for k, v in counts.items():
        c = bitstring_to_config(k)
        freq[sum(c[n] << i for i, n in enumerate("PAW"))] = v / tot
    x = np.arange(N_STATES)
    ax.bar(x, freq, color=_bar_colors(pred), width=0.7, edgecolor=SURFACE, linewidth=2)
    for xi, f in zip(x, freq):
        if f > 0.004:
            ax.text(xi, f + 0.01, f"{f:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, _labels(), rotation=60, fontsize=7.5)
    ax.set_ylabel("measured frequency")
    ax.set_ylim(0, max(0.6, freq.max() * 1.15))
    ax.set_title(title, loc="left", fontsize=10)
    _legend_valid(ax)


def plot_histogram(counts, pred, title, path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _hist(ax, counts, pred, title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ablation(counts_ok, counts_bad, succ_ok, succ_bad, predicted, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [3, 3, 1.4]})
    _hist(axes[0], counts_ok, is_valid, "Correct oracle V1 (with uncompute), t=1")
    _hist(axes[1], counts_bad, is_valid, "BROKEN oracle (uncompute skipped), t=1")
    ax = axes[2]
    ax.bar([0, 1], [succ_ok, succ_bad], color=[BLUE, ORANGE], width=0.6, edgecolor=SURFACE, linewidth=2)
    ax.axhline(predicted, color=INK2, ls="--", lw=1.2)
    ax.text(1.45, predicted, f"predicted\n{predicted:.3f}", va="center", fontsize=8, color=INK2)
    for xi, v in enumerate([succ_ok, succ_bad]):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks([0, 1], ["correct", "broken"])
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylabel("P(valid)")
    ax.set_title("Success, 8192 shots", loc="left", fontsize=10)
    fig.suptitle("Uncomputation ablation: leftover garbage entangles the workspace with (P,A,W) and kills interference",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

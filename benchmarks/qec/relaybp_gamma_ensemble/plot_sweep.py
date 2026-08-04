# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Reproduces the four figures in the gamma-ensemble Relay-BP user guide
# (docs/sphinx/examples_rst/qec/nv_qldpc_gamma_ensemble_user_guide.rst) from
# report_data.npz (run_sweep.py). Writes them under <QEC_DATA_ROOT>/figures/
# with the published relaybp_* names, matching the copies in assets/docs/.
#   QEC_DATA_ROOT=<same dir as run_sweep.py> python3 plot_sweep.py
import os, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker

ROOT = os.environ.get("QEC_DATA_ROOT", "report_data")
Z = np.load(os.path.join(ROOT, "report_data.npz"), allow_pickle=True)
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
ORDER, NS = ["bb72", "bb144", "bb288"], [1, 2, 4, 8]
LAB = {s: str(Z[f"{s}__label"]) for s in ORDER}
NCOL = {
    1: "#2a78d6",
    2: "#eb6834",
    4: "#1baf7a",
    8: "#eda100"
}  # by ensemble size
CC = {"bb72": "#eb6834", "bb144": "#e34948", "bb288": "#7b3294"}  # by code
SURF, INK, MUTED, GRID = "#fcfcfb", "#1a1a19", "#6b6a66", "#e3e2dd"
plt.rcParams.update({
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "font.size": 10.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID
})


def despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)


def logN(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS)
    ax.set_xlim(0.8, NS[-1] * 1.7)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("ensemble size N")


# --- Fig 1: relaybp_gamma_ensemble_perf (iterations / time-per-iter / mean latency) ---
P = {
    s:
        dict(I=np.array([np.median(Z[f"{s}__N{N}__iter"]) for N in NS]),
             per=Z[f"{s}__periter"] / 1000.0,
             Tm=np.array([np.mean(Z[f"{s}__N{N}__wall"]) for N in NS]))
    for s in ORDER
}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.7))


def edgelabels(a, items):
    items = sorted(items, key=lambda e: e[0])
    prev, ly = -1e9, []
    for yend, _, _ in items:
        v = max(np.log10(yend), prev + 0.05)
        ly.append(v)
        prev = v
    for (_, r, col), v in zip(items, ly):
        a.text(NS[-1] * 1.13,
               10**v,
               f"{r:.2f}×",
               color=col,
               fontsize=8.5,
               fontweight="bold",
               va="center")


lab = [[], [], []]
for s in ORDER:
    p = P[s]
    ax[0].plot(NS,
               p["I"],
               "-o",
               color=CC[s],
               lw=2,
               ms=7,
               mec=SURF,
               mew=1.2,
               label=LAB[s])
    ax[1].plot(NS, p["per"], "-o", color=CC[s], ms=7, mec=SURF, mew=1.2)
    ax[2].plot(NS, p["Tm"], "-o", color=CC[s], lw=2, ms=7, mec=SURF, mew=1.2)
    for k, key in enumerate(("I", "per", "Tm")):
        lab[k].append((p[key][-1], p[key][-1] / p[key][0], CC[s]))
for k in range(3):
    edgelabels(ax[k], lab[k])
for a, t, yl in [(ax[0], "(A) iterations to converge", "median num_iter"),
                 (ax[1], "(B) time per iteration", "µs / iteration"),
                 (ax[2], "(C) mean latency", "mean decode latency (ms)")]:
    a.set_title(t, loc="left", fontweight="bold")
    a.set_ylabel(yl)
    a.set_yscale("log")
    logN(a)
    despine(a)
ax[0].legend(frameon=False, fontsize=8.5)
fig.text(0.5,
         0.005,
         f"Right-edge labels: value at N={NS[-1]} relative to N=1 (×).",
         ha="center",
         fontsize=8.5,
         color=MUTED)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig(f"{OUT}/relaybp_gamma_ensemble_perf.png",
            dpi=150,
            bbox_inches="tight",
            facecolor=SURF)
plt.close(fig)

# --- Fig 2: relaybp_latency_percentiles (p50/p90/p99/p99.9/p99.99 vs N) ---
QS = [("p50", 50), ("p90", 90), ("p99", 99), ("p99.9", 99.9), ("p99.99", 99.99)]
PC = {
    "p50": "#2a78d6",
    "p90": "#eb6834",
    "p99": "#1baf7a",
    "p99.9": "#7b3294",
    "p99.99": "#c0392b"
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
for ax, s in zip(axes, ORDER):
    for name, q in QS:
        ys = [np.percentile(Z[f"{s}__N{N}__wall"], q) for N in NS]
        ax.plot(NS,
                ys,
                "-o",
                color=PC[name],
                lw=2,
                ms=7,
                mec=SURF,
                mew=1.2,
                label=name)
        ax.annotate(f"{ys[-1] / ys[0]:.2f}×", (NS[-1], ys[-1]),
                    textcoords="offset points",
                    xytext=(7, 0),
                    color=PC[name],
                    fontsize=9,
                    fontweight="bold",
                    va="center")
    ax.set_yscale("log")
    ax.set_title(LAB[s], fontsize=11, fontweight="bold")
    logN(ax)
    despine(ax)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h,
           l,
           frameon=False,
           fontsize=10.5,
           ncol=len(QS),
           loc="upper center",
           bbox_to_anchor=(0.5, 1.0),
           columnspacing=1.4,
           handletextpad=0.5,
           title="percentile",
           title_fontproperties={
               "weight": "bold",
               "size": 10
           })
fig.supylabel("latency (ms)", fontsize=11)
fig.text(0.5,
         0.012,
         f"Right-edge labels: each percentile at N={NS[-1]} relative to N=1",
         ha="center",
         fontsize=9,
         color=MUTED)
fig.tight_layout(rect=[0, 0.05, 1, 0.90])
fig.savefig(f"{OUT}/relaybp_latency_percentiles.png", dpi=150, facecolor=SURF)
plt.close(fig)

# --- Fig 3: relaybp_ler_multiplier (LER_1(t)/LER_N(t) vs deadline) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, s in zip(axes, ORDER):
    w1, le1 = Z[f"{s}__N1__wall"], Z[f"{s}__N1__lerr"]
    sh = len(w1)
    allw = np.concatenate([Z[f"{s}__N{N}__wall"] for N in NS])
    grid = np.logspace(np.log10(allw.min() * .7), np.log10(allw.max() * 1.3),
                       200)
    ler1 = np.array([((w1 > t) | le1).mean() for t in grid])
    for N in [2, 4, 8]:
        w, le = Z[f"{s}__N{N}__wall"], Z[f"{s}__N{N}__lerr"]
        lerN = np.array([((w > t) | le).mean() for t in grid])
        ax.plot(grid,
                np.where(lerN * sh >= 10,
                         ler1 / np.where(lerN > 0, lerN, np.nan), np.nan),
                color=NCOL[N],
                lw=2.2,
                label=f"N = {N}")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(LAB[s], fontsize=11, fontweight="bold")
    ax.set_xlabel("hard deadline (ms)")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("LER$_{1}$(t) / LER$_{N}$(t)")
axes[0].legend(frameon=False, fontsize=9.5)
fig.suptitle("LER improvement over regular Relay BP (N=1) vs hard deadline",
             fontsize=13,
             fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/relaybp_ler_multiplier.png", dpi=150, facecolor=SURF)
plt.close(fig)

# --- Fig 4: relaybp_hard_deadline_ler  LER(t) = P(latency>t OR logical error) ---
fig, axes = plt.subplots(
    1, 3, figsize=(15, 8.2),
    sharey=False)  # per-panel y; height matches plot_deadline_ler.py
for ax, s in zip(axes, ORDER):
    allw = np.concatenate([Z[f"{s}__N{N}__wall"] for N in NS])
    grid = np.logspace(np.log10(allw.min() * .7), np.log10(allw.max() * 1.3),
                       160)
    for N in NS:
        w, le = Z[f"{s}__N{N}__wall"], Z[f"{s}__N{N}__lerr"]
        y = np.array([((w > t) | le).mean() for t in grid])
        ax.plot(grid,
                np.where(y > 0, y, np.nan),
                color=NCOL[N],
                lw=2.2,
                label=f"N = {N}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(LAB[s], fontsize=12.5, fontweight="bold")
    ax.set_xlabel("hard deadline (ms)")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("logical error rate  LER(t)")
axes[0].legend(frameon=False, fontsize=9.5, loc="upper right")
fig.suptitle(
    "Logical error rate vs hard deadline  —  LER(t) = P(latency > t or logical error)",
    fontsize=13,
    fontweight="bold")
fig.tight_layout(rect=[0.02, 0, 1, 0.97])
fig.savefig(f"{OUT}/relaybp_hard_deadline_ler.png", dpi=150, facecolor=SURF)
plt.close(fig)

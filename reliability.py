"""
Reliability diagrams for the forecast vs conditional climatology (clim_raw):
  • one panel per onset bin (week1..later)  — reliability separately by bin
  • one panel pooling all bins together     — reliability jointly across bins
  • a sharpness panel (forecast-probability histograms)

Forecast probabilities are binned into deciles; a point is drawn where a decile holds
≥ MIN_N samples.  Usage: python reliability.py --version 1p1|v2
"""
import argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config as C, common

MIN_N = 25
EDGES = np.linspace(0, 1, 11)
BIN_NAMES = {1:"week1", 2:"week2", 3:"week3", 4:"week4", 5:"later"}

def curve(p, o):
    p = np.array(p); o = np.array(o); idx = np.clip(np.digitize(p, EDGES) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= MIN_N: xs.append(p[m].mean()); ys.append(o[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), ns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=list(C.FORECAST_SETS),
                    help="AIFS version to score (see config.FORECAST_SETS)")
    a = ap.parse_args()
    P = {k: [] for k in range(1, 6)}; Cc = {k: [] for k in range(1, 6)}; O = {k: [] for k in range(1, 6)}
    for c in common.cases(C.FORECAST_SETS[a.version]):
        if c["earlier"]: continue                    # onset already occurred at init -> not scored
        for k in range(1, 6):
            if c["resolved"][k]:
                P[k].append(c["p"]["fc"][k]); Cc[k].append(c["p"]["clim"][k]); O[k].append(c["oo"][k])
    out_dir = C.OUTPUT_DIR / C.VERSION_DIR[a.version]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10)); axes = axes.ravel()
    # panels 0..4: per bin
    for ax, k in zip(axes, range(1, 6)):
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
        xf, yf, _ = curve(P[k], O[k]); xc, yc, _ = curve(Cc[k], O[k])
        if len(xf): ax.plot(xf, yf, "-o", color="#1f77b4", label="forecast", ms=4)
        if len(xc): ax.plot(xc, yc, "-s", color="#ff7f0e", label="clim_raw", ms=4, alpha=0.8)
        base = np.mean(O[k]) if O[k] else np.nan
        ax.axhline(base, color="#1f77b4", lw=0.5, ls=":", alpha=0.5)
        ax.set_title(f"{BIN_NAMES[k]}  (n={len(O[k])}, base={base:.2f})", fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("forecast probability")
        ax.set_ylabel("observed frequency"); ax.legend(loc="upper left", fontsize=8); ax.grid(lw=0.3, alpha=0.4)
    # panel 5: sharpness
    ax = axes[5]
    for k in range(1, 6): ax.hist(P[k], bins=EDGES, histtype="step", label=BIN_NAMES[k], lw=1.3)
    ax.set_title("AIFS forecast probability distribution (sharpness)", fontsize=11)
    ax.set_xlabel("forecast probability"); ax.set_ylabel("count"); ax.legend(fontsize=8); ax.set_yscale("log")
    fig.suptitle(f"Reliability by bin — {a.version} forecasts vs clim_raw", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_dir.mkdir(parents=True, exist_ok=True)
    o1 = out_dir / "reliability_by_bin.png"; fig.savefig(o1, dpi=150, bbox_inches="tight"); print("wrote", o1)

    # joint (all bins pooled)
    pf = np.concatenate([P[k] for k in range(1, 6)]); pc = np.concatenate([Cc[k] for k in range(1, 6)])
    oo = np.concatenate([O[k] for k in range(1, 6)])
    fig2, ax = plt.subplots(figsize=(7, 7)); ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    xf, yf, _ = curve(pf, oo); xc, yc, _ = curve(pc, oo)
    ax.plot(xf, yf, "-o", color="#1f77b4", label="forecast", ms=5)
    ax.plot(xc, yc, "-s", color="#ff7f0e", label="clim_raw", ms=5, alpha=0.8)
    ax.axhline(oo.mean(), color="#666", lw=0.6, ls=":", label=f"base rate {oo.mean():.2f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("forecast probability"); ax.set_ylabel("observed frequency")
    ax.set_title(f"Reliability, all bins pooled  (n={len(oo)})", fontsize=12); ax.legend(fontsize=9); ax.grid(lw=0.3, alpha=0.4)
    o2 = out_dir / "reliability_joint.png"; fig2.savefig(o2, dpi=150, bbox_inches="tight"); print("wrote", o2)

if __name__ == "__main__":
    main()

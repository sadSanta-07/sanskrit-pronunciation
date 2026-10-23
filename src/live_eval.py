import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("outputs")
CORRECT = "krishna (correct)"


def auc(correct, errors):
    c, e = np.asarray(correct), np.asarray(errors)
    return float((e[:, None] > c[None, :]).mean() + 0.5 * (e[:, None] == c[None, :]).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="outputs/live_log.csv")
    ap.add_argument("--ref_set", default="Laptop microphone")
    ap.add_argument("--thr", type=float, default=None, help="flag threshold (default: the logged one)")
    ap.add_argument("--sweep", action="store_true", help="show false alarms / detections per threshold")
    ap.add_argument("--dedupe", action="store_true", help="drop repeated saves of the same attempt")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    df = pd.read_csv(a.log)
    df = df[(df.ref_set == a.ref_set) & (df.said != "other")].copy()
    dup = df.duplicated(["said", "distance"], keep=False)
    if dup.any():
        print("Possible double-saves (same label and distance):")
        print(df[dup][["time", "said", "distance"]].to_string(index=False))
        if not a.dedupe:
            print("-> re-run with --dedupe to drop the repeats\n")
    if a.dedupe:
        df = df.drop_duplicates(["said", "distance"])

    thr = a.thr if a.thr is not None else float(df.iloc[-1].thr)
    df["flagged"] = df.distance > thr
    cor, err = df[df.said == CORRECT], df[df.said != CORRECT]

    g = df.groupby("said").agg(n=("distance", "size"), mean_dist=("distance", "mean"),
                               min_dist=("distance", "min"), max_dist=("distance", "max"),
                               flagged_pct=("flagged", lambda s: 100 * s.mean())).round(2)
    print(f"Log: {a.log}  |  threshold {thr:.2f}\n")
    print(g.to_string())

    if len(cor) and len(err):
        print(f"\nFalse alarms on correct takes : {100 * cor.flagged.mean():.0f}%  ({cor.flagged.sum()}/{len(cor)})")
        print(f"Errors detected (overall)     : {100 * err.flagged.mean():.0f}%  ({err.flagged.sum()}/{len(err)})")
        print(f"AUC, correct vs all errors    : {auc(cor.distance, err.distance):.2f}")
        for t in sorted(set(err.said)):
            e_t = err[err.said == t]
            print(f"  {t:<9} AUC {auc(cor.distance, e_t.distance):.2f}   detected {int(e_t.flagged.sum())}/{len(e_t)}")

    if a.sweep and len(cor) and len(err):
        print("\nThreshold sweep:")
        print(f"{'thr':>6} {'false alarms':>13} {'detected':>9}")
        for t in np.arange(cor.distance.min() - 0.05, err.distance.max() + 0.01, 0.05):
            print(f"{t:6.2f} {100 * (cor.distance > t).mean():12.0f}% {100 * (err.distance > t).mean():8.0f}%")

    cats = [CORRECT] + sorted(set(df.said) - {CORRECT})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, c in enumerate(cats):
        d = df[df.said == c].distance.to_numpy()
        xs = i + (np.arange(len(d)) - (len(d) - 1) / 2) * 0.09
        ax.scatter(xs, d, s=60, edgecolor="k", linewidth=0.6, zorder=3,
                   color="#2CA02C" if c == CORRECT else f"C{i}")
    if len(cor) > 2:
        m, s = cor.distance.mean(), cor.distance.std(ddof=1)
        ax.axhspan(m - s, m + s, color="#2CA02C", alpha=0.15, label="correct attempts: mean +/- 1 sd")
    ax.axhline(thr, color="red", ls="--", lw=1.2, label=f"flag threshold {thr:.2f}")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.replace(" (correct)", "\n(correct)") for c in cats])
    ax.set_ylabel("DTW distance (lower = closer)")
    ax.set_title(f"Live attempts, {a.ref_set.lower()} references {a.tag}".strip())
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / f"live_eval{('_' + a.tag) if a.tag else ''}.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved {out}")
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.align import align
from src.audio_io import load_clean
from src.features import extract

OUT = Path("outputs")
MIN_SIGMA_FRAC = 0.05      # floor on sigma, as a fraction of the baseline mean


def session_of(label):
    return "A" if label.endswith("_01") else "B"


def error_type(label):
    return label.rsplit("_", 1)[0]


def auc(correct, errors):
    c, e = np.asarray(correct), np.asarray(errors)
    return float((e[:, None] > c[None, :]).mean() + 0.5 * (e[:, None] == c[None, :]).mean())


def dist_matrix(feats, rows, refs):
    M = pd.DataFrame(index=rows, columns=refs, dtype=float)
    for a in rows:
        for b in refs:
            if a != b:
                M.loc[a, b] = align(feats[b], feats[a])["dist"]
    return M


def plot(df, mu, sigma, thr, out_path):
    cats = ["correct"] + sorted(c for c in df.type.unique() if c != "correct")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    panels = [("distance", "DTW distance to the correct takes (lower = closer)"),
              ("score", "Similarity score (%)")]
    for ax, (col, title) in zip(axes, panels):
        for i, c in enumerate(cats):
            d = df[df.type == c]
            xs = i + (np.arange(len(d)) - (len(d) - 1) / 2) * 0.09
            for sess, mk in (("A", "^"), ("B", "o")):
                m = (d.session == sess).to_numpy()
                ax.scatter(xs[m], d[col].to_numpy()[m], marker=mk, s=60,
                           color="#2CA02C" if c == "correct" else f"C{i}",
                           edgecolor="k", linewidth=0.6, zorder=3)
        if col == "distance":
            ax.axhspan(mu - sigma, mu + sigma, color="#2CA02C", alpha=0.15,
                       label="correct takes: mean +/- 1 sd")
            ax.axhline(thr, color="red", ls="--", lw=1.2, label="flag threshold")
            ax.legend(fontsize=8, loc="upper left")
        else:
            ax.set_ylim(0, 105)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.text(0.99, 0.01, "triangle = session A (original recordings), circle = session B (memo takes)",
             ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="krishna")
    ap.add_argument("--session", choices=["all", "B"], default="all")
    ap.add_argument("--k", type=float, default=2.0, help="flag threshold = mean + k * sigma")
    ap.add_argument("--half_sigmas", type=float, default=3.0,
                    help="sigmas beyond the mean at which the score drops to 50%%")
    a = ap.parse_args()

    cat = pd.read_csv(OUT / "catalog.csv")
    cat = cat[cat.word == a.word]
    if a.session == "B":
        cat = cat[~cat.label.str.endswith("_01")]
    feats = {r.label: extract(load_clean(r.wav)[0]) for r in cat.itertuples()}
    refs = sorted(cat[cat.role == "reference"].label)
    studs = sorted(cat[cat.role == "student"].label)

    M = dist_matrix(feats, refs + studs, refs)
    d = M.median(axis=1, skipna=True)          # typical distance to the correct takes (self excluded)

    base = d[refs]
    mu = float(base.mean())
    sigma = max(float(base.std(ddof=1)), MIN_SIGMA_FRAC * mu)
    thr = mu + a.k * sigma

    def score(x):
        return 100 * 0.5 ** (max(0.0, x - mu) / (a.half_sigmas * sigma))

    df = pd.DataFrame([dict(label=l, type="correct" if l in refs else error_type(l),
                            session=session_of(l), distance=round(float(d[l]), 3),
                            score=round(score(d[l]), 1), flagged=bool(d[l] > thr))
                       for l in refs + studs])
    tag = f"{a.word}_{a.session}"
    df.to_csv(OUT / f"scores_{tag}.csv", index=False)

    print(f"Baseline (leave-one-out over {len(refs)} correct takes): "
          f"mean {mu:.3f}, sd {sigma:.3f}, flag threshold {thr:.3f}\n")
    summ = df.groupby("type").agg(n=("label", "size"), mean_dist=("distance", "mean"),
                                  mean_score=("score", "mean"), min_score=("score", "min"),
                                  max_score=("score", "max"),
                                  flagged_pct=("flagged", lambda s: 100 * s.mean())).round(2)
    print(summ.to_string(), "\n")

    cd = df[df.type == "correct"].distance
    ed = df[df.type != "correct"].distance
    print(f"False alarms on correct takes : {100 * df[df.type == 'correct'].flagged.mean():.0f}%")
    print(f"Errors detected (overall)     : {100 * df[df.type != 'correct'].flagged.mean():.0f}%")
    print(f"AUC, correct vs all errors    : {auc(cd, ed):.2f}")
    for t in sorted(set(df.type) - {"correct"}):
        print(f"  AUC vs {t:<9}: {auc(cd, df[df.type == t].distance):.2f}")

    print("\nMean distance by type and session (A = original, B = memo takes):")
    print(df.groupby(["type", "session"]).distance.mean().unstack().round(2).to_string())

    A = [l for l in refs if session_of(l) == "A"]
    B = [l for l in refs if session_of(l) == "B"]
    if A and len(B) > 1:
        within = np.nanmean(M.loc[B, B].to_numpy())
        cross = np.nanmean(M.loc[A, B].to_numpy())
        print(f"\nCorrect takes: distance within session B = {within:.2f}, "
              f"session A vs B = {cross:.2f}")

    plot(df, mu, sigma, thr, OUT / f"score_{tag}.png")
    print(f"\nSaved scores_{tag}.csv and score_{tag}.png in outputs/")
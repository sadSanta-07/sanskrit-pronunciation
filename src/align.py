import argparse
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.audio_io import SR, load_clean
from src.features import HOP, extract, smooth

OUT = Path("outputs")


def align(ref, stu):
    """DTW between two Features objects."""
    D, wp = librosa.sequence.dtw(X=ref.mfcc.T, Y=stu.mfcc.T, metric="euclidean")
    wp = wp[::-1]                                   # chronological; rows = (ref_idx, stu_idx)
    local = np.linalg.norm(ref.mfcc[wp[:, 0]] - stu.mfcc[wp[:, 1]], axis=1)
    per_ref = (pd.Series(local).groupby(wp[:, 0]).mean()
               .reindex(range(len(ref.mfcc))).ffill().to_numpy())
    warp = float(np.mean(np.abs(wp[:, 0] / len(ref.mfcc) - wp[:, 1] / len(stu.mfcc))))
    return dict(path=wp, local=local, per_ref=per_ref, dist=float(local.mean()), warp=warp)


def plot_alignment(ref_name, feats, results, out_path):
    names = list(results)
    ref = feats[ref_name]
    cols, rows = 2, (len(names) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.6 * rows), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax, nm in zip(axes.flat, names):
        st, res = feats[nm], results[nm]
        ax.imshow(cdist(ref.mfcc, st.mfcc), origin="lower", aspect="auto", cmap="viridis",
                  extent=[0, st.duration, 0, ref.duration])
        ax.plot(res["path"][:, 1] * HOP / SR, res["path"][:, 0] * HOP / SR, color="white", lw=1.8)
        ax.plot([0, st.duration], [0, ref.duration], "w--", lw=0.8, alpha=0.6)
        ax.set_title(f"{nm}   (distance {res['dist']:.2f})", fontsize=10)
        ax.set_xlabel(f"{nm} time (s)")
        ax.set_ylabel(f"{ref_name} time (s)")
    fig.suptitle("DTW alignment (white = warping path, dashed = no time-warping)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_strips(ref_name, ref_wav, feats, results, out_path):
    names = sorted(results, key=lambda n: results[n]["dist"])
    ref = feats[ref_name]
    y, _ = load_clean(ref_wav)
    S = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=512, hop_length=HOP, n_mels=80, fmax=8000), ref=np.max)
    vmax = max(smooth(results[n]["per_ref"]).max() for n in names)

    fig = plt.figure(figsize=(11, 3.4 + 0.8 * len(names)))
    gs = fig.add_gridspec(len(names) + 1, 1, height_ratios=[3] + [1] * len(names), hspace=0.2)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(S, origin="lower", aspect="auto", cmap="magma",
               extent=[0, ref.duration, 0, 8000], vmin=-70, vmax=0)
    ax0.set_ylabel("Hz")
    ax0.set_title(f"Where each recording differs from the reference '{ref_name}' "
                  f"(darker red = bigger difference)")
    ax0.set_xlim(0, ref.duration)
    ax0.tick_params(labelbottom=False)
    for k, nm in enumerate(names, 1):
        ax = fig.add_subplot(gs[k], sharex=ax0)
        ax.imshow(smooth(results[nm]["per_ref"])[None, :], aspect="auto", cmap="Reds",
                  vmin=0, vmax=vmax, extent=[0, ref.duration, 0, 1])
        ax.set_yticks([])
        ax.set_ylabel(f"{nm}\nd={results[nm]['dist']:.2f}", rotation=0, ha="right",
                      va="center", fontsize=9)
        if k < len(names):
            ax.tick_params(labelbottom=False)
    ax.set_xlabel("Reference time (s)")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="krishna")
    ap.add_argument("--ref", default=None, help="reference label used for the plots")
    a = ap.parse_args()

    cat = pd.read_csv(OUT / "catalog.csv")
    cat = cat[cat.word == a.word]
    wavs = dict(zip(cat.label, cat.wav))
    feats = {lab: extract(load_clean(p)[0]) for lab, p in wavs.items()}
    refs = sorted(cat[cat.role == "reference"].label)
    studs = sorted(cat[cat.role == "student"].label)
    primary = a.ref or refs[0]

    rows, plot_results = [], {}
    for s in studs:
        row, best = dict(student=s), None
        for r in refs:
            res = align(feats[r], feats[s])
            row[f"d_{r}"] = round(res["dist"], 3)
            if best is None or res["dist"] < best[1]["dist"]:
                best = (r, res)
            if r == primary:
                plot_results[s] = res
        row.update(best_ref=best[0], distance=round(best[1]["dist"], 3),
                   warp=round(best[1]["warp"], 3),
                   dur_ratio=round(feats[s].duration / feats[best[0]].duration, 2))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("distance")
    df.to_csv(OUT / f"distances_{a.word}.csv", index=False)
    print(df.to_string(index=False))

    plot_alignment(primary, feats, plot_results, OUT / f"alignment_{a.word}.png")
    plot_strips(primary, wavs[primary], feats, plot_results, OUT / f"mismatch_{a.word}.png")
    print(f"\nSaved alignment_{a.word}.png and mismatch_{a.word}.png in outputs/")
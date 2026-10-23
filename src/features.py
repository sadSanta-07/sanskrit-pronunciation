import argparse
from dataclasses import dataclass
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.audio_io import SR, load_clean

N_FFT, HOP = 512, 160            # 32 ms window, 10 ms hop at 16 kHz
N_MELS, N_MFCC = 40, 13
OUT = Path("outputs")
COLORS = ["#4C78A8", "#F58518", "#E45756", "#B279A2", "#72B7B2"]


@dataclass
class Features:
    mfcc: np.ndarray     # (T, 39): 13 static + 13 delta + 13 delta-delta, CMVN-normalized
    spec: dict           # name -> (T,) arrays
    times: np.ndarray    # (T,) seconds
    duration: float


def extract(y) -> Features:
    ye = librosa.effects.preemphasis(y, coef=0.97)
    mel = librosa.feature.melspectrogram(y=ye, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS, fmax=8000)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC + 1)[1:]
    mfcc = (mfcc - mfcc.mean(1, keepdims=True)) / (mfcc.std(1, keepdims=True) + 1e-8)
    d1 = librosa.feature.delta(mfcc, width=9, order=1, mode="nearest")
    d2 = librosa.feature.delta(mfcc, width=9, order=2, mode="nearest")

    kw = dict(n_fft=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    spec = dict(
        rms_db=librosa.amplitude_to_db(rms, ref=np.max),
        centroid=librosa.feature.spectral_centroid(y=y, sr=SR, **kw)[0],
        rolloff=librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.85, **kw)[0],
        bandwidth=librosa.feature.spectral_bandwidth(y=y, sr=SR, **kw)[0],
        flatness=librosa.feature.spectral_flatness(y=y, **kw)[0],
        zcr=librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0],
    )
    T = min(mfcc.shape[1], *(len(v) for v in spec.values()))
    return Features(
        mfcc=np.vstack([mfcc, d1, d2])[:, :T].T,
        spec={k: v[:T] for k, v in spec.items()},
        times=np.arange(T) * HOP / SR,
        duration=len(y) / SR,
    )


def summarize(f: Features) -> dict:
    active = f.spec["rms_db"] > -30          # ignore near-silent frames
    row = dict(duration_s=f.duration, active_s=active.sum() * HOP / SR)
    for k in ("centroid", "rolloff", "bandwidth", "flatness", "zcr"):
        row[f"{k}_mean"] = float(np.mean(f.spec[k][active]))
    return row


def smooth(x, n=5):
    return np.convolve(x, np.ones(n) / n, mode="same")


def plot_mfcc(feats, names, out_path):
    tmax = max(f.duration for f in feats) * 1.05
    fig, axes = plt.subplots(len(feats), 1, figsize=(11, 1.9 * len(feats)),
                             sharex=True, squeeze=False)
    for ax, f, name in zip(axes[:, 0], feats, names):
        ax.imshow(f.mfcc[:, :N_MFCC].T, origin="lower", aspect="auto", cmap="coolwarm",
                  vmin=-2.5, vmax=2.5, extent=[0, f.duration, 1, N_MFCC])
        ax.set_xlim(0, tmax)
        ax.set_ylabel(name, fontsize=9)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[0, 0].set_title("MFCC coefficients 1-13 over time (CMVN-normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_spectral(feats, names, out_path):
    panels = [("rms_db", "Energy (dB)"), ("centroid", "Spectral centroid (Hz)"),
              ("flatness", "Spectral flatness"), ("zcr", "Zero-crossing rate")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.3 * len(panels)), sharex=True)
    for ax, (key, title) in zip(axes, panels):
        for i, (f, name) in enumerate(zip(feats, names)):
            is_ref = name.startswith("ref")
            ax.plot(f.times, smooth(f.spec[key]), label=name,
                    color="#2CA02C" if is_ref else COLORS[i % len(COLORS)],
                    lw=2.8 if is_ref else 1.3)
        ax.set_ylabel(title, fontsize=9)
    axes[0].legend(ncol=len(feats), fontsize=8, loc="lower center")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="krishna")
    a = ap.parse_args()

    cat = pd.read_csv(OUT / "catalog.csv")
    cat = cat[cat.word == a.word].sort_values(["role", "label"]).reset_index(drop=True)
    feats = [extract(load_clean(p)[0]) for p in cat.wav]
    names = cat.label.tolist()

    rows = []
    for (r, f) in zip(cat.itertuples(), feats):
        rows.append(dict(label=r.label, role=r.role, frames=len(f.times), **summarize(f)))
    summ = pd.DataFrame(rows).round(3)
    summ.to_csv(OUT / f"feature_summary_{a.word}.csv", index=False)
    print(summ.to_string(index=False))

    plot_mfcc(feats, names, OUT / f"mfcc_{a.word}.png")
    plot_spectral(feats, names, OUT / f"spectral_{a.word}.png")
    print(f"\nFeature dims per frame: {feats[0].mfcc.shape[1]} MFCC-based + "
          f"{len(feats[0].spec)} spectral")
    print(f"Saved outputs/mfcc_{a.word}.png and outputs/spectral_{a.word}.png")
import argparse
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.audio_io import AUDIO_EXT, SR, load_clean, to_wav

DATA, WAV, OUT = Path("data"), Path("data/wav"), Path("outputs")


def build_catalog():
    rows = []
    for role in ("reference", "student"):
        base = DATA / role
        if not base.exists():
            continue
        for f in sorted(base.rglob("*")):
            if f.suffix.lower() not in AUDIO_EXT:
                continue
            word, label = f.parent.name.lower(), f.stem.lower()
            wav = WAV / role / word / f"{label}.wav"
            if not wav.exists() or wav.stat().st_mtime < f.stat().st_mtime:
                to_wav(f, wav)
            clean, raw, info = load_clean(wav, return_info=True)
            rows.append(dict(role=role, word=word, label=label, wav=str(wav),
                             raw_sec=round(len(raw) / SR, 2),
                             clean_sec=round(len(clean) / SR, 2),
                             speech_start=info["start_s"], speech_end=info["end_s"],
                             floor_db=info["floor_db"]))
    return pd.DataFrame(rows)


def plot_overview(df, word, out_path):
    d = df[df.word == word].sort_values(["role", "label"]).reset_index(drop=True)
    n, tmax = len(d), d.clean_sec.max() * 1.05
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.2 * n), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.4]})
    for (ax_w, ax_s), r in zip(axes, d.itertuples()):
        y, _ = load_clean(r.wav)
        t = np.arange(len(y)) / SR
        color = "#2CA02C" if r.role == "reference" else "#4C78A8"
        ax_w.plot(t, y, color=color, lw=0.6)
        ax_w.set_xlim(0, tmax)
        ax_w.set_ylim(-1, 1)
        ax_w.set_ylabel(f"{r.label}\n({r.role})", fontsize=9)
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=512, hop_length=160,
                                           n_mels=80, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        ax_s.imshow(S_db, origin="lower", aspect="auto", cmap="magma",
                    extent=[0, len(y) / SR, 0, 8000], vmin=-70, vmax=0)
        ax_s.set_xlim(0, tmax)
        ax_s.set_ylabel("Hz")
        ax_s.set_title(f"{r.clean_sec:.2f} s", fontsize=9, loc="right")
    axes[-1][0].set_xlabel("Time (s)")
    axes[-1][1].set_xlabel("Time (s)")
    axes[0][0].set_title("Waveform")
    fig.suptitle(f"Recordings of '{word}' (same time axis)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="krishna")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    df = build_catalog()
    if df.empty:
        raise SystemExit("No audio found under data/reference or data/student")
    df.to_csv(OUT / "catalog.csv", index=False)
    print(df[["role", "label", "raw_sec", "clean_sec", "speech_start", "speech_end", "floor_db"]].to_string(index=False))
    plot_overview(df, a.word, OUT / f"overview_{a.word}.png")
    print(f"\nSaved outputs/overview_{a.word}.png")
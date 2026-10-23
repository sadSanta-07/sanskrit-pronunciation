import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from src.audio_io import SR, to_wav

HOP, FRAME, PAD = 160, 400, 0.3


def find_takes(y, min_gap=0.4, min_len=0.25):
    sos = butter(4, [150, 4000], btype="bandpass", fs=SR, output="sos")
    rms = librosa.feature.rms(y=sosfiltfilt(sos, y), frame_length=FRAME, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-8)
    db -= db.max()
    thr = min(np.percentile(db, 10) + 15, -20.0)
    idx = np.where(db > thr)[0]
    if len(idx) == 0:
        return []
    segs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if (i - prev) * HOP / SR > min_gap:
            segs.append((start, prev))
            start = i
        prev = i
    segs.append((start, prev))
    return [(s * HOP / SR, e * HOP / SR + FRAME / SR) for s, e in segs
            if (e - s) * HOP / SR >= min_len]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--role", choices=["reference", "student"], required=True)
    ap.add_argument("--word", default="krishna")
    ap.add_argument("--label", required=True, help="e.g. ref, kisna, krishn")
    ap.add_argument("--start", type=int, default=2, help="first take number")
    ap.add_argument("--expect", type=int, default=None, help="how many takes you said")
    ap.add_argument("--min_gap", type=float, default=0.4, help="silence (s) that separates takes")
    a = ap.parse_args()

    tmp = Path("data/wav/_tmp") / (Path(a.audio).stem + ".wav")
    to_wav(a.audio, tmp)
    y, _ = librosa.load(str(tmp), sr=SR, mono=True)
    takes = find_takes(y, a.min_gap)
    print(f"Found {len(takes)} takes")
    if a.expect and len(takes) != a.expect:
        print(f"WARNING: expected {a.expect}. Too many found -> raise --min_gap; "
              f"too few -> lower it.")

    out_dir = Path("data") / a.role / a.word.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, (t0, t1) in enumerate(takes, a.start):
        seg = y[max(0, int((t0 - PAD) * SR)): int((t1 + PAD) * SR)]
        out = out_dir / f"{a.label}_{n:02d}.wav"
        sf.write(out, seg, SR, subtype="PCM_16")
        print(f"  {out.name}: speech {t0:.2f}-{t1:.2f}s ({t1 - t0:.2f}s)")
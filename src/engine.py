from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.align import align
from src.audio_io import SR, load_clean
from src.features import HOP, extract
from src.feedback import build_baseline, describe, excess, find_window, medoid, window_stats

CATALOG = Path("outputs/catalog.csv")
MIN_SIGMA_FRAC = 0.05


class Assessor:
    def __init__(self, word="krishna", ref_dir=None, k=2.0, half_sigmas=3.0):
        cat = pd.read_csv(CATALOG)
        self.cat = cat[cat.word == word]
        self.word, self.k, self.half = word, k, half_sigmas
        self.wavs = dict(zip(self.cat.label, self.cat.wav))
        if ref_dir is None:
            self.refs = sorted(self.cat[self.cat.role == "reference"].label)
        else:
            paths = sorted(Path(ref_dir).glob("*.wav"))
            self.refs = [p.stem for p in paths]
            self.wavs.update({p.stem: str(p) for p in paths})
        if len(self.refs) < 4:
            raise ValueError("Need at least 4 correct takes in data/reference/<word>/")
        self.feats = {l: extract(load_clean(self.wavs[l])[0]) for l in self.refs}
        self.primary = medoid(self.feats, self.refs)
        self.others = [r for r in self.refs if r != self.primary]
        self.base = build_baseline(self.feats, self.primary, self.others)

        # typical distance of a correct take to the other correct takes (leave-one-out)
        d = [np.median([align(self.feats[r], self.feats[o])["dist"]
                        for o in self.refs if o != r]) for r in self.refs]
        self.ref_dists = dict(zip(self.refs, d))
        self.mu = float(np.mean(d))
        self.sigma = max(float(np.std(d, ddof=1)), MIN_SIGMA_FRAC * self.mu)
        self.dist_thr = self.mu + k * self.sigma

        # how high can the per-frame difference get for a correct take held out?
        self.loo_peaks = []
        for r in self.others:
            b = build_baseline(self.feats, self.primary, [o for o in self.others if o != r])
            self.loo_peaks.append(float(excess(self.base["results"][r], b).max()))
        self.excess_thr = max(self.loo_peaks)

    def score(self, d):
        return 100 * 0.5 ** (max(0.0, d - self.mu) / (self.half * self.sigma))

    def assess(self, y):
        f = extract(y)
        d = float(np.median([align(self.feats[r], f)["dist"] for r in self.refs]))
        res = align(self.feats[self.primary], f)
        e = excess(res, self.base)
        out = dict(distance=d, score=self.score(d), matches=d <= self.dist_thr,
                   excess=e, peak=float(e.max()), localized=float(e.max()) > self.excess_thr,
                   window=None, where=None, cues=[])
        if out["localized"]:
            i0, i1 = find_window(e)
            bs = pd.DataFrame([window_stats(self.feats[o], self.base["results"][o], i0, i1)
                               for o in self.others]).mean()
            ts = pd.Series(window_stats(f, res, i0, i1))
            pos = (i0 + i1) / 2 / len(e)
            out.update(window=(i0 * HOP / SR, (i1 + 1) * HOP / SR),
                       where="beginning" if pos < 0.33 else "middle" if pos < 0.67 else "end",
                       cues=describe(ts, bs))
        return out


def make_figure(a, y, r):
    ref_y, _ = load_clean(a.wavs[a.primary])
    ref_dur, you_dur = len(ref_y) / SR, len(y) / SR

    def spec(sig):
        S = librosa.feature.melspectrogram(y=sig, sr=SR, n_fft=512, hop_length=HOP,
                                           n_mels=80, fmax=8000)
        return librosa.power_to_db(S, ref=np.max)

    fig = Figure(figsize=(10, 7.4))
    ax1, ax2, ax3 = fig.subplots(3, 1, gridspec_kw={"height_ratios": [3, 3, 1.1]})
    ax1.imshow(spec(y), origin="lower", aspect="auto", cmap="magma",
               extent=[0, you_dur, 0, 8000], vmin=-70, vmax=0)
    ax1.set_title("Your recording", fontsize=10)
    ax1.set_ylabel("Hz")
    ax2.imshow(spec(ref_y), origin="lower", aspect="auto", cmap="magma",
               extent=[0, ref_dur, 0, 8000], vmin=-70, vmax=0)
    ax2.set_title(f"Reference ({a.primary}, the most typical correct take)", fontsize=10)
    ax2.set_ylabel("Hz")
    e = r["excess"]
    ax3.imshow(np.clip(e, 0, 4)[None, :], aspect="auto", cmap="Reds", vmin=0, vmax=4,
               extent=[0, len(e) * HOP / SR, 0, 1])
    ax3.set_yticks([])
    ax3.set_xlabel("Time in the reference (s)")
    ax3.set_title("Difference beyond normal variation (red)", fontsize=9)
    for ax in (ax2, ax3):
        ax.set_xlim(0, ref_dur)
    if r["window"]:
        ax3.axvspan(*r["window"], fill=False, edgecolor="k", lw=1.8)
        ax2.axvspan(*r["window"], fill=False, edgecolor="white", lw=1.8)
    fig.tight_layout()
    return fig
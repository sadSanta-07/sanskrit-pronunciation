import argparse
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.align import align
from src.audio_io import SR, load_clean
from src.features import HOP, extract

OUT = Path("outputs")


def sm(x, n=7):
    p = n // 2
    return np.convolve(np.pad(np.asarray(x, float), p, mode="edge"), np.ones(n) / n, mode="valid")


def error_type(label):
    return label.rsplit("_", 1)[0]


def medoid(feats, refs):
    med = {a: np.median([align(feats[a], feats[b])["dist"] for b in refs if b != a]) for a in refs}
    return min(med, key=med.get)


def build_baseline(feats, primary, others):
    results = {o: align(feats[primary], feats[o]) for o in others}
    curves = np.array([sm(r["per_ref"]) for r in results.values()])
    m = curves.mean(axis=0)
    s = max(float(np.sqrt(((curves - m) ** 2).mean())), 0.05 * float(m.mean()))
    return dict(m=m, s=s, results=results)


def excess(res, base):
    return (sm(res["per_ref"]) - base["m"]) / base["s"]


def find_window(e, min_len=6):
    pk = int(np.argmax(e))
    thr = max(1.0, 0.5 * e[pk])
    i0 = i1 = pk
    while i0 > 0 and e[i0 - 1] > thr:
        i0 -= 1
    while i1 < len(e) - 1 and e[i1 + 1] > thr:
        i1 += 1
    while i1 - i0 + 1 < min_len and not (i0 == 0 and i1 == len(e) - 1):
        i0, i1 = max(0, i0 - 1), min(len(e) - 1, i1 + 1)
    return i0, i1


def window_stats(f, res, i0, i1):
    wp = res["path"]
    idx = np.unique(wp[(wp[:, 0] >= i0) & (wp[:, 0] <= i1), 1])
    idx = idx[idx < len(f.times)]
    return dict(dur=len(idx) / (i1 - i0 + 1),
                centroid=float(f.spec["centroid"][idx].mean()),
                zcr=float(f.spec["zcr"][idx].mean()),
                rms=float(f.spec["rms_db"][idx].mean()))


def describe(ts, bs):
    cues = []
    c = 100 * (ts.centroid / bs.centroid - 1)
    if abs(c) >= 10:
        cues.append(f"spectrum {'brighter' if c > 0 else 'darker'} ({c:+.0f}% centroid)")
    z = 100 * (ts.zcr / bs.zcr - 1)
    if abs(z) >= 20:
        cues.append(f"{'more' if z > 0 else 'less'} noise-like ({z:+.0f}% zero-crossings)")
    e = ts.rms - bs.rms
    if abs(e) >= 3:
        cues.append(f"{'louder' if e > 0 else 'quieter'} ({e:+.1f} dB)")
    d = 100 * (ts.dur / bs.dur - 1)
    if abs(d) >= 20:
        cues.append(f"{'longer' if d > 0 else 'shorter'} ({d:+.0f}% duration)")
    return cues or ["no single cue stands out; the change is in the overall spectral shape (MFCC)"]


def plot(wav, rows, out_path):
    y, _ = load_clean(wav)
    S = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=512, hop_length=HOP, n_mels=80, fmax=8000), ref=np.max)
    dur = len(y) / SR
    fig = plt.figure(figsize=(11, 3.2 + 0.85 * len(rows)))
    gs = fig.add_gridspec(len(rows) + 1, 1, height_ratios=[3] + [1] * len(rows), hspace=0.25)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(S, origin="lower", aspect="auto", cmap="magma",
               extent=[0, dur, 0, 8000], vmin=-70, vmax=0)
    ax0.set_xlim(0, dur)
    ax0.set_ylabel("Hz")
    ax0.tick_params(labelbottom=False)
    ax0.set_title("Differences beyond normal variation (red = standard deviations above the correct takes)")
    ax = ax0
    for k, (name, e, win) in enumerate(rows, 1):
        ax = fig.add_subplot(gs[k], sharex=ax0)
        ax.imshow(np.clip(e, 0, 4)[None, :], aspect="auto", cmap="Reds", vmin=0, vmax=4,
                  extent=[0, len(e) * HOP / SR, 0, 1])
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        if win:
            ax.axvspan(win[0], win[1], fill=False, edgecolor="k", lw=1.6)
        if k < len(rows):
            ax.tick_params(labelbottom=False)
    ax.set_xlabel("Time in the reference recording (s)")
    ax.set_xlim(0, dur)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="krishna")
    a = ap.parse_args()

    cat = pd.read_csv(OUT / "catalog.csv")
    cat = cat[cat.word == a.word]
    wavs = dict(zip(cat.label, cat.wav))
    feats = {l: extract(load_clean(p)[0]) for l, p in wavs.items()}
    refs = sorted(cat[cat.role == "reference"].label)
    studs = sorted(cat[cat.role == "student"].label)
    if len(refs) < 4:
        raise SystemExit("Need at least 4 correct takes (baseline + one held-out control)")

    primary = medoid(feats, refs)
    others = [r for r in refs if r != primary]
    control = others[-1]
    base = build_baseline(feats, primary, others)
    base_ctl = build_baseline(feats, primary, [o for o in others if o != control])
    T = len(base["m"])
    print(f"Primary reference (most typical correct take): {primary}")
    print(f"Baseline from {len(others)} other correct takes; control row = {control} held out\n")

    ctl_e = excess(base["results"][control], base_ctl)
    rows, fb = [(f"{control}\n(correct, held out)", ctl_e, None)], []
    for t in sorted({error_type(s) for s in studs}):
        takes = [s for s in studs if error_type(s) == t]
        res = {s: align(feats[primary], feats[s]) for s in takes}
        es = {s: excess(res[s], base) for s in takes}
        e_t = np.mean(list(es.values()), axis=0)
        i0, i1 = find_window(e_t)
        inside = sum(i0 <= int(np.argmax(e)) <= i1 for e in es.values())
        bs = pd.DataFrame([window_stats(feats[o], base["results"][o], i0, i1) for o in others]).mean()
        ts = pd.DataFrame([window_stats(feats[s], res[s], i0, i1) for s in takes]).mean()
        t0, t1 = i0 * HOP / SR, (i1 + 1) * HOP / SR
        pos = (i0 + i1) / 2 / T
        where = "beginning" if pos < 0.33 else "middle" if pos < 0.67 else "end"
        cues = describe(ts, bs)
        fb.append(dict(type=t, start_s=round(t0, 2), end_s=round(t1, 2), where=where,
                       peak_sd=round(float(e_t.max()), 1),
                       takes_peaking_inside=f"{inside}/{len(takes)}", cues="; ".join(cues)))
        rows.append((t, e_t, (t0, t1)))
        print(f"{t:<9} {t0:.2f}-{t1:.2f} s ({where} of the word), peak {e_t.max():.1f} sd, "
              f"{inside}/{len(takes)} takes peak inside the window\n"
              f"          -> {'; '.join(cues)}")

    print(f"\nControl (correct take held out): max {ctl_e.max():.1f} sd")
    pd.DataFrame(fb).to_csv(OUT / f"feedback_{a.word}.csv", index=False)
    plot(wavs[primary], rows, OUT / f"feedback_{a.word}.png")
    print(f"Saved feedback_{a.word}.png and feedback_{a.word}.csv in outputs/")
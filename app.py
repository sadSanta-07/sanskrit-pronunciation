from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
import streamlit as st

from src.audio_io import SR, load_clean, to_wav
from src.engine import Assessor, make_figure
from src.split_takes import PAD, find_takes
from datetime import datetime

st.set_page_config(page_title="Sanskrit Pronunciation Assessment", layout="wide")
TMP = Path("data/wav/_tmp")
LAPTOP_DIR = Path("data/wav/ref_laptop")


@st.cache_resource(show_spinner="Building the baseline from the correct recordings...")
def get_assessor(word, ref_dir):
    return Assessor(word, ref_dir)


def to_16k(up):
    TMP.mkdir(parents=True, exist_ok=True)
    src = TMP / ("upload_raw" + (Path(up.name).suffix or ".wav"))
    src.write_bytes(up.getvalue())
    return to_wav(src, TMP / "upload_16k.wav")


def clean_from_upload(up):
    wav = to_16k(up)
    return load_clean(wav)[0], wav


def n_laptop_refs(word):
    return len(list((LAPTOP_DIR / word).glob("*.wav")))

def dev_correct_distances():
    p = Path("outputs/live_log_dev.csv")
    if not p.exists():
        return []
    d = pd.read_csv(p).drop_duplicates(["said", "distance"])
    return d[(d.ref_set == "Laptop microphone") & (d.said == "krishna (correct)")].distance.tolist()

catalog = pd.read_csv("outputs/catalog.csv")
words = sorted(catalog[catalog.role == "reference"].word.unique())

st.title("Sanskrit Pronunciation Assessment")
st.caption("Using MFCC and spectral features, with DTW alignment against correct recordings")

word = st.sidebar.selectbox("Word", words)
opts = ["Phone recordings"] + (["Laptop microphone"] if n_laptop_refs(word) >= 4 else [])
ref_set = st.sidebar.radio("Compare against", opts, index=len(opts) - 1)
st.sidebar.caption("Pick the set recorded on the same device you are speaking into.")
ref_dir = str(LAPTOP_DIR / word) if ref_set == "Laptop microphone" else None

A = get_assessor(word, ref_dir)
dev = dev_correct_distances() if ref_set == "Laptop microphone" else []
center = float(pd.Series(dev).mean()) if len(dev) >= 5 else A.mu
default_thr = float(max(dev)) + 0.01 if len(dev) >= 5 else A.dist_thr
st.sidebar.markdown(
    f"**Baseline:** {len(A.refs)} correct takes  \n"
    f"typical distance of a correct attempt: {center:.2f}  \n"
    f"reference take: {A.primary}")
thr = st.sidebar.number_input("Flag threshold (distance)", value=round(default_thr, 2),
                              step=0.01, key=f"thr_{ref_set}")
with st.sidebar.expander("Per-take distances (spot outliers)"):
    st.dataframe(pd.DataFrame({"take": list(A.ref_dists),
                               "typical distance": [round(v, 2) for v in A.ref_dists.values()]}),
                 hide_index=True)

tab_assess, tab_cal = st.tabs(["Assess", "Calibrate laptop microphone"])

with tab_assess:
    mode = st.radio("Input", ["Record with microphone", "Upload a recording", "Recorded examples"],
                    horizontal=True)
    y = wav = None
    if mode == "Record with microphone":
        up = st.audio_input("Say the word")
        if up is not None:
            y, wav = clean_from_upload(up)
    elif mode == "Upload a recording":
        up = st.file_uploader("Audio file", type=["m4a", "wav", "mp3", "aac", "ogg"])
        if up is not None:
            y, wav = clean_from_upload(up)
    else:
        st.caption("These examples were recorded on the phone, so compare them against 'Phone recordings'.")
        pick = st.selectbox("Example", A.cat[A.cat.role == "student"].label.tolist())
        wav = A.wavs[pick]
        y = load_clean(wav)[0]

    if y is not None:
        dur = len(y) / SR
        if not 0.3 <= dur <= 2.2:
            st.warning(f"Detected {dur:.2f} s of speech; expected one short word (about 0.5-1.2 s). "
                       "Try again with a pause before and after the word.")
        r = A.assess(y)
        matches = r["distance"] <= thr
        score = 100 * 0.5 ** (max(0.0, r["distance"] - center) / max(thr - center, 0.05))

        c1, c2, c3 = st.columns(3)
        c1.metric("Similarity", f"{score:.0f}%")
        c2.metric("Distance", f"{r['distance']:.2f}", f"typical correct take {center:.2f}",
                  delta_color="off")
        c3.metric("Verdict", "Matches" if matches else "Differs")
        if matches:
            st.success("Within the normal variation of the correct recordings.")
        else:
            st.warning("Outside the normal variation of the correct recordings.")

        if r["localized"]:
            t0, t1 = r["window"]
            st.markdown(f"**Biggest difference:** {t0:.2f}-{t1:.2f} s of the reference "
                        f"({r['where']} of the word), {r['peak']:.1f} sd above normal.")
            for cue in r["cues"]:
                st.markdown(f"- {cue}")
        else:
            st.info("No segment differs by more than a correct recording normally does.")

        st.pyplot(make_figure(A, y, r))
        st.audio(str(wav))

        with st.expander("Log this attempt (for the evaluation)"):
            said = st.selectbox("What did you actually say?",
                                ["krishna (correct)", "kisna", "krishn", "krisna", "krishnda", "other"])
            if st.button("Save to log"):
                key = (said, round(r["distance"], 3))
                if st.session_state.get("last_saved") == key:
                    st.info("This attempt is already logged.")
                else:
                    path = Path("outputs/live_log.csv")
                    path.parent.mkdir(exist_ok=True)
                    row = dict(time=datetime.now().isoformat(timespec="seconds"), ref_set=ref_set,
                               said=said, distance=round(r["distance"], 3), score=round(score, 1),
                               verdict="Matches" if matches else "Differs",
                               peak_sd=round(r["peak"], 1),
                               window=f"{r['window'][0]:.2f}-{r['window'][1]:.2f}" if r["localized"] else "",
                               mu=round(center, 3), sigma=round(A.sigma, 3), thr=round(thr, 3))
                    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
                    st.session_state["last_saved"] = key
                    st.success(f"Logged. Total attempts so far: {len(pd.read_csv(path))}")

        st.caption(
            f"The percentage is a display scale: 100% at the typical correct attempt, 50% exactly at the "
            f"flag threshold. Scores are only meaningful when the references and your recording come from "
            f"the same device and room. The distance and the highlighted segment are the more reliable evidence.")

with tab_cal:
    st.markdown(
        f"Record **one clip** in which you say the *correct* word **5 or 6 times**, pausing about a "
        f"second between takes, using the microphone and room you will demo with. "
        f"Currently saved laptop takes for '{word}': **{n_laptop_refs(word)}**.")
    gap = st.slider("Silence that separates takes (s). Lower if takes are merged, raise if one word is split.",
                    0.2, 1.0, 0.4, 0.05)
    cal = st.audio_input("Record the correct takes", key="calib")
    if cal is not None:
        wav_c = to_16k(cal)
        y_raw, _ = librosa.load(str(wav_c), sr=SR, mono=True)
        takes = find_takes(y_raw, gap)
        st.write(f"Found **{len(takes)}** takes: " + ", ".join(f"{b - a:.2f}s" for a, b in takes))
        st.audio(str(wav_c))
        if len(takes) < 4:
            st.warning("Need at least 4 takes. Pause longer between takes, or lower the setting above.")
        elif st.button("Save as laptop references", type="primary"):
            out = LAPTOP_DIR / word
            out.mkdir(parents=True, exist_ok=True)
            for f in out.glob("*.wav"):
                f.unlink()
            for n, (a, b) in enumerate(takes, 1):
                seg = y_raw[max(0, int((a - PAD) * SR)): int((b + PAD) * SR)]
                sf.write(out / f"lap_{n:02d}.wav", seg, SR, subtype="PCM_16")
            st.cache_resource.clear()
            st.success(f"Saved {len(takes)} takes. Press R (rerun) and choose 'Laptop microphone' in the sidebar.")
import subprocess
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

SR = 16000
AUDIO_EXT = {".m4a", ".wav", ".mp3", ".mp4", ".aac", ".ogg", ".flac", ".opus"}


def to_wav(src, dst, sr=SR):
    import imageio_ffmpeg
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-i", str(src), "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src}: {r.stderr[-400:]}")
    return dst


def speech_bounds(y, sr=SR, frame=400, hop=160, pad_ms=80):
    """Speech start/end (samples) from speech-band energy vs. the noise floor."""
    sos = butter(4, [150, 4000], btype="bandpass", fs=sr, output="sos")
    rms = librosa.feature.rms(y=sosfiltfilt(sos, y), frame_length=frame, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-8)
    db -= db.max()
    floor = float(np.percentile(db, 10))
    thr = min(floor + 15, -20.0)
    idx = np.where(db > thr)[0]
    if len(idx) == 0:
        return 0, len(y), floor, thr
    pad = int(pad_ms / 1000 * sr)
    return max(0, idx[0] * hop - pad), min(len(y), idx[-1] * hop + frame + pad), floor, thr


def load_clean(path, return_info=False):
    """Load wav, remove rumble, trim to the speech, peak-normalize.
    Returns (clean, raw) or (clean, raw, info) if return_info=True."""
    raw, _ = librosa.load(str(path), sr=SR, mono=True)
    hp = sosfiltfilt(butter(4, 100, btype="highpass", fs=SR, output="sos"), raw)
    s, e, floor, thr = speech_bounds(hp)
    clean = hp[s:e]
    clean = clean / (np.max(np.abs(clean)) + 1e-9) * 0.95
    if return_info:
        return clean, raw, dict(floor_db=round(floor, 1), start_s=round(s / SR, 2),
                                end_s=round(e / SR, 2))
    return clean, raw
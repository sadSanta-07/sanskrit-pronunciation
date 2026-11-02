
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from src.audio_io import SR, load_clean, to_wav
from src.engine import Assessor, make_figure
from src.split_takes import PAD, find_takes

OUTPUT_DIR = ROOT / "outputs"
CATALOG_PATH = OUTPUT_DIR / "catalog.csv"
TMP_DIR = ROOT / "data" / "wav" / "_web_tmp"
LAPTOP_DIR = ROOT / "data" / "wav" / "ref_laptop"
WEB_DIR = ROOT / "web"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

ALLOWED_AUDIO = {
    ".wav", ".m4a", ".mp3", ".aac", ".ogg",
    ".flac", ".opus", ".webm", ".mp4"
}

app = FastAPI(
    title="Sanskrit Pronunciation API",
    version="1.0.0"
)

ASSESSORS = {}
ASSESSOR_LOCK = threading.RLock()
LOG_LOCK = threading.Lock()


def require_catalog():
    if not CATALOG_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "outputs/catalog.csv is missing. "
                "Run `python -m src.prepare_data` "
                "from the project root first."
            )
        )

    try:
        return pd.read_csv(CATALOG_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read catalog: {exc}"
        )


def safe_word(word: str):
    catalog = require_catalog()

    words = set(
        catalog.loc[
            catalog["role"] == "reference", "word"
        ].astype(str)
    )

    if word not in words:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown word: {word}"
        )

    return word


def laptop_reference_dir(word: str):
    return LAPTOP_DIR / word


def laptop_reference_count(word: str):
    path = laptop_reference_dir(word)

    if not path.exists():
        return 0

    return len(list(path.glob("*.wav")))


def get_assessor(word: str, ref_set: str):
    safe_word(word)

    if ref_set not in {
        "Phone recordings",
        "Laptop microphone"
    }:
        raise HTTPException(
            status_code=400,
            detail="Invalid reference set."
        )

    ref_dir = None

    if ref_set == "Laptop microphone":
        if laptop_reference_count(word) < 4:
            raise HTTPException(
                status_code=400,
                detail="At least four laptop references are required."
            )

        ref_dir = str(laptop_reference_dir(word))

    key = (word, ref_set)

    with ASSESSOR_LOCK:
        if key not in ASSESSORS:
            try:
                ASSESSORS[key] = Assessor(word, ref_dir)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not build baseline: {exc}"
                )

        return ASSESSORS[key]


def dev_correct_distances():
    path = OUTPUT_DIR / "live_log_dev.csv"

    if not path.exists():
        return []

    try:
        frame = pd.read_csv(path)

        required = {"ref_set", "said", "distance"}

        if not required.issubset(frame.columns):
            return []

        frame = frame.drop_duplicates(
            ["said", "distance"]
        )

        rows = frame[
            (frame["ref_set"] == "Laptop microphone") &
            (frame["said"] == "krishna (correct)")
        ]

        return (
            pd.to_numeric(
                rows["distance"],
                errors="coerce"
            )
            .dropna()
            .astype(float)
            .tolist()
        )

    except Exception:
        return []


def baseline_stats(assessor, ref_set: str):
    dev = (
        dev_correct_distances()
        if ref_set == "Laptop microphone"
        else []
    )

    if len(dev) >= 5:
        center = float(np.mean(dev))
        threshold = float(max(dev)) + 0.01
    else:
        center = float(assessor.mu)
        threshold = float(assessor.dist_thr)

    return {
        "center": center,
        "threshold": threshold,
        "reference_count": len(assessor.refs),
        "primary_reference": assessor.primary,
        "sigma": float(assessor.sigma),
        "take_distances": [
            {
                "take": str(label),
                "distance": float(distance)
            }
            for label, distance in assessor.ref_dists.items()
        ]
    }


def normalize_catalog_path(value: str):
    path = Path(str(value).replace("\\", "/"))

    if path.is_absolute():
        return path

    return ROOT / path


async def receive_audio(upload: UploadFile):
    suffix = (
        Path(upload.filename or "recording.wav")
        .suffix.lower() or ".wav"
    )

    if suffix not in ALLOWED_AUDIO:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: {suffix}"
        )

    payload = await upload.read(MAX_UPLOAD_BYTES + 1)

    if not payload:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio is empty."
        )

    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Audio upload exceeds 25 MB."
        )

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex

    source = TMP_DIR / f"{token}{suffix}"
    wav_path = TMP_DIR / f"{token}_16k.wav"

    source.write_bytes(payload)

    try:
        to_wav(source, wav_path)

    except Exception as exc:
        source.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=422,
            detail=f"Could not decode audio: {exc}"
        )

    return source, wav_path


def figure_as_data_url(assessor, y, result):
    fig = make_figure(assessor, y, result)

    buffer = io.BytesIO()

    try:
        fig.savefig(
            buffer,
            format="png",
            dpi=130,
            bbox_inches="tight"
        )
    finally:
        fig.clear()

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "sanskrit-pronunciation"
    }


@app.get("/api/config")
def config():
    catalog = require_catalog()

    words = sorted(
        catalog.loc[
            catalog["role"] == "reference", "word"
        ].astype(str).unique()
    )

    entries = []

    for word in words:
        options = ["Phone recordings"]

        if laptop_reference_count(word) >= 4:
            options.append("Laptop microphone")

        examples = catalog[
            (catalog["word"] == word) &
            (catalog["role"] == "student")
        ]

        entries.append({
            "id": word,
            "display": "कृष्ण" if word == "krishna" else word,
            "transliteration": "kṛṣṇa" if word == "krishna" else word,
            "reference_sets": options,
            "examples": [
                str(x) for x in examples["label"].tolist()
            ]
        })

    return {
        "words": entries,
        "sample_rate": SR,
        "max_upload_mb": 25
    }


@app.get("/api/baseline")
def baseline(
    word: str = "krishna",
    ref_set: str = "Phone recordings"
):
    assessor = get_assessor(word, ref_set)

    return {
        "word": word,
        "ref_set": ref_set,
        **baseline_stats(assessor, ref_set)
    }


@app.get("/api/examples/{label}/audio")
def example_audio(
    label: str,
    word: str = "krishna"
):
    safe_word(word)

    catalog = require_catalog()

    rows = catalog[
        (catalog["word"].astype(str) == word) &
        (catalog["role"].astype(str) == "student") &
        (catalog["label"].astype(str) == label)
    ]

    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail="Recorded example not found."
        )

    path = normalize_catalog_path(
        rows.iloc[0]["wav"]
    )

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Example audio is missing: {path.name}"
        )

    return FileResponse(
        path,
        media_type="audio/wav",
        filename=path.name
    )


@app.post("/api/assess")
async def assess(
    word: str = Form("krishna"),
    ref_set: str = Form("Phone recordings"),
    threshold: float | None = Form(None),
    example_label: str | None = Form(None),
    file: UploadFile | None = File(None)
):
    assessor = get_assessor(word, ref_set)
    stats = baseline_stats(assessor, ref_set)

    used_source = "upload"

    source_path = None
    wav_path = None

    try:
        if example_label:
            if ref_set != "Phone recordings":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Recorded examples are phone recordings. "
                        "Select Phone recordings as the reference set."
                    )
                )

            catalog = require_catalog()

            rows = catalog[
                (catalog["word"].astype(str) == word) &
                (catalog["role"].astype(str) == "student") &
                (catalog["label"].astype(str) == example_label)
            ]

            if rows.empty:
                raise HTTPException(
                    status_code=404,
                    detail="Recorded example not found."
                )

            example_path = normalize_catalog_path(
                rows.iloc[0]["wav"]
            )

            if not example_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=f"Example audio is missing: {example_path.name}"
                )

            y = load_clean(example_path)[0]
            used_source = "example"

        else:
            if file is None:
                raise HTTPException(
                    status_code=400,
                    detail="Upload or record audio first."
                )

            source_path, wav_path = await receive_audio(file)
            y = load_clean(wav_path)[0]

        duration = float(len(y) / SR)

        if duration < 0.15:
            raise HTTPException(
                status_code=422,
                detail="No clear speech detected. Try recording again."
            )

        # Existing pronunciation engine — unchanged.
        result = assessor.assess(y)

        center = stats["center"]

        active_threshold = (
            float(threshold)
            if threshold is not None
            else stats["threshold"]
        )

        matches = float(result["distance"]) <= active_threshold

        # Same display-scale formula used by the Streamlit app.
        score = 100 * 0.5 ** (
            max(0.0, float(result["distance"]) - center) /
            max(active_threshold - center, 0.05)
        )

        window = result["window"]

        return {
            "word": word,
            "ref_set": ref_set,
            "source": used_source,
            "duration": duration,
            "distance": float(result["distance"]),
            "score": float(score),
            "matches": bool(matches),
            "threshold": active_threshold,
            "center": center,
            "reference_count": stats["reference_count"],
            "primary_reference": stats["primary_reference"],
            "peak": float(result["peak"]),
            "localized": bool(result["localized"]),
            "window": list(window) if window else None,
            "where": result["where"],
            "cues": list(result["cues"]),
            "spectrogram": figure_as_data_url(
                assessor, y, result
            ),
            "disclaimer": (
                "The percentage is a display scale, not a probability. "
                "Compare recordings made with the same device and in the same room."
            )
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Assessment failed: {exc}"
        )

    finally:
        if source_path:
            source_path.unlink(missing_ok=True)

        if wav_path:
            wav_path.unlink(missing_ok=True)

        if file is not None:
            await file.close()


class LogAttempt(BaseModel):
    ref_set: Literal[
        "Phone recordings",
        "Laptop microphone"
    ]

    said: Literal[
        "krishna (correct)",
        "kisna",
        "krishn",
        "krisna",
        "krishnda",
        "other"
    ]

    distance: float = Field(ge=0)
    score: float = Field(ge=0, le=100)
    verdict: Literal["Matches", "Differs"]
    peak_sd: float = Field(ge=0)
    window: str = ""
    mu: float
    sigma: float = Field(ge=0)
    thr: float = Field(ge=0)


@app.post("/api/log")
def log_attempt(attempt: LogAttempt):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / "live_log.csv"

    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        **attempt.model_dump()
    }

    with LOG_LOCK:
        pd.DataFrame([row]).to_csv(
            path,
            mode="a",
            header=not path.exists(),
            index=False
        )

        with path.open("r", encoding="utf-8") as handle:
            total = sum(1 for _ in handle) - 1

    return {
        "saved": True,
        "total_attempts": max(total, 0)
    }


async def decode_calibration_upload(upload: UploadFile):
    source, wav_path = await receive_audio(upload)

    try:
        y, _ = librosa.load(
            str(wav_path),
            sr=SR,
            mono=True
        )

        return source, wav_path, y

    except Exception as exc:
        source.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=422,
            detail=f"Could not read calibration audio: {exc}"
        )


@app.post("/api/calibration/preview")
async def calibration_preview(
    word: str = Form("krishna"),
    gap: float = Form(0.4),
    file: UploadFile = File(...)
):
    safe_word(word)

    if not 0.2 <= gap <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Gap must be between 0.2 and 1.0 seconds."
        )

    source, wav_path, y = await decode_calibration_upload(file)

    try:
        takes = find_takes(y, gap)

        return {
            "word": word,
            "count": len(takes),
            "takes": [
                {
                    "index": i,
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                    "duration": round(float(end - start), 2)
                }
                for i, (start, end) in enumerate(takes)
            ],
            "can_save": len(takes) >= 4,
            "warning": (
                "Use the same microphone and room for calibration "
                "and assessment. Multiple takes in one clip can be "
                "more similar than independent attempts."
            )
        }

    finally:
        source.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)
        await file.close()


@app.post("/api/calibration/save")
async def calibration_save(
    word: str = Form("krishna"),
    gap: float = Form(0.4),
    selected_indices: str = Form("[]"),
    file: UploadFile = File(...)
):
    safe_word(word)

    if not 0.2 <= gap <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Gap must be between 0.2 and 1.0 seconds."
        )

    try:
        selected = sorted(
            set(int(x) for x in json.loads(selected_indices))
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="selected_indices must be a JSON array."
        )

    if len(selected) < 4:
        raise HTTPException(
            status_code=400,
            detail="Select at least four takes."
        )

    source, wav_path, y = await decode_calibration_upload(file)

    stage_dir = None
    backup_dir = None

    try:
        takes = find_takes(y, gap)

        if any(
            index < 0 or index >= len(takes)
            for index in selected
        ):
            raise HTTPException(
                status_code=400,
                detail="Selected take is invalid. Preview again."
            )

        out_dir = laptop_reference_dir(word)
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        token = uuid.uuid4().hex

        stage_dir = out_dir.parent / f".{word}_new_{token}"
        backup_dir = out_dir.parent / f".{word}_backup_{token}"

        stage_dir.mkdir(parents=True)

        for number, index in enumerate(selected, start=1):
            start, end = takes[index]

            segment = y[
                max(0, int((start - PAD) * SR)):
                int((end + PAD) * SR)
            ]

            sf.write(
                stage_dir / f"lap_{number:02d}.wav",
                segment,
                SR,
                subtype="PCM_16"
            )

        # Replace old references only after all new takes are written.
        if out_dir.exists():
            os.replace(out_dir, backup_dir)

        try:
            os.replace(stage_dir, out_dir)

        except Exception:
            if backup_dir.exists() and not out_dir.exists():
                os.replace(backup_dir, out_dir)
            raise

        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        with ASSESSOR_LOCK:
            ASSESSORS.pop(
                (word, "Laptop microphone"),
                None
            )

        return {
            "saved": len(selected),
            "reference_set": "Laptop microphone",
            "message": (
                "Laptop references saved. "
                "Select Laptop microphone to rebuild the baseline."
            )
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save calibration references: {exc}"
        )

    finally:
        if stage_dir and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)

        source.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)

        await file.close()


if WEB_DIR.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(WEB_DIR), html=True),
        name="web"
    )

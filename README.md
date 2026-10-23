# Sanskrit Pronunciation Using MFCC and Spectral Features

Compares a spoken Sanskrit word with correct recordings and reports a similarity score, a verdict, and **where in the word** the difference is. Built for the SVP-4 project work; the word studied is कृष्ण (kṛṣṇa).

## Approach

```
audio -> 16 kHz mono, high-pass, trim to speech
      -> MFCC (13) + delta + delta-delta = 39 dims per 10 ms frame, per-recording CMVN
      -> spectral cues: centroid, rolloff, bandwidth, flatness, zero-crossing rate, energy
      -> DTW alignment against each correct recording -> distance (median over references)
      -> distance vs. the variation of correct recordings -> score, verdict, per-frame difference map
```

- **Preprocessing.** Recordings are converted to 16 kHz mono, rumble below 100 Hz is removed, and the speech is trimmed using band energy (150-4000 Hz) relative to each file's own noise floor.
- **Features.** MFCCs with deltas capture the sound and how it changes; per-recording mean and variance normalization (CMVN) removes constant microphone and voice colouring. Spectral centroid, flatness and zero-crossing rate separate hissy sounds ("sh", "s") from vowels.
- **Alignment.** Dynamic time warping aligns the speaker's frames to a reference so different speaking speeds can be compared. The distance is the mean frame distance along the warping path.
- **Score.** A recording's distance is the median over all correct references. The verdict compares it to a threshold; the percentage is a display scale (100% at a typical correct attempt, 50% at the threshold).
- **Feedback.** Each frame of the most typical correct take is compared with how much *other correct takes* vary there. The window with the largest excess is reported with simple cues (brighter/darker, noisier, louder/quieter, longer/shorter).

## Data

All recordings are by one speaker. Correct "Krishna": 5 takes. Four error variants, each 3 takes: **kisna** (r dropped), **krishn** (final "a" dropped), **krisna** ("s" for "sh"), **krishnda** (extra "d"). Phone recordings were used for development; the web app also supports laptop-microphone recordings.

## Results

**1. Phone recordings (leave-one-out over 5 correct takes).** All 5 correct takes score 91-100%; all 12 error takes are flagged (AUC 1.00). Mean distance: correct 2.12, kisna 2.70, krisna 2.76, krishnda 2.89, krishn 3.25. Caveat: the baseline and the tests come from the same recording session, which turned out to be optimistic (see 3).

**2. Recording session matters.** A correct take recorded in a separate session landed at 2.86, versus 2.11 within the main session; correct takes from the first session sat 0.2-0.7 higher than their counterparts. The distance measures the pronunciation *and* the recording conditions.

**3. Calibrating on one clip is too tight.** A laptop-microphone baseline built from 8 takes recorded back-to-back gave a threshold of 3.15, but separately recorded correct attempts averaged 3.20 and **71% (5/7) were falsely flagged**. Takes inside one recording are closer to each other than a fresh recording is to them, so the threshold must come from independent attempts.

**4. Dev/test evaluation (laptop microphone).** Threshold rule, fixed *before* the test: just above the highest correct dev attempt (3.41), i.e. 3.42.

| | Correct flagged | Errors detected | AUC |
|---|---|---|---|
| Dev (7 correct, 12 errors) | 1/7 expected at 3.42 | 10/12 | 0.96 |
| **Test (6 correct, 12 errors), held out** | **0/6** | **9/12** | **0.96** |

Test detection by error: krishn 3/3, krishnda 3/3, kisna 2/3, krisna 1/3. The subtle sound substitutions are the weak spot. Intervals are wide: 0/6 false alarms has a 95% upper bound of about 46%; 9/12 detections has a 95% interval of roughly 43-95%.

**5. Localization.** On phone recordings the strongest-difference window was consistent across the three takes of each error (3/3). krishn: end of the word (shorter, quieter, darker); krishnda: a noisy burst at about 0.55-0.65 s; kisna and krisna: around the fricative. A held-out correct take reached 4.4 sd, against 6.4-11.7 sd for errors, so the margin for kisna and krisna is small.

## Limitations

- One speaker, one phone, one laptop; the errors are deliberate, not real learner mistakes.
- Small sample (5-8 correct takes, 12 error takes per batch): thresholds are rough and intervals are wide.
- References must be recorded on the same device and in the same room as the test; otherwise the distances shift by more than a subtle error does.
- Whole-word DTW distance dilutes short, localized differences: "s" vs "sh" (krisna) is detected in 1 of 3 test attempts.
- The "difference" cues are simple acoustic statistics, not phoneme-level diagnoses. No audio of other speakers or a second word was tested.
- The similarity percentage is a display scale, not a probability.

## Future work

Phoneme-level forced alignment so that each sound is scored on its own; a fricative-specific feature (spectral centroid inside the "sh" segment); more speakers and more takes; a learned pronunciation model (for example wav2vec 2.0 embeddings).

## Run it

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m src.prepare_data          # convert audio, build the catalog
python -m src.features              # MFCC and spectral plots
python -m src.score --session B     # leave-one-out scores (phone recordings)
python -m src.feedback              # per-segment difference map
streamlit run app.py                # live app: record, upload, calibrate
python -m src.live_eval --dedupe --thr 3.42 --tag test   # evaluate logged attempts
```

## Structure

```
app.py                Streamlit app (assess, calibrate laptop microphone, log attempts)
src/audio_io.py       conversion, noise-aware trimming
src/features.py       MFCC (+delta) and spectral features, plots
src/align.py          DTW alignment and distance
src/score.py          leave-one-out baseline, score, evaluation
src/feedback.py       baseline-relative difference map and cues
src/engine.py         reusable assessor used by the app
src/split_takes.py    split a multi-take voice memo into files
src/live_eval.py      evaluate logged live attempts
data/                 recordings; outputs/ generated results and logs
```
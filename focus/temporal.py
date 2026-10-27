"""Temporal (window) features computed from the 3 Hz per-frame stream.

Design rules
- Nothing derived from a label goes in (the v2.1 state contained
  is_distracted_label and consecutive_frames, i.e. the answer).
- Eye and head features are relative to the person's own reference, taken
  from the first REF_SAMPLES of the session (the "calibration"), so the same
  model works for people with small eyes, glasses or a low camera.
- One function, `window_features(buffer, ref)`, computes the vector from the
  last W_LONG samples. web/core.js has an identical implementation.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

HZ = 3.0
W_SHORT = 60      # 20 s
W_LONG = 180      # 60 s
REF_SAMPLES = 60  # first 20 s of the session
CLOSED = 0.7      # eye counts as closed below 70 % of the person's open-eye EAR
YAWN_MAR = 0.5

FEATURES = [
    "ear_rel_mean20", "ear_rel_std20", "perclos60", "closures_per_min60",
    "longest_closure60", "mar_mean60", "yawn_frac60", "pitch_rel_mean20",
    "pitch_std20", "yaw_std20", "abs_yaw_rel_mean20", "gaze_std20", "face_missing60",
]


def reference(samples: Sequence[dict]) -> dict:
    """Median open-eye EAR and neutral head pose from the first samples with a face."""
    face = [s for s in samples[:REF_SAMPLES] if s["ear_score"] > 0]
    if len(face) < 10:
        face = [s for s in samples if s["ear_score"] > 0] or [{"ear_score": 0.28, "head_pitch": 0.0, "head_yaw": 0.0}]
    med = lambda k: float(np.median([s[k] for s in face]))
    return {"ear": max(med("ear_score"), 1e-3), "pitch": med("head_pitch"), "yaw": med("head_yaw")}


def _mean(a):
    return sum(a) / len(a) if a else 0.0


def _std(a):
    if not a:
        return 0.0
    m = _mean(a)
    return math.sqrt(_mean([(v - m) ** 2 for v in a]))


def window_features(buf: Sequence[dict], ref: dict) -> list[float]:
    """buf = the most recent samples (oldest first), each with ear_score,
    mar_score, head_pitch, head_yaw, gaze_ratio. ear_score == 0 means no face."""
    long_ = list(buf[-W_LONG:])
    short = long_[-W_SHORT:]
    lf = [s for s in long_ if s["ear_score"] > 0]
    sf = [s for s in short if s["ear_score"] > 0]

    ear_rel_l = [s["ear_score"] / ref["ear"] for s in lf]
    ear_rel_s = [s["ear_score"] / ref["ear"] for s in sf]
    closed = [1 if e < CLOSED else 0 for e in ear_rel_l]

    closures = 0
    longest = run = 0
    for i, c in enumerate(closed):
        if c and (i == 0 or not closed[i - 1]):
            closures += 1
        run = run + 1 if c else 0
        longest = max(longest, run)
    minutes = max(len(long_) / HZ / 60.0, 1 / 60.0)

    pitch_rel = [s["head_pitch"] - ref["pitch"] for s in sf]
    yaw_rel = [s["head_yaw"] - ref["yaw"] for s in sf]
    mars = [s["mar_score"] for s in lf]

    return [
        _mean(ear_rel_s) if ear_rel_s else 1.0,
        _std(ear_rel_s),
        _mean(closed),
        closures / minutes,
        longest / HZ,
        _mean(mars),
        _mean([1 if m > YAWN_MAR else 0 for m in mars]),
        _mean(pitch_rel),
        _std(pitch_rel),
        _std(yaw_rel),
        _mean([abs(v) for v in yaw_rel]),
        _std([s["gaze_ratio"] for s in sf]),
        1 - len(lf) / len(long_) if long_ else 1.0,
    ]


def session_windows(df, stride: int = 3, min_samples: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Slide over one session (a DataFrame sorted by time). Returns
    (X [n, len(FEATURES)], index of the sample each row ends at)."""
    d = df
    if "mp_pitch" in df.columns:   # pose from MediaPipe's matrix = the convention the web app uses
        d = df.drop(columns=["head_pitch", "head_yaw"]).rename(columns={"mp_pitch": "head_pitch", "mp_yaw": "head_yaw"})
    cols = ["ear_score", "mar_score", "head_pitch", "head_yaw", "gaze_ratio"]
    recs = d[cols].to_dict("records")
    ref = reference(recs)
    X, idx = [], []
    for t in range(min_samples - 1, len(recs), stride):
        X.append(window_features(recs[max(0, t - W_LONG + 1): t + 1], ref))
        idx.append(t)
    return np.asarray(X, dtype=np.float64).reshape(-1, len(FEATURES)), np.asarray(idx, dtype=int)

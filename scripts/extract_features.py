"""
extract_features.py — turn videos from public datasets into focus_dataset.csv rows
===================================================================================

Runs the same per-frame features as the desktop app (MediaPipe FaceLandmarker +
demo_webcam.py metrics, optional YOLO phone detection), resamples to FPS=3 like
the training data, and writes one CSV per call with the schema of
data/focus_dataset.csv plus three extra columns:

    subject_id   who is in the video (for leave-subject-out splits)
    source       dataset name
    gt_label     the dataset's own label (NaN when it has none)

`is_distracted_label` is still produced by the hand-written rule so the file stays
compatible with train_model.py. Use `gt_label` for evaluation: it is the only label
that does not come from the same rule the agent is rewarded on.

Examples
--------
# UTA-RLDD: folder per subject, files 0.mp4 / 5.mp4 / 10.mp4 = alert / low vigilant / drowsy
python scripts/extract_features.py --dataset uta-rldd --root D:/UTA-RLDD --out data/uta_rldd.csv

# NTHU-DDD: per-frame label file next to each video (one '0'/'1' character per frame)
python scripts/extract_features.py --dataset nthu-ddd --root D:/NTHU-DDD --out data/nthu.csv

# DAiSEE: Labels/AllLabels.csv gives Engagement 0-3 per clip
python scripts/extract_features.py --dataset daisee --root D:/DAiSEE --out data/daisee.csv

# Any folder of your own recordings, no ground truth
python scripts/extract_features.py --dataset folder --root D:/my_videos --out data/mine.csv

Needs: opencv-python, mediapipe, pandas, numpy (+ ultralytics for --phone).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import string
import sys
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

from demo_webcam import (  # noqa: E402
    calculate_brow_dist, calculate_ear, calculate_gaze_ratio, calculate_mar, estimate_head_pose,
)

TARGET_FPS = 3.0
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
COLUMNS = [
    "session_id", "timestamp", "head_pitch", "head_yaw", "head_roll", "ear_score", "mar_score",
    "brow_dist", "person_detected", "phone_count", "consecutive_frames", "is_distracted_label",
    "delta_yaw", "delta_pitch", "delta_ear", "gaze_ratio", "fatigue_score",
    "subject_id", "source", "gt_label",
]


# ── helpers ───────────────────────────────────────────────────────────────

def wrap_angle(a: float) -> float:
    """Fold Euler angles into [-90, 90]. cv2.decomposeProjectionMatrix can return
    values near ±180 for a frontal face, which put 27% of the original rows on the
    wrong side of the scale."""
    a = (a + 180.0) % 360.0 - 180.0
    if a > 90.0:
        a -= 180.0
    elif a < -90.0:
        a += 180.0
    return a


def rule_label(pitch, yaw, ear, phone, person, looking_down_frames) -> int:
    """Same rule as FocusAgentInference.process_frame, counted at 3 fps (30 s = 90 frames)."""
    note_taking = pitch > 20 and abs(yaw) < 20 and phone == 0 and ear >= 0.20 and looking_down_frames < 90
    return int(ear < 0.20 or abs(yaw) > 40 or (abs(pitch) > 30 and not note_taking)
               or pitch < -25 or phone > 0 or person == 0)


def new_session_id() -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=8))


# ── dataset adapters: yield (video_path, subject_id, gt_fn) ───────────────
# gt_fn(frame_index) -> label or NaN

GtFn = Callable[[int], float]


def _videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT)


def adapt_folder(root: Path) -> Iterator[tuple[Path, str, GtFn]]:
    for v in _videos(root):
        yield v, v.parent.name, (lambda i: math.nan)


def adapt_uta_rldd(root: Path) -> Iterator[tuple[Path, str, GtFn]]:
    # file stem is the KSS-based class: 0 alert, 5 low vigilant, 10 drowsy → 0 / 1 / 2
    cls = {"0": 0.0, "5": 1.0, "10": 2.0}
    for v in _videos(root):
        label = cls.get(v.stem, math.nan)
        yield v, v.parent.name, (lambda i, y=label: y)


def adapt_nthu_ddd(root: Path) -> Iterator[tuple[Path, str, GtFn]]:
    # label files are strings of 0/1, one char per frame; name conventions vary between
    # releases, so look for any .txt in the same folder whose name starts with the video stem
    for v in _videos(root):
        cands = sorted(v.parent.glob(f"{v.stem}*drowsiness*.txt")) or sorted(v.parent.glob(f"{v.stem}*.txt"))
        labels = cands[0].read_text().strip() if cands else ""
        subject = v.parts[len(root.parts)] if len(v.parts) > len(root.parts) + 1 else v.parent.name
        yield v, subject, (lambda i, s=labels: float(s[i]) if i < len(s) and s[i] in "01" else math.nan)


def adapt_daisee(root: Path) -> Iterator[tuple[Path, str, GtFn]]:
    # Labels/AllLabels.csv: ClipID, Boredom, Engagement, Confusion, Frustration (0-3)
    lab_file = next(iter(root.rglob("AllLabels.csv")), None)
    eng = {}
    if lab_file:
        df = pd.read_csv(lab_file)
        df.columns = [c.strip() for c in df.columns]
        eng = {str(r["ClipID"]).strip(): float(r["Engagement"]) for _, r in df.iterrows()}
    for v in _videos(root):
        label = eng.get(v.name, eng.get(v.stem + ".avi", eng.get(v.stem + ".mp4", math.nan)))
        subject = v.parts[len(root.parts) + 1] if len(v.parts) > len(root.parts) + 2 else v.parent.name
        yield v, subject, (lambda i, y=label: y)


ADAPTERS = {"folder": adapt_folder, "uta-rldd": adapt_uta_rldd, "nthu-ddd": adapt_nthu_ddd, "daisee": adapt_daisee}


# ── per-video extraction ──────────────────────────────────────────────────

def process_video(path: Path, subject: str, gt: GtFn, source: str, detector, yolo,
                  max_seconds: Optional[float]) -> list[dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  [SKIP] cannot open {path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / TARGET_FPS)))
    sid = new_session_id()
    rows: list[dict] = []
    last = {"yaw": 0.0, "pitch": 0.0, "ear": 0.0}
    consec = looking_down = 0
    t0_frames = None
    idx = -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = idx / fps
        if max_seconds and t > max_seconds:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        pitch = yaw = roll = ear = mar = brow = 0.0
        gaze = 0.5
        person = 0
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            pitch, yaw, roll = (wrap_angle(a) for a in estimate_head_pose(lm, frame.shape))
            ear, mar = calculate_ear(lm), calculate_mar(lm)
            brow, gaze = calculate_brow_dist(lm, frame.shape), calculate_gaze_ratio(lm)
            person = 1

        phone = 0
        if yolo is not None:
            r = yolo(frame, verbose=False)[0]
            phone = sum(1 for b in r.boxes if int(b.cls) == 67 and float(b.conf) > 0.5)
            if not person and any(int(b.cls) == 0 and float(b.conf) > 0.5 for b in r.boxes):
                person = 1

        looking_down = looking_down + 1 if (pitch > 20 and abs(yaw) < 20 and ear >= 0.22) else 0
        label = rule_label(pitch, yaw, ear, phone, person, looking_down)
        consec = consec + 1 if label else 0
        elapsed_norm = min(1.0, t / 3600.0)
        fatigue = min(1.0, elapsed_norm * (0.4 - min(ear, 0.4)) * 2.5)

        rows.append({
            "session_id": sid, "timestamp": round(t, 3),
            "head_pitch": round(pitch, 2), "head_yaw": round(yaw, 2), "head_roll": round(roll, 2),
            "ear_score": round(ear, 4), "mar_score": round(mar, 4), "brow_dist": round(brow, 3),
            "person_detected": person, "phone_count": phone,
            "consecutive_frames": consec, "is_distracted_label": label,
            "delta_yaw": round(yaw - last["yaw"], 3), "delta_pitch": round(pitch - last["pitch"], 3),
            "delta_ear": round(ear - last["ear"], 4), "gaze_ratio": gaze, "fatigue_score": fatigue,
            "subject_id": subject, "source": source, "gt_label": gt(idx),
        })
        last = {"yaw": yaw, "pitch": pitch, "ear": ear}
    cap.release()
    face_rate = np.mean([r["ear_score"] > 0 for r in rows]) if rows else 0
    print(f"  {path.name}: {len(rows)} rows @ {TARGET_FPS:g} fps, face found {face_rate:.0%}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--face-model", default=str(ROOT / "models" / "face_landmarker.task"))
    ap.add_argument("--phone", action="store_true", help="run YOLOv8n phone detection (slow on CPU)")
    ap.add_argument("--max-seconds", type=float, default=None, help="cap per video, e.g. 600")
    ap.add_argument("--limit", type=int, default=None, help="process at most N videos (for a quick test)")
    args = ap.parse_args()

    detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.face_model), num_faces=1))
    yolo = None
    if args.phone:
        from ultralytics import YOLO
        yolo = YOLO(str(ROOT / "models" / "yolov8n.pt"))

    items = list(ADAPTERS[args.dataset](args.root))[: args.limit]
    if not items:
        sys.exit(f"No videos found under {args.root}")
    print(f"[*] {len(items)} videos from {args.dataset}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    total = 0
    for i, (video, subject, gt) in enumerate(items, 1):
        print(f"[{i}/{len(items)}] subject={subject}")
        rows = process_video(video, subject, gt, args.dataset, detector, yolo, args.max_seconds)
        if rows:
            pd.DataFrame(rows, columns=COLUMNS).to_csv(args.out, mode="a" if wrote_header else "w",
                                                       header=not wrote_header, index=False)
            wrote_header = True
            total += len(rows)
    print(f"[+] {total} rows → {args.out}")


if __name__ == "__main__":
    main()

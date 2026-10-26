"""Per-frame features from MediaPipe face landmarks.

Everything here has a line-for-line twin in web/core.js; tests/test_parity.py
checks both give the same numbers.

Coordinates are MediaPipe's normalised (x, y) in [0, 1]. Ratios are computed
as if the frame were 640x480 (to_4x3), because the original model and
thresholds were tuned on that webcam.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Sequence

RIGHT_EYE = (33, 160, 158, 133, 153, 144)   # corners 33 / 133
LEFT_EYE = (263, 387, 385, 362, 380, 373)   # corners 362 / 263
MOUTH = (61, 291, 13, 14)


def _d(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def to_4x3(lm: Sequence, width: int, height: int):
    """Rescale x so distances behave as on a 4:3 frame (no-op for 4:3 input)."""
    k = (width / height) * 0.75
    if abs(k - 1) < 1e-3:
        return lm
    return [SimpleNamespace(x=p.x * k, y=p.y, z=getattr(p, "z", 0.0)) for p in lm]


def ear(lm) -> float:
    def one(ix):
        p = [lm[i] for i in ix]
        h = _d(p[0], p[3])
        return 0.0 if h == 0 else (_d(p[1], p[5]) + _d(p[2], p[4])) / (2 * h)
    return (one(LEFT_EYE) + one(RIGHT_EYE)) / 2


def mar(lm) -> float:
    h = _d(lm[61], lm[291])
    return 0.0 if h == 0 else _d(lm[13], lm[14]) / h


def brow_dist(lm, width: int = 640, height: int = 480) -> float:
    a, b = lm[105], lm[159]
    return math.hypot((a.x - b.x) * width, (a.y - b.y) * height)


def gaze_ratio(lm) -> float:
    """Iris position inside each eye, measured from the image-left corner.
    0.5 = centred. 468 belongs to the eye with corners 33/133, 473 to 362/263."""
    if len(lm) < 478:
        return 0.5

    def g(iris, c0, c1):
        w = _d(lm[c0], lm[c1])
        return 0.5 if w == 0 else _d(lm[c0], lm[iris]) / w
    return (g(468, 33, 133) + g(473, 362, 263)) / 2


def pose_from_matrix(m: Sequence[float], invert_pitch: bool = False) -> tuple[float, float, float]:
    """Head pose (degrees) from MediaPipe's 4x4 facial transformation matrix,
    given column-major as MediaPipe JS returns it. pitch > 0 = looking down.
    No ±180° wrap-around, unlike cv2.decomposeProjectionMatrix."""
    r = lambda i, j: m[j * 4 + i]
    fwd = (r(0, 2), r(1, 2), r(2, 2))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -fwd[1]))))
    yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
    roll = math.degrees(math.atan2(r(1, 0), r(0, 0)))
    return (-pitch if invert_pitch else pitch), yaw, roll


def matrix_to_colmajor(mat) -> list[float]:
    """MediaPipe Python returns a 4x4 numpy array (row-major). Convert to the
    column-major list used by pose_from_matrix / the JS side."""
    return [float(mat[i][j]) for j in range(4) for i in range(4)]


def wrap_angle(a: float) -> float:
    """Fold an Euler angle into [-90, 90] (fixes decomposeProjectionMatrix flips)."""
    a = (a + 180.0) % 360.0 - 180.0
    if a > 90.0:
        a -= 180.0
    elif a < -90.0:
        a += 180.0
    return a


def frame_features(lm, width: int, height: int, matrix=None) -> dict:
    """All per-frame features for one detected face."""
    l43 = to_4x3(lm, width, height)
    out = {
        "ear_score": ear(l43),
        "mar_score": mar(l43),
        "brow_dist": brow_dist(l43),
        "gaze_ratio": gaze_ratio(l43),
    }
    if matrix is not None:
        out["head_pitch"], out["head_yaw"], out["head_roll"] = pose_from_matrix(matrix)
    return out


def pose_pnp(lm, width: int, height: int) -> tuple[float, float, float]:
    """Legacy head pose (v2.1): solvePnP on 6 landmarks against a generic 3D face,
    focal length = image width, then Euler angles folded into [-90, 90].
    Kept so the extracted CSVs stay comparable with focus_dataset.csv; new code
    uses pose_from_matrix."""
    import cv2
    import numpy as np
    model = np.array([[0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
                      [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]])
    img = np.array([[lm[i].x * width, lm[i].y * height] for i in (1, 152, 263, 33, 287, 57)], dtype=np.float64)
    cam = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
    _, rvec, tvec = cv2.solvePnP(model, img, cam, np.zeros((4, 1)))
    rot, _ = cv2.Rodrigues(rvec)
    ang = cv2.decomposeProjectionMatrix(np.hstack((rot, tvec.reshape(3, 1))))[6]
    return tuple(wrap_angle(float(a[0])) for a in ang)

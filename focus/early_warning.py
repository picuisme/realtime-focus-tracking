"""Early warning by temporal-difference value prediction.

Idea (suggested by Bui Van Hieu, FPT University): instead of feeding long frame
sequences to a model, treat each frame as a state s_t and learn its value

    V(s_t) = E[ sum_k gamma^k c_{t+k} ]

where the cumulant c_t is 1 while the person is distracted / eyes closed and 0
otherwise. V(s) rises before the event starts, so a threshold on V warns early.
V is linear in a small feature vector phi(s), so it runs in the browser and can
keep learning on one person with TD(lambda) using a cumulant that the app can
compute itself (the rule detector), i.e. without any labels.

Everything here is measured in seconds, so the sample rate can differ between
datasets (DMD ~10 Hz) and the web app.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CLOSED = 0.7          # eye counts as closed below 70 % of the open-eye reference
REF_S = 20.0          # calibration window at the start of a session

FEATURES = [
    "bias",
    # eyes
    "ear_rel", "ear_rel_mean2", "ear_rel_slope2", "closed_frac5", "closed_frac60",
    "closures_per_min60", "mean_closure_s60", "closed_run_s",
    # mouth
    "mar", "mar_max5", "yawns60",
    # head and gaze, relative to calibration
    "abs_yaw", "abs_pitch", "yaw_speed1", "pitch_speed1", "gaze_dev",
    "off_frac5", "off_frac30", "off_run_s", "glances_per_min60",
    "no_face",
]
D = len(FEATURES)


def reference(df: pd.DataFrame) -> dict:
    """Open-eye EAR and neutral pose from the first REF_S seconds with a face."""
    first = df[(df.t <= df.t.iloc[0] + REF_S) & (df.person == 1)]
    if len(first) < 10:
        first = df[df.person == 1]
    if first.empty:
        return {"ear": 0.25, "yaw": 0.0, "pitch": 0.0}
    return {"ear": float(first.ear.quantile(0.7)), "yaw": float(first.yaw.median()),
            "pitch": float(first.pitch.median())}


def rule_flags(df: pd.DataFrame, ref: dict, yaw_max=20.0, pitch_max=15.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample rule detectors the app already has: eyes closed, looking away."""
    face = df.person.values == 1
    closed = face & (df.ear.values < CLOSED * ref["ear"])
    off = (~face) | (np.abs(df.yaw.values - ref["yaw"]) > yaw_max) | (np.abs(df.pitch.values - ref["pitch"]) > pitch_max)
    return closed, off & ~closed


def _run_length(mask: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros(len(mask))
    run = 0.0
    for i, m in enumerate(mask):
        run = run + dt if m else 0.0
        out[i] = run
    return out


def _onsets(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(int)
    return np.r_[m[0], np.diff(m) == 1].astype(float)


def state_features(df: pd.DataFrame, ref: dict | None = None) -> np.ndarray:
    """phi(s_t) for every sample of one session (causal: uses only the past)."""
    ref = ref or reference(df)
    dt = float(np.median(np.diff(df.t.values))) if len(df) > 1 else 0.1
    n = lambda s: max(1, int(round(s / dt)))
    face = (df.person.values == 1)
    ear_rel = np.where(face, df.ear.values / max(ref["ear"], 1e-6), 0.0)
    closed, off = rule_flags(df, ref)
    yaw = np.where(face, df.yaw.values - ref["yaw"], 0.0)
    pitch = np.where(face, df.pitch.values - ref["pitch"], 0.0)
    mar = np.where(face, df.mar.values, 0.0)
    yawn = mar > 0.5

    S = lambda a: pd.Series(a.astype(float))
    r = lambda a, s: S(a).rolling(n(s), min_periods=1)
    c_on = _onsets(closed)
    closure_s = np.where(np.r_[closed[1:] < closed[:-1], False], _run_length(closed, dt), 0.0)  # length at each closure end
    n_cl = r(c_on, 60).sum().values
    f = {
        "bias": np.ones(len(df)),
        "ear_rel": ear_rel,
        "ear_rel_mean2": r(ear_rel, 2).mean().values,
        "ear_rel_slope2": (S(ear_rel) - S(ear_rel).shift(n(2)).bfill()).values / 2.0,
        "closed_frac5": r(closed, 5).mean().values,
        "closed_frac60": r(closed, 60).mean().values,
        "closures_per_min60": n_cl * 60.0 / np.minimum(60.0, (np.arange(len(df)) + 1) * dt),
        "mean_closure_s60": r(closure_s, 60).sum().values / np.maximum(r(closure_s > 0, 60).sum().values, 1),
        "closed_run_s": np.minimum(_run_length(closed, dt), 5.0),
        "mar": mar,
        "mar_max5": r(mar, 5).max().values,
        "yawns60": r(_onsets(yawn), 60).sum().values,
        "abs_yaw": np.abs(yaw) / 30.0,
        "abs_pitch": np.abs(pitch) / 30.0,
        "yaw_speed1": np.abs(S(yaw).diff(n(1)).fillna(0).values) / 30.0,
        "pitch_speed1": np.abs(S(pitch).diff(n(1)).fillna(0).values) / 30.0,
        "gaze_dev": np.where(face, np.abs(df.gaze.values - 0.5), 0.0),
        "off_frac5": r(off, 5).mean().values,
        "off_frac30": r(off, 30).mean().values,
        "off_run_s": np.minimum(_run_length(off, dt), 5.0),
        "glances_per_min60": r(_onsets(off), 60).sum().values * 60.0 / np.minimum(60.0, (np.arange(len(df)) + 1) * dt),
        "no_face": (~face).astype(float),
    }
    return np.column_stack([f[k] for k in FEATURES]).astype(np.float64)


def gamma_for(tau_s: float, dt: float) -> float:
    """Discount so that the value looks about tau_s seconds ahead."""
    return math.exp(-dt / tau_s)


@dataclass
class Scaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Scaler":
        m, s = X.mean(0), X.std(0)
        m[0], s[0] = 0.0, 1.0          # keep the bias column
        s[s < 1e-8] = 1.0
        return cls(m, s)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def lstd_lambda(episodes, gamma: float, lam: float = 0.8, ridge: float = 1.0) -> np.ndarray:
    """Batch LSTD(lambda) for a linear value function.
    episodes: iterable of (Phi [T x d], c [T]) with Phi already scaled.
    Returns w with V(s) = phi(s) @ w, in units of the discounted cumulant
    (multiply by 1-gamma to read it as an expected fraction of time)."""
    A = None
    b = None
    for Phi, c in episodes:
        d = Phi.shape[1]
        if A is None:
            A, b = ridge * np.eye(d), np.zeros(d)
        z = np.zeros(d)
        for t in range(len(Phi) - 1):
            z = gamma * lam * z + Phi[t]
            A += np.outer(z, Phi[t] - gamma * Phi[t + 1])
            b += z * c[t]
    return np.linalg.solve(A, b)


@dataclass
class OnlineTD:
    """TD(lambda) that keeps learning on one person (what the web app runs).
    The cumulant comes from the app's own rule detector, so no labels are needed."""
    w: np.ndarray
    scaler: Scaler
    gamma: float
    lam: float = 0.8
    alpha: float = 0.002
    z: np.ndarray = field(default=None)
    prev: np.ndarray = field(default=None)

    def value(self, phi: np.ndarray) -> float:
        return float(self.scaler(phi) @ self.w) * (1 - self.gamma)

    def step(self, phi: np.ndarray, c_prev: float) -> float:
        """phi: features now; c_prev: cumulant observed for the previous step."""
        x = self.scaler(phi)
        if self.prev is not None:
            if self.z is None:
                self.z = np.zeros_like(x)
            self.z = self.gamma * self.lam * self.z + self.prev
            delta = c_prev + self.gamma * x @ self.w - self.prev @ self.w
            self.w = self.w + self.alpha * delta * self.z
        self.prev = x
        return float(x @ self.w) * (1 - self.gamma)

"""Study-session logic shared by the desktop app (and mirrored in web/app.js).

Feed it one frame of features at a time with `step()`. It keeps the calibration,
the 3 Hz sample stream, the drowsiness score, the distraction episode with
hysteresis, the nudge decisions and their outcomes. No camera, no UI: the
desktop GUI calls it and renders what it returns, and tests drive it directly.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import temporal as T
from .bandit import ACTIONS, LinUCB, context, reward
from .rules import StudyRules
from .simulator import NudgeBudget

REFOCUS_S = 10.0     # a nudge "worked" if the person is back within this many seconds
TOAST_S = 8.0        # nudges stay open this long; resolution waits so "not helpful" counts
DEFAULT_EAR = 0.2 / 0.7   # used if calibration fails: closed-eye threshold = 0.2, as in v2.1

LEVELS = {"low": {"per_hour": 4, "first_s": 8}, "medium": {"per_hour": 6, "first_s": 5}, "high": {"per_hour": 10, "first_s": 5}}


def episode_step(st, away: bool, dt_s: float, t: float) -> bool:
    """One step of the distraction episode with hysteresis (twin of core.js episodeStep).
    st has distracted_s, focused_s, episode_start, back_since, last_return_t, episodes.
    Returns True when an episode just closed."""
    if away:
        if st.distracted_s == 0:
            st.episode_start = t
        st.distracted_s += dt_s
        st.focused_s = 0.0
        st.back_since = None
        return False
    if st.distracted_s <= 0:
        return False
    if st.back_since is None:
        st.back_since = t
    st.focused_s += dt_s
    need = 4.5 if st.distracted_s >= 9 else 1.5 if st.distracted_s >= 3 else 0.5
    if st.focused_s < need:
        return False
    if st.distracted_s >= 5:
        st.episodes.append((st.episode_start, st.distracted_s))
    st.last_return_t, st.back_since = st.back_since, None
    st.distracted_s = st.focused_s = 0.0
    return True


def resolve_pending(pending, t: float, closed: bool, last_return_t) -> list:
    """Twin of core.js resolvePending. A decision worked if the person came back within
    REFOCUS_S (the first focused frame of a return that then held, so the hysteresis hold
    is not charged to the nudge). Nudges wait TOAST_S so 'not helpful' still counts."""
    done = []
    for d in pending:
        if d.resolved:
            continue
        if closed and d.return_t is None and last_return_t is not None and last_return_t >= d.t:
            d.return_t = last_return_t
        if t < d.t + (TOAST_S if d.action != "wait" else 0.0):
            continue
        if d.return_t is not None or t - d.t >= REFOCUS_S + 4.5:
            d.resolved = True
            d.refocused = d.return_t is not None and d.return_t - d.t <= REFOCUS_S
            d.reward = reward(d.action, d.refocused, d.not_helpful)
            done.append(d)
    return done


def rule_v21(distracted_s: float) -> str:
    return "sound" if distracted_s >= 90 else "toast" if distracted_s >= 9 else "wait"


@dataclass
class Decision:
    t: float
    x: list
    action: str
    p: float
    why: str
    condition: str
    distracted_s: float
    reasons: list
    drowsy: float | None
    not_helpful: bool = False
    return_t: float | None = None
    resolved: bool = False
    refocused: bool = False
    reward: float = 0.0


@dataclass
class StudySession:
    condition: str = "C"                  # A none, B rule v2.1, C bandit
    level: str = "medium"
    bandit: LinUCB | None = None
    drowsy_model: dict | None = None
    rules: StudyRules = field(default_factory=StudyRules)
    calib_s: float = 4.0
    seed: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        if self.bandit is None:
            self.bandit = LinUCB(seed=self.seed)
        self.t = 0.0
        self.calib = {"ear": DEFAULT_EAR, "pitch": 0.0, "yaw": 0.0, "done": False, "samples": []}
        self.samples, self.decisions, self.pending, self.episodes, self.timeline = [], [], [], [], []
        self.next_sample_t = 0.0
        self.n_samples = 0                  # total samples taken (the buffer itself is capped)
        self.back_since: float | None = None  # first focused frame of the current return, if any
        self.last_return_t: float | None = None
        self.distracted_s = self.focused_s = self.look_down_s = 0.0
        self.focus_s = self.away_s = 0.0
        self.episode_start = None
        self.reasons: list[str] = []
        self.drowsy: float | None = None
        self.drowsy_hist: list[float] = []
        self.ref = None
        self.budget = NudgeBudget(per_hour=LEVELS[self.level]["per_hour"])
        self.next_decision_at = LEVELS[self.level]["first_s"]
        self.nudge_times: list[float] = []
        self.meeting = self.notes = False

    # ── calibration ──────────────────────────────────────────────────────
    def recalibrate(self):
        self.calib.update(done=False, samples=[], start=self.t)

    def _calibrate(self, f):
        c = self.calib
        c.setdefault("start", self.t)
        if f["ear_score"] > 0:
            c["samples"].append(f)
        if self.t - c["start"] >= self.calib_s:
            if len(c["samples"]) >= 10:
                med = lambda k: float(np.median([s[k] for s in c["samples"]]))
                c.update(ear=med("ear_score"), pitch=med("head_pitch"), yaw=med("head_yaw"))
            c["done"] = True

    # ── main step ────────────────────────────────────────────────────────
    def step(self, dt_s: float, f: dict, phone: int = 0, hour: float | None = None) -> list[Decision]:
        """f: ear_score, mar_score, head_pitch, head_yaw, gaze_ratio, person (1/0).
        ear_score == 0 means no face. Returns decisions made during this step."""
        self.t += dt_s
        if not self.calib["done"]:
            self._calibrate(f)

        if self.t >= self.next_sample_t:
            self.next_sample_t = self.t + 1 / T.HZ
            self.samples.append({k: f[k] for k in ("ear_score", "mar_score", "head_pitch", "head_yaw", "gaze_ratio")})
            self.samples = self.samples[-600:]
            self.n_samples += 1
            if self.n_samples >= 30 and self.n_samples % 3 == 0:
                if len(self.samples) >= T.REF_SAMPLES and self.ref is None:
                    self.ref = T.reference(self.samples)
                ref = self.ref or {k: self.calib[k] for k in ("ear", "pitch", "yaw")}
                feats = T.window_features(self.samples, ref)
                if self.drowsy_model:
                    m = self.drowsy_model
                    z = m["intercept"] + sum(w * (v - mu) / (s or 1) for w, v, mu, s in zip(m["coef"], feats, m["mean"], m["scale"]))
                    self.drowsy = 1 / (1 + math.exp(-max(-500.0, min(500.0, z))))
                    self.drowsy_hist.append(self.drowsy)

        pitch, yaw = f["head_pitch"] - self.calib["pitch"], f["head_yaw"] - self.calib["yaw"]
        open_ = f["ear_score"] >= T.CLOSED * self.calib["ear"]
        self.look_down_s = self.look_down_s + dt_s if (pitch > 20 and abs(yaw) < 20 and open_) else 0.0
        ref = {k: self.calib[k] for k in ("ear", "pitch", "yaw")}
        frame = {"ear_score": f["ear_score"], "head_pitch": f["head_pitch"], "head_yaw": f["head_yaw"], "person": f.get("person", 1)}
        self.reasons = [] if not self.calib["done"] else self.rules.reasons(frame, ref, 0 if self.notes else self.look_down_s, phone)
        if self.notes:
            self.reasons = [r for r in self.reasons if r != "head_down"]
        away = bool(self.reasons)

        closed = episode_step(self, away, dt_s, self.t)
        if closed:
            self.next_decision_at = LEVELS[self.level]["first_s"]
        if self.distracted_s > 0:
            self.away_s += dt_s
        else:
            self.focus_s += dt_s

        made = []
        if self.distracted_s >= self.next_decision_at:
            now = dt.datetime.now()
            made.append(self._decide(hour if hour is not None else now.hour + now.minute / 60))
            self.next_decision_at += 15
        self._resolve(closed)
        return made

    def _decide(self, hour: float) -> Decision:
        since = self.t - self.nudge_times[-1] if self.nudge_times else 600.0
        recent = sum(1 for n in self.nudge_times if self.t - n < 600)
        reasons = ["absent" if r == "no_face" else r for r in self.reasons]
        x = context(self.distracted_s, self.drowsy or 0.0, since, recent, self.t, hour, reasons)
        action, p, why = "wait", 1.0, "policy"
        if self.meeting or self.notes:
            why = "meeting" if self.meeting else "notes"
        elif not self.budget.available(self.t):
            why = "budget"
        elif self.condition == "B":
            action = rule_v21(self.distracted_s)
        elif self.condition == "C":
            action, p = self.bandit.act(x)
        if action != "wait":
            self.budget.spend(self.t)
            self.nudge_times.append(self.t)
        d = Decision(self.t, x.tolist(), action, p, why, self.condition, self.distracted_s, reasons, self.drowsy)
        self.pending.append(d)
        return d

    def _resolve(self, closed: bool):
        for d in resolve_pending(self.pending, self.t, closed, self.last_return_t):
            if d.why == "policy" and d.condition == "C":
                self.bandit.update(np.asarray(d.x), d.action, d.reward)
            self.decisions.append(d)
        self.pending = [d for d in self.pending if not d.resolved]

    # ── summary ──────────────────────────────────────────────────────────
    def summary(self) -> dict:
        eps = self.episodes + ([(self.episode_start, self.distracted_s)] if self.distracted_s >= 5 else [])
        nudges = [d for d in self.decisions if d.action != "wait"]
        return {
            "condition": self.condition, "level": self.level, "minutes": round(self.t / 60, 2),
            "focused_minutes": round(self.focus_s / 60, 2), "distraction_episodes": len(eps),
            "mean_return_s": round(float(np.mean([e[1] for e in eps])), 1) if eps else None,
            "nudges": len(nudges), "nudges_per_hour": round(len(nudges) / self.t * 3600, 1) if self.t else 0,
            "useful_nudge_rate": round(float(np.mean([d.refocused for d in nudges])), 3) if nudges else None,
            "mean_drowsiness": round(float(np.mean(self.drowsy_hist)), 3) if self.drowsy_hist else None,
        }


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None

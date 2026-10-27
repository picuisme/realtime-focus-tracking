"""A simple simulated learner, used ONLY to test the bandit and the off-policy
estimators before real logs exist. Every number it produces is a property of
these assumptions, not of real people; reports must say so.

Each simulated person has:
- a chance of coming back on their own within 10 s (self_return),
- a different responsiveness to each nudge type (heterogeneous on purpose:
  that is the case where personalisation should beat a fixed rule),
- habituation: each nudge in the last 10 minutes makes the next one weaker,
- a chance of pressing "not helpful" after a sound alert.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bandit import ACTIONS, REASONS, context, reward

DECISION_FIRST_S = 5
DECISION_EVERY_S = 15


@dataclass
class NudgeBudget:
    """Token bucket: at most `capacity` nudges in a burst, refilled at
    `per_hour` tokens per hour, never two nudges closer than `min_gap_s`.
    Long-run rate <= per_hour. Same logic runs in the web app."""
    per_hour: float = 6.0
    capacity: float = 2.0
    min_gap_s: float = 60.0
    tokens: float = 1.0
    last_t: float = 0.0
    last_nudge: float = -1e9

    def _refill(self, t: float) -> None:
        self.tokens = min(self.capacity, self.tokens + (t - self.last_t) * self.per_hour / 3600)
        self.last_t = t

    def available(self, t: float) -> bool:
        self._refill(t)
        return self.tokens >= 1 and t - self.last_nudge >= self.min_gap_s

    def spend(self, t: float) -> None:
        self._refill(t)
        self.tokens -= 1
        self.last_nudge = t


@dataclass
class SimUser:
    self_return: float
    resp: dict
    habituation: float
    annoy_sound: float
    drowsy_bias: float

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "SimUser":
        return cls(
            self_return=rng.uniform(0.10, 0.40),
            resp={"wait": 0.0, "toast": rng.uniform(0.05, 0.70), "sound": rng.uniform(0.20, 0.80),
                  "break": rng.uniform(0.05, 0.30), "dim": rng.uniform(0.05, 0.50)},
            habituation=rng.uniform(0.75, 0.95),
            annoy_sound=rng.uniform(0.0, 0.5),
            drowsy_bias=rng.uniform(0.0, 0.6),
        )

    def p_refocus(self, action: str, reason: str, drowsy: float, recent_nudges: int) -> float:
        r = self.resp[action]
        if action == "break" and reason == "eyes_closed":
            r = min(0.9, r + 0.5 * drowsy)
        r *= self.habituation ** recent_nudges
        q = self.self_return * (0.5 if reason == "eyes_closed" else 1.0)
        return 1 - (1 - q) * (1 - r)


def run_session(user: SimUser, choose, rng: np.random.Generator, minutes: int = 25, rate_per_min: float = 0.25,
                hour: float = 20.0, budget: "NudgeBudget | None" = None, learn=None) -> dict:
    """choose(x, info) -> (action, propensity). learn(x, a, r) optional.
    Returns per-session stats and the decision log."""
    budget = budget or NudgeBudget()
    t, nudges, log, times = 0.0, [], [], []
    horizon = minutes * 60
    while True:
        t += rng.exponential(60 / rate_per_min)
        if t >= horizon:
            break
        reason = REASONS[rng.choice(4, p=[0.35, 0.35, 0.2, 0.1])]
        drowsy = float(np.clip(rng.normal(user.drowsy_bias if reason == "eyes_closed" else 0.1, 0.15), 0, 1))
        elapsed = DECISION_FIRST_S
        while True:
            now = t + elapsed
            recent = [n for n in nudges if now - n < 600]
            capped = not budget.available(now)
            since = now - nudges[-1] if nudges else 600.0
            x = context(elapsed, drowsy, since, len(recent), now, hour, [reason])
            a, p = ("wait", 1.0) if capped else choose(x, {"elapsed": elapsed})
            ok = rng.random() < user.p_refocus(a, reason, drowsy, len(recent))
            nh = a == "sound" and rng.random() < user.annoy_sound
            r = reward(a, ok, nh)
            if a != "wait":
                nudges.append(now)
                budget.spend(now)
            if not capped:
                log.append({"x": x, "a": a, "p": p, "r": r, "refocus": ok, "nudge": a != "wait",
                            "state": (reason, drowsy, len(recent))})
                if learn:
                    learn(x, a, r)
            if ok:
                times.append(elapsed + rng.uniform(0, 10))
                break
            elapsed += DECISION_EVERY_S
            if elapsed > 300:
                times.append(elapsed)
                break
        t += times[-1]
    useful = [l["refocus"] for l in log if l["nudge"]]
    return {"log": log, "minutes": minutes, "nudges": len(nudges),
            "mean_return_s": float(np.mean(times)) if times else float("nan"), "episodes": len(times),
            "useful_rate": float(np.mean(useful)) if useful else float("nan"),
            "reward": float(np.mean([l["r"] for l in log])) if log else float("nan")}


# Baseline policies -----------------------------------------------------------

def never(x, info):
    return "wait", 1.0


def rule_v21(x, info):
    """What v2.1 effectively did after its overrides: toast from 9 s, sound from 90 s."""
    e = info["elapsed"]
    return ("sound" if e >= 90 else "toast" if e >= 9 else "wait"), 1.0


def rule_v21_probs(x) -> dict:
    """rule_v21 as a target policy for off-policy evaluation (elapsed is x[1] * 180)."""
    e = x[1] * 180
    a = "sound" if e >= 90 else "toast" if e >= 9 else "wait"
    return {a: 1.0}


def expected_reward(user: SimUser, action: str, state) -> float:
    """Ground-truth expected reward of an action in a logged state (simulation only)."""
    reason, drowsy, recent = state
    p = user.p_refocus(action, reason, drowsy, recent)
    return p - {"wait": 0.0, "toast": 0.10, "sound": 0.30, "break": 0.15, "dim": 0.20}[action] \
        - (0.5 * user.annoy_sound if action == "sound" else 0.0)

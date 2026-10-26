"""Nudge selection as a contextual bandit, plus off-policy evaluation.

Why a bandit and not DQN: a nudge changes what the person does in the next
seconds, then the episode is over. There is no long-horizon state the agent
controls, so a one-step contextual bandit is the honest formulation, and its
logged propensities make off-policy evaluation possible.

Decision point: the person has been distracted for >= 5 s (then every 15 s
while it lasts), outside meeting / note-taking mode, under the safety cap.
Reward (observed 10 s later): 1 if they are back on task, minus the action's
annoyance cost, minus 0.5 if they press "not helpful".

web/core.js implements the same LinUCB arithmetic (Sherman-Morrison updates),
so a policy trained here can be shipped to the browser and keep learning there.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

ACTIONS = ["wait", "toast", "sound", "break", "dim"]
COST = {"wait": 0.0, "toast": 0.10, "sound": 0.30, "break": 0.15, "dim": 0.20}
REASONS = ["phone", "head_turned", "eyes_closed", "absent"]
CONTEXT = ["bias", "distracted_min", "drowsy", "since_nudge", "nudges_10min", "session_h",
           "hour_sin", "hour_cos"] + [f"r_{r}" for r in REASONS]
D = len(CONTEXT)


def context(distracted_s: float, drowsy: float, since_nudge_s: float, nudges_10min: int,
            session_s: float, hour: float, reasons: Sequence[str]) -> np.ndarray:
    a = 2 * math.pi * hour / 24
    x = [1.0, min(distracted_s / 60, 1.0), float(drowsy), min(since_nudge_s / 600, 1.0),
         min(nudges_10min / 6, 1.0), min(session_s / 7200, 1.0), math.sin(a), math.cos(a)]
    x += [1.0 if r in reasons else 0.0 for r in REASONS]
    return np.asarray(x, dtype=np.float64)


def reward(action: str, refocused: bool, not_helpful: bool = False) -> float:
    return float(refocused) - COST[action] - (0.5 if not_helpful else 0.0)


@dataclass
class LinUCB:
    """Disjoint LinUCB with epsilon-uniform exploration (so every action keeps a
    known, non-zero propensity for off-policy evaluation)."""
    alpha: float = 0.5
    epsilon: float = 0.1
    ridge: float = 1.0
    allowed: Sequence[str] = tuple(ACTIONS)
    seed: int = 0
    A_inv: dict = field(default_factory=dict)
    b: dict = field(default_factory=dict)
    offset: dict = field(default_factory=dict)   # fixed prior weights; the model learns the residual

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        for a in ACTIONS:
            self.A_inv.setdefault(a, np.eye(D) / self.ridge)
            self.b.setdefault(a, np.zeros(D))
            self.offset.setdefault(a, np.zeros(D))

    def scores(self, x: np.ndarray, explore: bool = True) -> dict:
        out = {}
        for a in self.allowed:
            theta = self.offset[a] + self.A_inv[a] @ self.b[a]
            bonus = self.alpha * math.sqrt(max(float(x @ self.A_inv[a] @ x), 0.0)) if explore else 0.0
            out[a] = float(theta @ x) + bonus
        return out

    def probs(self, x: np.ndarray, explore: bool = True) -> dict:
        s = self.scores(x, explore)
        best = max(s, key=s.get)
        k = len(s)
        eps = self.epsilon if explore else 0.0
        return {a: (1 - eps) * (a == best) + eps / k for a in s}

    def act(self, x: np.ndarray) -> tuple[str, float]:
        p = self.probs(x)
        acts = list(p)
        a = acts[self.rng.choice(len(acts), p=np.array([p[k] for k in acts]))]
        return a, p[a]

    def update(self, x: np.ndarray, a: str, r: float) -> None:
        Ai = self.A_inv[a]
        Ax = Ai @ x
        self.A_inv[a] = Ai - np.outer(Ax, Ax) / (1.0 + x @ Ax)   # Sherman-Morrison
        self.b[a] = self.b[a] + (r - float(self.offset[a] @ x)) * x

    def theta(self, a: str) -> np.ndarray:
        return self.offset[a] + self.A_inv[a] @ self.b[a]

    def personalised(self, ridge: float = 5.0, alpha: float = 0.2, seed: int = 0) -> "LinUCB":
        """A per-person model: this (population) model's weights become a fixed
        offset, and the new model learns only how this person differs from it.
        A strong ridge keeps it close to the population until evidence builds up."""
        m = LinUCB(alpha=alpha, epsilon=self.epsilon, ridge=ridge, allowed=self.allowed, seed=seed)
        for a in ACTIONS:
            m.offset[a] = self.theta(a).copy()
        return m

    def to_json(self) -> str:
        return json.dumps({"alpha": self.alpha, "epsilon": self.epsilon, "context": CONTEXT, "actions": ACTIONS,
                           "A_inv": {a: self.A_inv[a].tolist() for a in ACTIONS},
                           "b": {a: self.b[a].tolist() for a in ACTIONS},
                           "offset": {a: self.offset[a].tolist() for a in ACTIONS}})

    @classmethod
    def from_json(cls, s: str, **kw) -> "LinUCB":
        d = json.loads(s)
        m = cls(alpha=d["alpha"], epsilon=d["epsilon"], **kw)
        m.A_inv = {a: np.asarray(v) for a, v in d["A_inv"].items()}
        m.b = {a: np.asarray(v) for a, v in d["b"].items()}
        m.offset = {a: np.asarray(v) for a, v in d.get("offset", {}).items()} or m.offset
        return m


# ── off-policy evaluation ────────────────────────────────────────────────────

Policy = Callable[[np.ndarray], dict]   # x -> {action: prob}


def ips(logs: Sequence[dict], target: Policy) -> float:
    """logs: dicts with x, a, p (logging propensity), r."""
    return float(np.mean([target(l["x"]).get(l["a"], 0.0) / l["p"] * l["r"] for l in logs]))


def snips(logs: Sequence[dict], target: Policy) -> float:
    w = np.array([target(l["x"]).get(l["a"], 0.0) / l["p"] for l in logs])
    r = np.array([l["r"] for l in logs])
    return float((w * r).sum() / w.sum()) if w.sum() > 0 else float("nan")


def doubly_robust(logs: Sequence[dict], target: Policy, ridge: float = 1.0, folds: int = 2, seed: int = 0) -> float:
    """DR with a per-action ridge reward model, cross-fitted so the model never
    scores the rows it was fit on."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, size=len(logs))
    est = []
    for k in range(folds):
        train = [l for l, f in zip(logs, fold) if f != k]
        test = [l for l, f in zip(logs, fold) if f == k]
        W = {}
        for a in ACTIONS:
            X = np.array([l["x"] for l in train if l["a"] == a]).reshape(-1, D)
            y = np.array([l["r"] for l in train if l["a"] == a])
            W[a] = np.linalg.solve(X.T @ X + ridge * np.eye(D), X.T @ y) if len(y) else np.zeros(D)
        for l in test:
            pi = target(l["x"])
            dm = sum(p * float(W[a] @ l["x"]) for a, p in pi.items())
            est.append(dm + pi.get(l["a"], 0.0) / l["p"] * (l["r"] - float(W[l["a"]] @ l["x"])))
    return float(np.mean(est))

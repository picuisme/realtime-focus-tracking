"""Evaluation metrics: event-level detection, false alarms per hour, latency,
and bootstrap confidence intervals grouped by person."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


@dataclass
class EventReport:
    events: int
    detected: int
    recall: float
    false_alarms: int
    negative_hours: float
    false_alarms_per_hour: float
    median_latency_s: float

    def as_dict(self):
        return self.__dict__.copy()


def event_metrics(gt: np.ndarray, pred: np.ndarray, hz: float = 3.0, min_event_s: float = 5.0,
                  tolerance_s: float = 2.0) -> EventReport:
    """gt, pred: binary arrays on the same time grid.
    - An event = a run of gt == 1 lasting >= min_event_s.
    - Detected if pred fires anywhere in [start - tolerance, end + tolerance].
    - Latency = first firing time - start (0 if it fired at or just before the start).
    - A false alarm = a pred onset that is not within tolerance of any gt == 1 sample."""
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    if gt.shape != pred.shape:
        raise ValueError(f"gt and pred differ in length: {gt.shape} vs {pred.shape}")
    tol = int(round(tolerance_s * hz))
    events = [(s, e) for s, e in runs(gt) if (e - s) / hz >= min_event_s]
    detected, lat = 0, []
    for s, e in events:
        lo = max(0, s - tol)
        hit = np.where(pred[lo: min(len(pred), e + tol)])[0]
        if len(hit):
            detected += 1
            lat.append(max(0, hit[0] + lo - s) / hz)   # an alarm already on (or within tolerance before) = 0 latency
    near_pos = np.zeros_like(gt)
    for s, e in runs(gt):
        near_pos[max(0, s - tol): min(len(gt), e + tol)] = True
    fa = sum(1 for s, _ in runs(pred) if not near_pos[s])
    neg_h = float((~gt).sum()) / hz / 3600.0
    return EventReport(
        events=len(events), detected=detected,
        recall=detected / len(events) if events else float("nan"),
        false_alarms=fa, negative_hours=neg_h,
        false_alarms_per_hour=fa / neg_h if neg_h > 0 else float("nan"),
        median_latency_s=float(np.median(lat)) if lat else float("nan"),
    )


def grouped_bootstrap_ci(y, score, groups, metric=roc_auc_score, n=1000, seed=0, alpha=0.05):
    """Point estimate and 95 % CI, resampling whole people (not rows)."""
    rng = np.random.default_rng(seed)
    y, score, groups = map(np.asarray, (y, score, groups))
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    point = metric(y, score)
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx_by[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[pick])) < 2:
            continue
        vals.append(metric(y[pick], score[pick]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if vals else (np.nan, np.nan)
    return float(point), float(lo), float(hi)

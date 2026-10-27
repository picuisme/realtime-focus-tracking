import numpy as np

from focus.metrics import event_metrics, grouped_bootstrap_ci, runs


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1])) == [(1, 3), (4, 5)]


def test_event_metrics_counts_events_latency_and_false_alarms():
    hz = 3
    gt = np.zeros(3600 * hz, bool)
    gt[300:330] = True          # 10 s event, detected 1 s late
    gt[900:912] = True          # 4 s event: shorter than min_event_s, ignored
    gt[2000:2060] = True        # 20 s event, missed
    pred = np.zeros_like(gt)
    pred[303:330] = True
    pred[5000:5010] = True      # false alarm
    pred[6000:6003] = True      # false alarm
    r = event_metrics(gt, pred, hz=hz, min_event_s=5)
    assert r.events == 2 and r.detected == 1 and r.recall == 0.5
    assert r.median_latency_s == 1.0
    assert r.false_alarms == 2
    assert abs(r.false_alarms_per_hour - 2 / r.negative_hours) < 1e-9


def test_alarm_next_to_an_event_is_not_false():
    gt = np.zeros(600, bool); gt[100:130] = True
    pred = np.zeros(600, bool); pred[131:135] = True   # within 2 s tolerance after the end
    assert event_metrics(gt, pred).false_alarms == 0


def test_grouped_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = y + rng.normal(0, 1, 400)
    g = np.repeat(np.arange(20), 20)
    p, lo, hi = grouped_bootstrap_ci(y, s, g, n=200)
    assert lo <= p <= hi and 0.5 < p < 1


def test_short_series_alignment():
    gt = np.array([0, 0, 0, 0, 1], bool)
    pred = np.array([1, 0, 0, 0, 0], bool)       # 4 samples before the positive, tolerance 2 s = 6 samples
    assert event_metrics(gt, pred, min_event_s=0).false_alarms == 0


def test_alarm_just_before_event_counts_as_detection():
    gt = np.zeros(300, bool); gt[100:130] = True
    pred = np.zeros(300, bool); pred[97:99] = True
    r = event_metrics(gt, pred, min_event_s=5)
    assert r.detected == 1 and r.false_alarms == 0 and r.median_latency_s == 0

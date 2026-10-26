import numpy as np

from focus.rules import DriverRules, StudyRules
from focus.temporal import FEATURES, reference, window_features

REF = {"ear": 0.3, "pitch": 0.0, "yaw": 0.0}


def frame(**kw):
    f = {"ear_score": 0.3, "head_pitch": 0.0, "head_yaw": 0.0, "person": 1}
    f.update(kw)
    return f


def test_study_rules_are_relative_to_the_person():
    r = StudyRules()
    assert r.reasons(frame(head_yaw=50), REF, 0, 0) == ["head_turned"]
    # same absolute yaw, but this person's neutral pose is turned 30 degrees (camera to the side)
    assert r.reasons(frame(head_yaw=50), {**REF, "yaw": 30}, 0, 0) == []
    # small eyes: 0.18 is closed for someone at 0.3, open for someone at 0.2
    assert "eyes_closed" in r.reasons(frame(ear_score=0.18), REF, 0, 0)
    assert r.reasons(frame(ear_score=0.18), {**REF, "ear": 0.2}, 0, 0) == []


def test_note_taking_allowed_for_30s_only():
    r = StudyRules()
    assert r.reasons(frame(head_pitch=35), REF, 10, 0) == []
    assert r.reasons(frame(head_pitch=35), REF, 31, 0) == ["head_down"]
    assert "head_down" in r.reasons(frame(head_pitch=35), REF, 10, 1)


def test_driver_microsleep_wins_and_resets():
    d = DriverRules()
    lv = None
    for _ in range(16):
        lv = d.step(0.1, closed=True, off_road=False, phone=False, perclos=0, perclos_ready=False, yawns_5min=0)
    assert lv == ("alarm", "microsleep")
    assert d.step(0.1, False, False, False, 0, False, 0) == ("ok", "alert")


def test_window_features_shape_and_no_label_inputs():
    rng = np.random.default_rng(1)
    buf = [{"ear_score": float(rng.uniform(.2, .35)), "mar_score": 0.1, "head_pitch": 0.0, "head_yaw": 0.0,
            "gaze_ratio": 0.5, "is_distracted_label": 1, "consecutive_frames": 99} for _ in range(200)]
    f1 = window_features(buf, reference(buf))
    for b in buf:   # labels must not change anything
        b["is_distracted_label"], b["consecutive_frames"] = 0, 0
    assert len(f1) == len(FEATURES) and f1 == window_features(buf, reference(buf))


def test_perclos_counts_closed_fraction():
    buf = [{"ear_score": 0.3 if i % 4 else 0.1, "mar_score": 0.0, "head_pitch": 0.0, "head_yaw": 0.0,
            "gaze_ratio": 0.5} for i in range(180)]
    f = dict(zip(FEATURES, window_features(buf, REF)))
    assert abs(f["perclos60"] - 0.25) < 1e-9 and f["closures_per_min60"] == 45

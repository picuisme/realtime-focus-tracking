from focus.session import StudySession

FOCUS = {"ear_score": 0.30, "mar_score": 0.05, "head_pitch": 5.0, "head_yaw": 2.0, "gaze_ratio": 0.5, "person": 1}
AWAY = {**FOCUS, "head_yaw": 60.0}
DT = 0.1


def run(sess, frame, seconds, **kw):
    out = []
    for _ in range(int(round(seconds / DT))):
        out += sess.step(DT, frame, hour=20, **kw)
    return out


def test_rule_condition_waits_then_toasts_and_counts_the_episode():
    s = StudySession(condition="B", seed=1)
    run(s, FOCUS, 5)                      # calibration + settle
    d = run(s, AWAY, 30)
    assert [x.action for x in d] == ["wait", "toast"]      # decisions at 5 s and 20 s
    run(s, FOCUS, 10)
    sm = s.summary()
    assert sm["distraction_episodes"] == 1 and sm["nudges"] == 1
    assert s.decisions[-1].refocused is False    # still away 10 s after the toast


def test_short_glance_is_not_an_episode():
    s = StudySession(condition="B")
    run(s, FOCUS, 5)
    assert run(s, AWAY, 2) == []
    run(s, FOCUS, 3)
    assert s.summary()["distraction_episodes"] == 0


def test_calibration_makes_rules_relative():
    s = StudySession(condition="B")
    turned = {**FOCUS, "head_yaw": 45.0}          # camera placed to the side: 45 deg is this person's normal
    run(s, turned, 5)
    assert run(s, turned, 30) == [] and s.reasons == []


def test_meeting_mode_suppresses_nudges():
    s = StudySession(condition="B")
    run(s, FOCUS, 5)
    s.meeting = True
    d = run(s, AWAY, 30)
    assert d and all(x.action == "wait" and x.why == "meeting" for x in d)


def test_bandit_condition_logs_propensity_and_learns():
    s = StudySession(condition="C", seed=3)
    run(s, FOCUS, 5)
    before = {a: s.bandit.b[a].copy() for a in s.bandit.b}
    for _ in range(3):
        run(s, AWAY, 22)
        run(s, FOCUS, 8)
    pol = [d for d in s.decisions if d.why == "policy"]
    assert pol and all(0 < d.p <= 1 for d in pol)
    assert any((s.bandit.b[a] != before[a]).any() for a in before)


def test_budget_caps_nudges():
    s = StudySession(condition="B", level="low")
    run(s, FOCUS, 5)
    for _ in range(20):
        run(s, AWAY, 25)
        run(s, FOCUS, 6)
    assert s.summary()["nudges"] <= 2 + 4 * s.t / 3600 + 1e-9

"""focus/session.py episode_step + resolve_pending must behave exactly like web/core.js."""
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from focus.session import Decision, episode_step, resolve_pending

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def script(seed):
    rng = np.random.default_rng(seed)
    frames, away = [], False
    for _ in range(3000):                      # 300 s at 10 Hz, random switches
        if rng.random() < 0.02:
            away = not away
        frames.append([int(away), int(rng.integers(0, 4)) if rng.random() < 0.01 else None])
    decisions = sorted([[round(float(t), 1), a] for t, a in zip(rng.uniform(5, 280, 25), rng.choice(["wait", "toast", "sound"], 25))])
    return {"dt": 0.1, "frames": frames, "decisions": decisions}


def run_python(c):
    st = SimpleNamespace(distracted_s=0.0, focused_s=0.0, episode_start=None, back_since=None, last_return_t=None, episodes=[])
    pending, resolved, closes, t, di = [], [], [], 0.0, 0
    for away, nh in c["frames"]:
        t = round(t + c["dt"], 9)
        closed = episode_step(st, bool(away), c["dt"], t)
        if closed:
            closes.append([t, st.last_return_t])
        while di < len(c["decisions"]) and c["decisions"][di][0] <= t + 1e-9:
            dt_, a = c["decisions"][di]
            pending.append(Decision(dt_, [], a, 1.0, "policy", "B", 0.0, [], None))
            di += 1
        if nh is not None and nh < len(pending):
            pending[nh].not_helpful = True
        resolved += [[d.t, d.action, d.refocused, d.reward, d.return_t] for d in resolve_pending(pending, t, closed, st.last_return_t)]
    return {"closes": closes, "episodes": [list(e) for e in st.episodes], "resolved": resolved}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_session_logic_matches_js(tmp_path, seed):
    c = script(seed)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(c))
    js = json.loads(subprocess.check_output(["node", str(ROOT / "tests/session_parity.mjs"), str(p)], text=True))
    py = run_python(c)
    assert len(py["resolved"]) == len(js["resolved"]) > 5
    assert np.allclose(np.array(py["closes"], float), np.array(js["closes"], float))
    assert np.allclose(np.array(py["episodes"], float).reshape(-1, 2), np.array(js["episodes"], float).reshape(-1, 2))
    for a, b in zip(py["resolved"], js["resolved"]):
        assert a[0] == pytest.approx(b[0]) and a[1] == b[1] and a[2] == b[2] and a[3] == pytest.approx(b[3])
        assert (a[4] is None and b[4] is None) or a[4] == pytest.approx(b[4])

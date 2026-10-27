// core.js — pure logic, no DOM. Every function here has a Python twin in focus/
// and tests/test_parity.py checks they agree.

// ── per-frame features (focus/features.py) ───────────────────────────────
// Ratios are computed as if the frame were 640x480, the setup the thresholds were
// tuned on. MediaPipe normalises x and y separately, so on 16:9 video EAR would be
// ~1.33x larger without this.
export function to43(lm, w, h) {
  const k = (w / h) * 0.75;
  if (Math.abs(k - 1) < 1e-3) return lm;
  return lm.map(p => ({ x: p.x * k, y: p.y, z: p.z }));
}
const d2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

export function ear(lm) {
  const one = ix => {
    const p = ix.map(i => lm[i]);
    const h = d2(p[0], p[3]);
    return h === 0 ? 0 : (d2(p[1], p[5]) + d2(p[2], p[4])) / (2 * h);
  };
  return (one([263, 387, 385, 362, 380, 373]) + one([33, 160, 158, 133, 153, 144])) / 2;
}

export function mar(lm) {
  const h = d2(lm[61], lm[291]);
  return h === 0 ? 0 : d2(lm[13], lm[14]) / h;
}

export function browDist(lm, w = 640, h = 480) {
  const a = lm[105], b = lm[159];
  return Math.hypot((a.x - b.x) * w, (a.y - b.y) * h);
}

// Iris position inside each eye from its image-left corner; 0.5 = centred.
// 468 sits in the eye with corners 33/133, 473 in the eye with corners 362/263.
export function gazeRatio(lm) {
  if (lm.length < 478) return 0.5;
  const g = (iris, c0, c1) => {
    const w = d2(lm[c0], lm[c1]);
    return w === 0 ? 0.5 : d2(lm[c0], lm[iris]) / w;
  };
  return (g(468, 33, 133) + g(473, 362, 263)) / 2;
}

// Head pose (degrees) from MediaPipe's column-major 4x4 transformation matrix.
// pitch > 0 = looking down. No ±180° wrap-around.
export function poseFromMatrix(m, invertPitch = false) {
  const r = (i, j) => m[j * 4 + i];
  const deg = 180 / Math.PI;
  const fwd = [r(0, 2), r(1, 2), r(2, 2)];
  const pitch = Math.asin(Math.max(-1, Math.min(1, -fwd[1]))) * deg;
  return { pitch: invertPitch ? -pitch : pitch, yaw: Math.atan2(fwd[0], fwd[2]) * deg, roll: Math.atan2(r(1, 0), r(0, 0)) * deg };
}

// ── temporal features (focus/temporal.py) ────────────────────────────────
export const HZ = 3, W_SHORT = 60, W_LONG = 180, REF_SAMPLES = 60, CLOSED = 0.7, YAWN_MAR = 0.5;
export const FEATURES = ['ear_rel_mean20', 'ear_rel_std20', 'perclos60', 'closures_per_min60', 'longest_closure60',
  'mar_mean60', 'yawn_frac60', 'pitch_rel_mean20', 'pitch_std20', 'yaw_std20', 'abs_yaw_rel_mean20', 'gaze_std20',
  'face_missing60'];

export const median = a => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const mean = a => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
const std = a => { if (!a.length) return 0; const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) ** 2))); };

export function reference(samples) {
  let face = samples.slice(0, REF_SAMPLES).filter(s => s.ear_score > 0);
  if (face.length < 10) face = samples.filter(s => s.ear_score > 0);
  if (!face.length) face = [{ ear_score: 0.28, head_pitch: 0, head_yaw: 0 }];
  return { ear: Math.max(median(face.map(s => s.ear_score)), 1e-3), pitch: median(face.map(s => s.head_pitch)), yaw: median(face.map(s => s.head_yaw)) };
}

export function windowFeatures(buf, ref) {
  const long = buf.slice(-W_LONG), short = long.slice(-W_SHORT);
  const lf = long.filter(s => s.ear_score > 0), sf = short.filter(s => s.ear_score > 0);
  const earL = lf.map(s => s.ear_score / ref.ear), earS = sf.map(s => s.ear_score / ref.ear);
  const closed = earL.map(e => (e < CLOSED ? 1 : 0));
  let closures = 0, longest = 0, run = 0;
  closed.forEach((c, i) => {
    if (c && (i === 0 || !closed[i - 1])) closures++;
    run = c ? run + 1 : 0; longest = Math.max(longest, run);
  });
  const minutes = Math.max(long.length / HZ / 60, 1 / 60);
  const pitchRel = sf.map(s => s.head_pitch - ref.pitch), yawRel = sf.map(s => s.head_yaw - ref.yaw);
  const mars = lf.map(s => s.mar_score);
  return [
    earS.length ? mean(earS) : 1, std(earS), mean(closed), closures / minutes, longest / HZ,
    mean(mars), mean(mars.map(m => (m > YAWN_MAR ? 1 : 0))), mean(pitchRel), std(pitchRel), std(yawRel),
    mean(yawRel.map(Math.abs)), std(sf.map(s => s.gaze_ratio)), long.length ? 1 - lf.length / long.length : 1,
  ];
}

// Logistic model exported by scripts/train_drowsiness.py (web/models/drowsiness.json)
export function drowsiness(model, f) {
  let z = model.intercept;
  for (let i = 0; i < f.length; i++) z += model.coef[i] * (f[i] - model.mean[i]) / (model.scale[i] || 1);
  return 1 / (1 + Math.exp(-z));
}

// ── nudge policy (focus/bandit.py) ───────────────────────────────────────
export const ACTIONS = ['wait', 'toast', 'sound', 'break', 'dim'];
export const COST = { wait: 0, toast: 0.1, sound: 0.3, break: 0.15, dim: 0.2 };
export const REASONS = ['phone', 'head_turned', 'eyes_closed', 'absent'];
export const D = 8 + REASONS.length;

export function context(distractedS, drowsy, sinceNudgeS, nudges10min, sessionS, hour, reasons) {
  const a = 2 * Math.PI * hour / 24;
  return [1, Math.min(distractedS / 180, 1), drowsy, Math.min(sinceNudgeS / 600, 1), Math.min(nudges10min / 6, 1),
    Math.min(sessionS / 7200, 1), Math.sin(a), Math.cos(a), ...REASONS.map(r => (reasons.includes(r) ? 1 : 0))];
}

export const reward = (action, refocused, notHelpful = false) => (refocused ? 1 : 0) - COST[action] - (notHelpful ? 0.5 : 0);

const matVec = (M, v) => M.map(row => row.reduce((s, m, j) => s + m * v[j], 0));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const eye = (n, s) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? s : 0)));

export class LinUCB {
  constructor({ alpha = 0.5, epsilon = 0.1, ridge = 1, allowed = ACTIONS, A_inv, b, offset, rand = Math.random } = {}) {
    Object.assign(this, { alpha, epsilon, allowed, rand });
    this.A_inv = A_inv ?? Object.fromEntries(ACTIONS.map(a => [a, eye(D, 1 / ridge)]));
    this.b = b ?? Object.fromEntries(ACTIONS.map(a => [a, Array(D).fill(0)]));
    this.offset = offset ?? Object.fromEntries(ACTIONS.map(a => [a, Array(D).fill(0)]));
  }
  static fromJSON(o, extra = {}) { return new LinUCB({ alpha: o.alpha, epsilon: o.epsilon, A_inv: o.A_inv, b: o.b, offset: o.offset, ...extra }); }
  toJSON() { return { alpha: this.alpha, epsilon: this.epsilon, A_inv: this.A_inv, b: this.b, offset: this.offset }; }
  theta(a) { const t = matVec(this.A_inv[a], this.b[a]); return t.map((v, i) => v + this.offset[a][i]); }
  scores(x, explore = true) {
    const out = {};
    for (const a of this.allowed) {
      const bonus = explore ? this.alpha * Math.sqrt(Math.max(dot(x, matVec(this.A_inv[a], x)), 0)) : 0;
      out[a] = dot(this.theta(a), x) + bonus;
    }
    return out;
  }
  probs(x, explore = true) {
    const s = this.scores(x, explore), acts = Object.keys(s);
    const best = acts.reduce((m, a) => (s[a] > s[m] ? a : m), acts[0]);
    const eps = explore ? this.epsilon : 0;
    return Object.fromEntries(acts.map(a => [a, (1 - eps) * (a === best ? 1 : 0) + eps / acts.length]));
  }
  act(x) {
    const p = this.probs(x), acts = Object.keys(p);
    let u = this.rand(), acc = 0;
    for (const a of acts) { acc += p[a]; if (u < acc) return [a, p[a]]; }
    return [acts.at(-1), p[acts.at(-1)]];
  }
  update(x, a, r) {
    const Ai = this.A_inv[a], Ax = matVec(Ai, x), den = 1 + dot(x, Ax);
    this.A_inv[a] = Ai.map((row, i) => row.map((v, j) => v - Ax[i] * Ax[j] / den));
    const resid = r - dot(this.offset[a], x);
    this.b[a] = this.b[a].map((v, i) => v + resid * x[i]);
  }
}

// Token bucket: long-run <= perHour nudges, bursts of `capacity`, minGapS apart.
export class NudgeBudget {
  constructor({ perHour = 6, capacity = 2, minGapS = 60, tokens = 1 } = {}) {
    Object.assign(this, { perHour, capacity, minGapS, tokens, lastT: 0, lastNudge: -1e9 });
  }
  refill(t) { this.tokens = Math.min(this.capacity, this.tokens + (t - this.lastT) * this.perHour / 3600); this.lastT = t; }
  available(t) { this.refill(t); return this.tokens >= 1 && t - this.lastNudge >= this.minGapS; }
  spend(t) { this.refill(t); this.tokens -= 1; this.lastNudge = t; }
}

// What v2.1 effectively did after its overrides (condition B of the study).
export const ruleV21 = distractedS => (distractedS >= 90 ? 'sound' : distractedS >= 9 ? 'toast' : 'wait');

// ── driver alarms (focus/rules.py DriverRules) ───────────────────────────
export const DRIVER = { microsleepS: 1.5, offRoadS: 2, phoneS: 1, perclosWarn: 0.15, perclosAlarm: 0.3, yawnWarnCount: 3, drowsyWarn: 0.7 };

// ── study-session pieces shared with focus/session.py (tested for parity) ──
export const REFOCUS_S = 10, TOAST_S = 8;

// One step of the distraction episode with hysteresis. st: {distractedS, focusedS,
// episodeStart, backSince, lastReturnT, episodes}. Returns true when an episode just closed.
export function episodeStep(st, away, dt, t) {
  if (away) {
    if (st.distractedS === 0) st.episodeStart = t;
    st.distractedS += dt; st.focusedS = 0; st.backSince = null;
    return false;
  }
  if (st.distractedS <= 0) return false;
  st.backSince ??= t;
  st.focusedS += dt;
  const need = st.distractedS >= 9 ? 4.5 : st.distractedS >= 3 ? 1.5 : 0.5;
  if (st.focusedS < need) return false;
  if (st.distractedS >= 5) st.episodes.push({ start: st.episodeStart, dur: st.distractedS });
  st.lastReturnT = st.backSince; st.backSince = null;
  st.distractedS = 0; st.focusedS = 0;
  return true;
}

// Resolve pending decisions. A decision "worked" if the person came back within REFOCUS_S
// (first focused frame of a return that then held). Nudges wait for their toast to close.
// Returns the decisions resolved now (each with refocused and reward set).
export function resolvePending(pending, t, closed, lastReturnT) {
  const done = [];
  for (const d of pending) {
    if (d.resolved) continue;
    if (closed && d.returnT == null && lastReturnT != null && lastReturnT >= d.t) d.returnT = lastReturnT;
    if (t < d.t + (d.action !== 'wait' ? TOAST_S : 0)) continue;
    if (d.returnT != null || t - d.t >= REFOCUS_S + 4.5) {
      d.resolved = true;
      d.refocused = d.returnT != null && d.returnT - d.t <= REFOCUS_S;
      d.reward = reward(d.action, d.refocused, d.notHelpful);
      done.push(d);
    }
  }
  return done;
}

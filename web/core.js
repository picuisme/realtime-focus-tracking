// core.js — pure logic ported from focus_env.py / demo_webcam.py.
// No DOM, no camera: everything here can be unit-tested in Node.

export const WINDOW_SIZE = 10;
export const POLICY_HZ = 3;              // training data assumed 3 fps (focus_env.FPS)
export const MAX_CONSECUTIVE = 300;
export const MAX_WORK_SEC = 14400;
export const CONT_COLS = ['head_pitch','head_yaw','head_roll','ear_score','mar_score','brow_dist','delta_yaw','delta_pitch','delta_ear','gaze_ratio'];
export const TREND_COLS = ['head_pitch','head_yaw','ear_score'];
export const ACTION_NAMES = ['Do Nothing','Gentle Visual Reminder','Sound Alert','Suggest Short Break','Dim Screen','Play Focus Music'];
export const OCCUPATIONS = { student: 0, office: 0.25, creative: 0.5, technical: 0.75, other: 1 };

// StandardScaler from models/scaler_v2.1.pkl
export const SCALER_MEAN  = [-7.623769, 0.14203, 1.92292, 0.24446, 0.032771, 21.642049, -0.000659, 0.002603, 0.000119, 0.500495];
export const SCALER_SCALE = [19.596203, 20.905710, 93.854901, 0.176328, 0.099095, 16.857744, 24.718392, 21.167936, 0.133483, 0.068432];

// ── landmark metrics (same indices as demo_webcam.py) ─────────────────────
const d2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

export function ear(lm) {
  const one = (ix) => {
    const p = ix.map(i => lm[i]);
    const h = d2(p[0], p[3]);
    return h === 0 ? 0 : (d2(p[1], p[5]) + d2(p[2], p[4])) / (2 * h);
  };
  return (one([263,387,385,362,380,373]) + one([33,160,158,133,153,144])) / 2;
}

export function mar(lm) {
  const h = d2(lm[61], lm[291]);
  return h === 0 ? 0 : d2(lm[13], lm[14]) / h;
}

export function browDist(lm, w, h) {
  const a = lm[105], b = lm[159];
  return Math.hypot((a.x - b.x) * w, (a.y - b.y) * h);
}

export function gazeRatio(lm) {
  if (lm.length < 478) return 0.5;
  const g = (iris, r, l) => {
    const width = d2(lm[r], lm[l]);
    return width === 0 ? 0.5 : d2(lm[r], lm[iris]) / width;
  };
  return (g(468, 362, 263) + g(473, 33, 133)) / 2;
}

// Head pose from MediaPipe's 4x4 facial transformation matrix (column-major).
// Returns degrees; pitch > 0 = looking down (the convention focus_env's note-taking rule assumes).
// No ±180° wrap-around, unlike cv2.decomposeProjectionMatrix used in the Python app.
export function poseFromMatrix(m, invertPitch = false) {
  const r = (i, j) => m[j * 4 + i];
  const deg = 180 / Math.PI;
  const fwd = [r(0,2), r(1,2), r(2,2)];               // face forward axis in camera space
  let pitch = Math.asin(Math.max(-1, Math.min(1, -fwd[1]))) * deg;
  const yaw = Math.atan2(fwd[0], fwd[2]) * deg;
  const roll = Math.atan2(r(1,0), r(0,0)) * deg;
  if (invertPitch) pitch = -pitch;
  return { pitch, yaw, roll };
}

// ── window features (extract_window_features) ─────────────────────────────
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const std = a => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) ** 2))); };
function slope(y) {
  const n = y.length; if (n < 2) return 0;
  const xm = (n - 1) / 2, ym = mean(y);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (i - xm) * (y[i] - ym); den += (i - xm) ** 2; }
  return num / den;
}

export function scaleRow(row) {
  const out = { ...row };
  CONT_COLS.forEach((c, i) => { out[c] = (row[c] - SCALER_MEAN[i]) / SCALER_SCALE[i]; });
  return out;
}

export function windowFeatures(win) {           // win: array of scaled rows
  const col = c => win.map(r => r[c]);
  const headMotion = (std(col('delta_yaw')) + std(col('delta_pitch'))) / 2;
  const eyeW = headMotion < 0.1 ? 1.3 : 1.0, headW = headMotion < 0.1 ? 0.7 : 1.0;
  const wts = [headW, headW, headW, eyeW, eyeW, eyeW, headW, headW, eyeW, eyeW];
  const f = [];
  CONT_COLS.forEach((c, i) => f.push(mean(col(c)) * wts[i]));
  CONT_COLS.forEach((c, i) => f.push(std(col(c)) * wts[i]));
  f.push(mean(col('person_detected')));
  f.push(mean(col('phone_count').map(v => Math.min(v, 1))));
  TREND_COLS.forEach(c => f.push(slope(col(c)) * (c === 'ear_score' ? eyeW : headW)));
  f.push(mean(col('is_distracted_label')));
  f.push(Math.max(...col('consecutive_frames')) / MAX_CONSECUTIVE);
  f.push(win[win.length - 1].fatigue_score);
  return f;
}

export function encodeContext(age, hour, occupation, workSec) {
  const a = 2 * Math.PI * hour / 24;
  return [Math.sin(a), Math.cos(a),
    Math.min(1, Math.max(0, (age - 18) / 47)),
    OCCUPATIONS[occupation] ?? 1,
    Math.min(1, Math.max(0, workSec / MAX_WORK_SEC))];
}

// ── DQN q-network: 33 → 256 → 256 → 6, ReLU (weights in policy.bin) ────────
export class QNet {
  constructor(buf) {
    const f = new Float32Array(buf); let o = 0;
    const take = n => { const v = f.subarray(o, o + n); o += n; return v; };
    this.w0 = take(256 * 33); this.b0 = take(256);
    this.w1 = take(256 * 256); this.b1 = take(256);
    this.w2 = take(6 * 256);  this.b2 = take(6);
  }
  static dense(w, b, x, nOut, relu) {
    const nIn = x.length, y = new Float32Array(nOut);
    for (let i = 0; i < nOut; i++) {
      let s = b[i]; const row = i * nIn;
      for (let j = 0; j < nIn; j++) s += w[row + j] * x[j];
      y[i] = relu && s < 0 ? 0 : s;
    }
    return y;
  }
  q(x) {
    const h0 = QNet.dense(this.w0, this.b0, Float32Array.from(x), 256, true);
    const h1 = QNet.dense(this.w1, this.b1, h0, 256, true);
    return Array.from(QNet.dense(this.w2, this.b2, h1, 6, false));
  }
  act(x) { const q = this.q(x); return { action: q.indexOf(Math.max(...q)), q }; }
}

// ── Driver-monitoring rules (no RL: safety-critical, must be deterministic) ─
// Thresholds follow common DMS practice (PERCLOS, eye-closure duration, Euro NCAP-style
// long-distraction timing). They are starting points to tune, not certified values.
export const DRIVER = {
  microsleepSec: 1.5,      // eyes closed continuously → immediate alarm
  perclosWindowSec: 60,
  perclosWarn: 0.15, perclosSevere: 0.30,
  offRoadSec: 2.0,         // head/eyes away from the road continuously
  yawnMar: 0.55, yawnMinSec: 1.2, yawnWindowSec: 300, yawnWarnCount: 3,
  phoneSec: 1.0,
};

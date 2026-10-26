// app.js — camera loop and UI for Focus Tracker v3.
// Study mode: calibrated rules detect "away", a drowsiness model scores fatigue,
// and a nudge policy (none / v2.1 rule / LinUCB bandit) decides what to do.
// Driver mode: deterministic alarms only.
import { FaceLandmarker, ObjectDetector, FilesetResolver }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import * as C from './core.js';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const OBJ_MODEL = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite';

const $ = id => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), octx = overlay.getContext('2d');
const store = {
  get(k, d = null) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};

let face, objects, drowsyModel = null, banditPrior = null, stream;
let mode = 'study', running = false;

// ── settings ──────────────────────────────────────────────────────────────
const S0 = { participant: '', condition: 'C', level: 'medium', phone: true, sound: true, invertPitch: false, camera: 'user' };
const settings = { ...S0, ...store.get('ft.settings', {}) };
const LEVELS = { low: { perHour: 4, firstS: 8 }, medium: { perHour: 6, firstS: 5 }, high: { perHour: 10, firstS: 5 } };
function applySettingsToForm() {
  $('participant').value = settings.participant; $('condition').value = settings.condition; $('level').value = settings.level;
  $('phoneToggle').checked = settings.phone; $('soundToggle').checked = settings.sound; $('invertPitch').checked = settings.invertPitch;
  $('cameraSel').value = settings.camera;
}
function readSettings() {
  Object.assign(settings, {
    participant: $('participant').value.trim(), condition: $('condition').value, level: $('level').value,
    phone: $('phoneToggle').checked, sound: $('soundToggle').checked, invertPitch: $('invertPitch').checked, camera: $('cameraSel').value,
  });
  store.set('ft.settings', settings);
  renderConditionNote();
}
['participant', 'condition', 'level', 'phoneToggle', 'soundToggle', 'invertPitch', 'cameraSel'].forEach(id => $(id).addEventListener('change', readSettings));

const CONDITION_TEXT = { A: 'A · Không nhắc (chỉ ghi nhận)', B: 'B · Luật cố định của v2.1', C: 'C · Bandit học theo phản ứng' };
function renderConditionNote() { $('conditionNote').textContent = CONDITION_TEXT[settings.condition]; }

// ── calibration ───────────────────────────────────────────────────────────
const calib = { active: false, until: 0, samples: [], pitch: 0, yaw: 0, ear: 0.28, done: false };
function startCalibration() {
  Object.assign(calib, { active: true, samples: [], until: performance.now() + 4000 });
  toast(mode === 'study' ? 'Nhìn vào màn hình như lúc đang làm việc trong 4 giây…' : 'Nhìn thẳng ra đường trong 4 giây…', 'info', 4000);
}
function finishCalibration() {
  calib.active = false;
  if (calib.samples.length < 10) { toast('Chưa thấy khuôn mặt rõ. Bấm Hiệu chỉnh lại khi đã ngồi đúng chỗ.', 'warn'); return; }
  const med = k => { const a = calib.samples.map(s => s[k]).sort((x, y) => x - y); return a[a.length >> 1]; };
  Object.assign(calib, { pitch: med('pitch'), yaw: med('yaw'), ear: med('ear'), done: true });
  $('calibInfo').textContent = `Đã hiệu chỉnh: mắt mở EAR ${calib.ear.toFixed(3)}, coi là nhắm khi dưới ${(calib.ear * C.CLOSED).toFixed(3)}.`;
  toast('Đã hiệu chỉnh.', 'ok');
}
const earClosed = () => (calib.done ? calib.ear * C.CLOSED : 0.2);

// ── session state ─────────────────────────────────────────────────────────
let S;
function newSession() {
  const now = performance.now();
  S = {
    id: Math.random().toString(36).slice(2, 10), startedAt: new Date().toISOString(), t0: now, lastT: now, lastSampleT: 0,
    samples: [], log: [], decisions: [], timeline: [],
    distractedS: 0, focusedS: 0, lookDownS: 0, reasons: [], episodes: [], episodeStart: null,
    focusS: 0, awayS: 0, drowsy: null, drowsyHist: [],
    budget: new C.NudgeBudget({ perHour: LEVELS[settings.level].perHour }),
    nextDecisionAt: LEVELS[settings.level].firstS, pending: [], nudgeTimes: [],
    selfLabel: 0, meeting: false, notes: false, phoneCount: 0, phoneBoxes: [], objPerson: 0, lastObjT: 0,
    // driver
    closedS: 0, offRoadS: 0, phoneS: 0, perclos: [], yawnOpenS: 0, yawns: [], lastAlarmT: 0, level: 'ok',
  };
  bandit = makeBandit();
}

// ── nudge policy ──────────────────────────────────────────────────────────
let bandit = null;
function banditKey() { return `ft.bandit.${settings.participant || 'anon'}`; }
function makeBandit() {
  const saved = store.get(banditKey());
  if (saved) return C.LinUCB.fromJSON(saved);
  return banditPrior ? C.LinUCB.fromJSON(banditPrior) : new C.LinUCB();
}
function saveBandit() { if (bandit) store.set(banditKey(), bandit.toJSON()); }
$('resetBandit').addEventListener('click', () => {
  try { localStorage.removeItem(banditKey()); } catch { /* ignore */ }
  bandit = makeBandit(); toast('Đã đặt lại model nhắc nhở về trạng thái ban đầu.', 'ok');
});

function decide(x) {
  if (settings.condition === 'A') return ['wait', 1];
  if (settings.condition === 'B') return [C.ruleV21(S.distractedS), 1];
  return bandit.act(x);
}

const NUDGE_TEXT = {
  toast: 'Bạn đang rời khỏi việc đang làm. Quay lại nhé.',
  sound: 'Quay lại tập trung nào.',
  break: 'Có vẻ bạn đang mệt. Nghỉ 5 phút rồi quay lại sẽ hiệu quả hơn.',
  dim: 'Màn hình tối đi một chút để nhắc bạn quay lại.',
};
function performNudge(action, decision) {
  if (action === 'wait') return;
  if (action === 'sound') beep(660, 0.25, 0.12);
  if (action === 'dim') { $('dim').hidden = false; setTimeout(() => { $('dim').hidden = true; }, 5000); }
  toast(NUDGE_TEXT[action], action === 'break' ? 'info' : 'warn', 8000, () => { decision.notHelpful = true; toast('Đã ghi nhận. Cảm ơn bạn.', 'ok', 2000); });
}

function decisionPoint(now, elapsed) {
  const sinceNudge = S.nudgeTimes.length ? elapsed - S.nudgeTimes.at(-1) : 600;
  const recent = S.nudgeTimes.filter(t => elapsed - t < 600).length;
  const reasons = S.reasons.map(r => (r === 'no_face' ? 'absent' : r));
  const x = C.context(S.distractedS, S.drowsy ?? 0, sinceNudge, recent, elapsed, new Date().getHours() + new Date().getMinutes() / 60, reasons);
  const suppressed = S.meeting || S.notes;
  const capped = !S.budget.available(elapsed);
  let action = 'wait', p = 1, why = 'policy';
  if (suppressed) why = S.meeting ? 'meeting' : 'notes';
  else if (capped) why = 'budget';
  else [action, p] = decide(x);
  const d = { t: elapsed, x, action, p, why, condition: settings.condition, distractedS: S.distractedS, reasons, drowsy: S.drowsy, notHelpful: false, resolved: false };
  if (action !== 'wait') { S.budget.spend(elapsed); S.nudgeTimes.push(elapsed); }
  performNudge(action, d);
  S.pending.push(d);
  S.lastDecision = d;
  renderPolicy(x);
}

function resolveDecisions(elapsed) {
  for (const d of S.pending) {
    if (d.resolved) continue;
    const back = S.distractedS === 0;
    if (back || elapsed - d.t >= 10) {
      d.resolved = true; d.refocused = back && elapsed - d.t <= 10;
      d.reward = C.reward(d.action, d.refocused, d.notHelpful);
      if (d.why === 'policy' && settings.condition === 'C') { bandit.update(d.x, d.action, d.reward); saveBandit(); }
      S.decisions.push(d);
    }
  }
  S.pending = S.pending.filter(d => !d.resolved);
}

// ── models ────────────────────────────────────────────────────────────────
async function loadModels() {
  setLoading('Đang tải mô hình khuôn mặt…');
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  face = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' }, runningMode: 'VIDEO', numFaces: 1, outputFacialTransformationMatrixes: true,
  });
  setLoading('Đang tải bộ phát hiện điện thoại…');
  objects = await ObjectDetector.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: OBJ_MODEL, delegate: 'GPU' }, runningMode: 'VIDEO', scoreThreshold: 0.45, maxResults: 5,
  });
}
async function loadPolicies() {
  try { drowsyModel = await (await fetch('./models/drowsiness.json')).json(); } catch { drowsyModel = null; }
  try { banditPrior = await (await fetch('./models/nudge_prior.json')).json(); } catch { banditPrior = null; }
  $('drowsyInfo').textContent = drowsyModel
    ? `Logistic regression, ${drowsyModel.subjects} người trong UTA-RLDD, AUROC chéo ${drowsyModel.window_auroc_oof.toFixed(2)}`
    : 'Chưa có model buồn ngủ, chỉ dùng PERCLOS.';
}

async function start() {
  try {
    $('startBtn').disabled = true;
    if (!face) await loadModels();
    setLoading('Đang xin quyền camera…');
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: settings.camera }, audio: false });
    video.srcObject = stream; await video.play();
    document.body.dataset.mirror = settings.camera === 'user' ? '1' : '0';
    overlay.width = video.videoWidth; overlay.height = video.videoHeight;
    setLoading(null);
    $('startBtn').hidden = true; $('stopBtn').hidden = false; $('calibBtn').disabled = false; $('summary').hidden = true;
    newSession(); running = true;
    if (mode === 'driver') requestWakeLock();
    startCalibration();
    requestAnimationFrame(loop);
  } catch (e) {
    setLoading(null); $('startBtn').disabled = false;
    toast(e.name === 'NotAllowedError' ? 'Trình duyệt đang chặn camera. Cho phép camera cho trang này rồi thử lại.'
      : `Không khởi động được: ${e.message}`, 'warn', 8000);
  }
}
function stop() {
  running = false;
  stream?.getTracks().forEach(t => t.stop());
  wakeLock?.release?.(); wakeLock = null;
  $('startBtn').hidden = false; $('startBtn').disabled = false; $('stopBtn').hidden = true; $('calibBtn').disabled = true;
  document.body.dataset.alarm = '0';
  if (S && mode === 'study') showSummary();
}

// ── main loop ─────────────────────────────────────────────────────────────
function loop() {
  if (!running) return;
  const now = performance.now();
  const dt = Math.min(1.0, (now - S.lastT) / 1000); S.lastT = now;   // capped so a backgrounded tab does not jump
  const elapsed = (now - S.t0) / 1000;

  const res = face.detectForVideo(video, now);
  if (settings.phone && now - S.lastObjT > 400) {
    S.lastObjT = now;
    const det = objects.detectForVideo(video, now).detections;
    const cat = d => d.categories[0].categoryName;
    S.phoneBoxes = det.filter(d => cat(d) === 'cell phone').map(d => d.boundingBox);
    S.phoneCount = S.phoneBoxes.length;
    S.objPerson = det.some(d => cat(d) === 'person') ? 1 : 0;
  } else if (!settings.phone) { S.phoneCount = 0; S.phoneBoxes = []; }

  let m = { pitch: 0, yaw: 0, roll: 0, ear: 0, mar: 0, brow: 0, gaze: 0.5, person: S.objPerson };
  const lm = res.faceLandmarks?.[0];
  if (lm) {
    const pose = C.poseFromMatrix(res.facialTransformationMatrixes[0].data, settings.invertPitch);
    const l43 = C.to43(lm, video.videoWidth, video.videoHeight);
    m = { ...pose, ear: C.ear(l43), mar: C.mar(l43), brow: C.browDist(l43), gaze: C.gazeRatio(l43), person: 1 };
    if (calib.active) calib.samples.push({ pitch: m.pitch, yaw: m.yaw, ear: m.ear });
  }
  if (calib.active && now > calib.until) finishCalibration();

  // 3 Hz sample stream (same rate as the training data)
  if (now - S.lastSampleT >= 1000 / C.HZ) {
    S.lastSampleT = now;
    S.samples.push({ ear_score: m.ear, mar_score: m.mar, head_pitch: m.pitch, head_yaw: m.yaw, gaze_ratio: m.gaze });
    if (S.samples.length > 600) S.samples.shift();
    if (S.samples.length >= 30 && S.samples.length % 3 === 0) {
      const ref = S.samples.length >= C.REF_SAMPLES ? (S.ref ??= C.reference(S.samples)) : { ear: calib.ear, pitch: calib.pitch, yaw: calib.yaw };
      S.features = C.windowFeatures(S.samples, ref);
      S.drowsy = drowsyModel ? C.drowsiness(drowsyModel, S.features) : null;
      if (S.drowsy != null) S.drowsyHist.push(S.drowsy);
    }
  }

  drawOverlay(lm);
  if (mode === 'study') studyStep(m, dt, elapsed); else driverStep(m, dt, now);
  requestAnimationFrame(loop);
}

// ── study mode ────────────────────────────────────────────────────────────
function studyReasons(m) {
  if (!m.person) return ['absent'];
  if (m.ear <= 0) return ['no_face'];
  const pitch = m.pitch - calib.pitch, yaw = m.yaw - calib.yaw, open = m.ear >= earClosed();
  const noteTaking = (pitch > 20 && Math.abs(yaw) < 20 && S.phoneCount === 0 && open && S.lookDownS < 30) || S.notes;
  const out = [];
  if (!open) out.push('eyes_closed');
  if (Math.abs(yaw) > 40) out.push('head_turned');
  if (pitch > 30 && !noteTaking) out.push('head_down');
  if (pitch < -25) out.push('looking_up');
  if (S.phoneCount > 0) out.push('phone');
  return out;
}
const REASON_TEXT = { absent: 'Không thấy người', no_face: 'Không thấy khuôn mặt', eyes_closed: 'Mắt nhắm', head_turned: 'Quay đầu sang bên',
  head_down: 'Cúi đầu quá lâu', looking_up: 'Nhìn lên', phone: 'Có điện thoại' };

function studyStep(m, dt, elapsed) {
  const pitch = m.pitch - calib.pitch, yaw = m.yaw - calib.yaw;
  S.lookDownS = (pitch > 20 && Math.abs(yaw) < 20 && m.ear >= earClosed()) ? S.lookDownS + dt : 0;
  S.reasons = calib.active ? [] : studyReasons(m);
  const away = S.reasons.length > 0;

  // hysteresis: the longer the distraction, the longer focus must hold before it counts as over
  if (away) {
    if (S.distractedS === 0) S.episodeStart = elapsed;
    S.distractedS += dt; S.focusedS = 0;
  } else if (S.distractedS > 0) {
    S.focusedS += dt;
    const need = S.distractedS >= 9 ? 4.5 : S.distractedS >= 3 ? 1.5 : 0.5;
    if (S.focusedS >= need) {
      if (S.distractedS >= 5) S.episodes.push({ start: S.episodeStart, dur: S.distractedS });
      S.distractedS = 0; S.focusedS = 0; S.nextDecisionAt = LEVELS[settings.level].firstS;
    }
  }
  if (S.distractedS > 0) S.awayS += dt; else S.focusS += dt;

  if (S.distractedS >= S.nextDecisionAt) { decisionPoint(performance.now(), elapsed); S.nextDecisionAt += 15; }
  resolveDecisions(elapsed);

  if (S.timeline.length === 0 || elapsed - S.timeline.at(-1).t >= 1) {
    S.timeline.push({ t: elapsed, away: S.distractedS > 0 ? 1 : 0, nudge: S.lastDecision && S.lastDecision.t > elapsed - 1 && S.lastDecision.action !== 'wait' ? 1 : 0 });
    S.log.push({
      session_id: S.id, timestamp: elapsed.toFixed(2), head_pitch: m.pitch, head_yaw: m.yaw, head_roll: m.roll, ear_score: m.ear,
      mar_score: m.mar, brow_dist: m.brow, gaze_ratio: m.gaze, person_detected: m.person, phone_count: S.phoneCount,
      is_distracted_label: away ? 1 : 0, distracted_s: S.distractedS, drowsy: S.drowsy ?? '', gt_label: S.selfLabel,
      meeting: S.meeting ? 1 : 0, notes: S.notes ? 1 : 0,
    });
  }
  renderStudy(m);
}

// ── driver mode: deterministic alarms ─────────────────────────────────────
function driverStep(m, dt, now) {
  const D = C.DRIVER, t = now / 1000, faceSeen = m.ear > 0;
  const closed = faceSeen && m.ear < earClosed();
  S.closedS = closed ? S.closedS + dt : 0;
  S.perclos.push([t, closed ? 1 : 0]);
  while (S.perclos.length && S.perclos[0][0] < t - 60) S.perclos.shift();
  const perclos = S.perclos.reduce((s, p) => s + p[1], 0) / Math.max(1, S.perclos.length);
  const coverage = S.perclos.length ? (S.perclos.at(-1)[0] - S.perclos[0][0]) / 60 : 0;
  const pitch = m.pitch - calib.pitch, yaw = m.yaw - calib.yaw;
  const offRoad = !faceSeen || Math.abs(yaw) > 30 || pitch > 25 || pitch < -20;
  S.offRoadS = offRoad && !closed ? S.offRoadS + dt : 0;
  S.phoneS = S.phoneCount > 0 ? S.phoneS + dt : 0;
  if (m.mar > C.YAWN_MAR) S.yawnOpenS += dt; else { if (S.yawnOpenS >= 1.2) S.yawns.push(t); S.yawnOpenS = 0; }
  S.yawns = S.yawns.filter(y => y > t - 300);

  let level = 'ok', msg = 'Tỉnh táo, mắt hướng về phía trước';
  if (S.yawns.length >= D.yawnWarnCount) { level = 'warn'; msg = `Ngáp ${S.yawns.length} lần trong 5 phút. Cân nhắc dừng nghỉ.`; }
  if (S.drowsy != null && S.drowsy > D.drowsyWarn) { level = 'warn'; msg = `Model đánh giá buồn ngủ ${(S.drowsy * 100).toFixed(0)}%`; }
  if (coverage > 0.5 && perclos > D.perclosWarn) { level = 'warn'; msg = `Dấu hiệu buồn ngủ (PERCLOS ${(perclos * 100).toFixed(0)}%)`; }
  if (coverage > 0.5 && perclos > D.perclosAlarm) { level = 'alarm'; msg = 'Buồn ngủ nặng. Hãy dừng xe ở nơi an toàn.'; }
  if (S.offRoadS > D.offRoadS) { level = 'alarm'; msg = faceSeen ? 'Mắt rời khỏi mặt đường!' : 'Không thấy khuôn mặt tài xế!'; }
  if (S.phoneS > D.phoneS) { level = 'alarm'; msg = 'Đang dùng điện thoại khi lái!'; }
  if (S.closedS > D.microsleepS) { level = 'alarm'; msg = 'NGỦ GẬT! Mở mắt ra!'; }
  if (calib.active) { level = 'ok'; msg = 'Đang hiệu chỉnh…'; }
  if (level === 'alarm' && t - S.lastAlarmT > 0.6) { S.lastAlarmT = t; beep(880, 0.25, 0.25); }
  if (S.timeline.length === 0 || t - S.timeline.at(-1).t >= 1) S.timeline.push({ t, away: level === 'ok' ? 0 : 1 });
  renderDriver(m, { perclos, coverage, level, msg });
}

// ── rendering ─────────────────────────────────────────────────────────────
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function drawOverlay(lm) {
  const w = overlay.width, h = overlay.height;
  octx.clearRect(0, 0, w, h);
  if (lm) {
    octx.fillStyle = css('--mesh');
    for (let i = 0; i < lm.length; i += 3) octx.fillRect(lm[i].x * w - 1, lm[i].y * h - 1, 2, 2);
    octx.fillStyle = css('--accent');
    [33, 133, 362, 263, 468, 473, 13, 14, 61, 291].forEach(i => lm[i] && octx.fillRect(lm[i].x * w - 2, lm[i].y * h - 2, 4, 4));
  }
  octx.strokeStyle = css('--bad'); octx.lineWidth = 3;
  (S.phoneBoxes || []).forEach(b => octx.strokeRect(b.originX, b.originY, b.width, b.height));
}
const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '–');
const mmss = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
function setStatus(text, level) { const el = $('status'); el.textContent = text; el.dataset.level = level; }

function renderCommon(m) {
  $('mPitch').textContent = `${fmt(m.pitch - calib.pitch)}°`;
  $('mYaw').textContent = `${fmt(m.yaw - calib.yaw)}°`;
  $('mEar').textContent = fmt(m.ear, 3);
  $('mMar').textContent = fmt(m.mar, 2);
  $('mGaze').textContent = fmt(m.gaze, 2);
  $('mPhone').textContent = S.phoneCount ? 'Có' : 'Không';
  $('elapsed').textContent = mmss((performance.now() - S.t0) / 1000);
  const dz = S.drowsy;
  $('drowsyVal').textContent = dz == null ? 'đang đo…' : `${Math.round(dz * 100)}%`;
  $('drowsyBar').style.setProperty('--w', `${dz == null ? 0 : dz * 100}%`);
  drawStrip();
}

function renderStudy(m) {
  renderCommon(m);
  const s = S.distractedS;
  if (calib.active) setStatus('Đang hiệu chỉnh', 'ok');
  else if (S.meeting) setStatus('Đang họp', 'ok');
  else if (!m.person) setStatus('Vắng mặt', 'alarm');
  else if (s < 3) setStatus('Tập trung', 'ok');
  else if (s < 9) setStatus('Cảnh báo sớm', 'warn');
  else setStatus('Mất tập trung', 'alarm');
  $('reason').textContent = S.reasons.length ? S.reasons.map(r => REASON_TEXT[r]).join(' · ') : (S.notes ? 'Chế độ ghi chép' : 'Đang nhìn vào màn hình');
  $('timer').textContent = `${fmt(s)} s`;
  const done = S.decisions.filter(d => d.action !== 'wait');
  $('statNudges').textContent = String(S.nudgeTimes.length);
  $('statUseful').textContent = done.length ? `${Math.round(100 * done.filter(d => d.refocused).length / done.length)}%` : '–';
  $('statFocus').textContent = mmss(S.focusS);
}

const ACTION_TEXT = { wait: 'Chờ', toast: 'Nhắc nhẹ', sound: 'Âm thanh', break: 'Gợi ý nghỉ', dim: 'Làm tối màn hình' };
function renderPolicy(x) {
  const d = S.lastDecision;
  $('lastAction').textContent = d ? `${ACTION_TEXT[d.action]}${d.why !== 'policy' ? ` (bỏ qua: ${{ meeting: 'đang họp', notes: 'đang ghi chép', budget: 'đã nhắc đủ' }[d.why]})` : ''}` : '–';
  $('lastProb').textContent = d && d.why === 'policy' ? `${Math.round(d.p * 100)}%` : '–';
  const show = settings.condition === 'C';
  $('qbars').hidden = !show;
  if (!show) return;
  const sc = bandit.scores(x, false), vals = Object.values(sc), max = Math.max(...vals), min = Math.min(...vals);
  C.ACTIONS.forEach(a => {
    const bar = $(`q-${a}`);
    bar.style.setProperty('--w', `${max === min ? 100 : 15 + 85 * (sc[a] - min) / (max - min)}%`);
    bar.dataset.best = sc[a] === max ? '1' : '0';
    bar.querySelector('b').textContent = sc[a].toFixed(2);
  });
}

function renderDriver(m, r) {
  renderCommon(m);
  setStatus(r.level === 'ok' ? 'Tỉnh táo' : r.level === 'warn' ? 'Cảnh báo' : 'NGUY HIỂM', r.level);
  $('reason').textContent = r.msg;
  $('timer').textContent = `${fmt(S.offRoadS)} s`;
  $('dPerclos').textContent = r.coverage > 0.5 ? `${Math.round(r.perclos * 100)}%` : `đang đo (${Math.round(r.coverage * 100)}%)`;
  $('dPerclosBar').style.setProperty('--w', `${Math.min(100, r.perclos / 0.4 * 100)}%`);
  $('dClosed').textContent = `${fmt(S.closedS)} s`;
  $('dOff').textContent = `${fmt(S.offRoadS)} s`;
  $('dYawn').textContent = String(S.yawns.length);
  document.body.dataset.alarm = r.level === 'alarm' ? '1' : '0';
}

function drawStrip(canvas = $('strip'), tl = S.timeline.slice(-180), n = 180) {
  const x = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * devicePixelRatio, H = canvas.height = canvas.clientHeight * devicePixelRatio;
  x.clearRect(0, 0, W, H);
  const bw = W / n;
  tl.forEach((p, i) => {
    const xi = (n - tl.length + i) * bw;
    x.fillStyle = css(p.away ? '--bad' : '--good'); x.globalAlpha = p.away ? 0.9 : 0.35;
    x.fillRect(xi, p.away ? H * 0.15 : H * 0.6, Math.max(1, bw - 0.5), p.away ? H * 0.85 : H * 0.4);
    if (p.nudge) { x.globalAlpha = 1; x.fillStyle = css('--accent'); x.fillRect(xi, 0, Math.max(2, bw), H * 0.12); }
  });
  x.globalAlpha = 1;
}

// ── session summary + questionnaire + exports ─────────────────────────────
function summary() {
  const total = (S.lastT - S.t0) / 1000;
  const eps = [...S.episodes, ...(S.distractedS >= 5 ? [{ start: S.episodeStart, dur: S.distractedS }] : [])];
  const nudges = S.decisions.filter(d => d.action !== 'wait');
  return {
    session_id: S.id, participant: settings.participant || 'anon', condition: settings.condition, level: settings.level,
    started_at: S.startedAt, minutes: +(total / 60).toFixed(2), focused_minutes: +(S.focusS / 60).toFixed(2),
    distraction_episodes: eps.length, mean_return_s: eps.length ? +(eps.reduce((s, e) => s + e.dur, 0) / eps.length).toFixed(1) : null,
    nudges: nudges.length, nudges_per_hour: total > 0 ? +(nudges.length / total * 3600).toFixed(1) : 0,
    useful_nudge_rate: nudges.length ? +(nudges.filter(d => d.refocused).length / nudges.length).toFixed(3) : null,
    not_helpful: nudges.filter(d => d.notHelpful).length,
    mean_drowsiness: S.drowsyHist.length ? +(S.drowsyHist.reduce((a, b) => a + b, 0) / S.drowsyHist.length).toFixed(3) : null,
    self_rated_focus: $('qFocus').value ? Number($('qFocus').value) : null,
    self_rated_annoyance: $('qAnnoy').value ? Number($('qAnnoy').value) : null,
  };
}
function showSummary() {
  const s = summary();
  $('sumMinutes').textContent = `${s.focused_minutes} / ${s.minutes} phút`;
  $('sumEpisodes').textContent = String(s.distraction_episodes);
  $('sumReturn').textContent = s.mean_return_s == null ? '–' : `${s.mean_return_s} s`;
  $('sumNudges').textContent = `${s.nudges} (${s.useful_nudge_rate == null ? '–' : `${Math.round(s.useful_nudge_rate * 100)}% có ích`})`;
  $('sumDrowsy').textContent = s.mean_drowsiness == null ? '–' : `${Math.round(s.mean_drowsiness * 100)}%`;
  $('summary').hidden = false;
  drawStrip($('sumStrip'), S.timeline, Math.max(S.timeline.length, 1));
  $('summary').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function download(name, text, type = 'text/csv') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
const csv = (cols, rows) => [cols.join(','), ...rows.map(r => cols.map(c => {
  const v = r[c]; return typeof v === 'number' ? +v.toFixed(5) : Array.isArray(v) ? `"${v.join(' ')}"` : (v ?? '');
}).join(','))].join('\n');
const pid = () => settings.participant || 'anon';

$('dlFeatures').addEventListener('click', () => {
  if (!S?.log.length) return toast('Chưa có dữ liệu.', 'info');
  const rows = S.log.map(r => ({ ...r, subject_id: pid(), source: 'web', condition: settings.condition }));
  download(`features_${pid()}_${S.id}.csv`, csv(['session_id', 'timestamp', 'head_pitch', 'head_yaw', 'head_roll', 'ear_score', 'mar_score',
    'brow_dist', 'gaze_ratio', 'person_detected', 'phone_count', 'is_distracted_label', 'distracted_s', 'drowsy', 'meeting', 'notes',
    'gt_label', 'subject_id', 'source', 'condition'], rows));
});
$('dlDecisions').addEventListener('click', () => {
  if (!S?.decisions.length) return toast('Chưa có quyết định nhắc nào trong buổi này.', 'info');
  const rows = S.decisions.map(d => ({ session_id: S.id, participant: pid(), condition: d.condition, t: d.t, action: d.action, p: d.p, why: d.why,
    refocused: d.refocused ? 1 : 0, not_helpful: d.notHelpful ? 1 : 0, reward: d.reward, distracted_s: d.distractedS, drowsy: d.drowsy ?? '',
    reasons: d.reasons, x: d.x }));
  download(`decisions_${pid()}_${S.id}.csv`, csv(['session_id', 'participant', 'condition', 't', 'action', 'p', 'why', 'refocused', 'not_helpful',
    'reward', 'distracted_s', 'drowsy', 'reasons', 'x'], rows));
});
$('dlSummary').addEventListener('click', () => {
  if (!S) return;
  const s = summary(); store.set(`ft.summary.${S.id}`, s);
  download(`summary_${pid()}_${S.id}.json`, JSON.stringify(s, null, 2), 'application/json');
});
$('saveQ').addEventListener('click', () => {
  if (!S) return;
  store.set(`ft.summary.${S.id}`, summary());
  toast('Đã lưu câu trả lời vào bản tổng kết.', 'ok');
});

// ── small utilities ───────────────────────────────────────────────────────
let audio, wakeLock = null;
function beep(freq, dur, gain = 0.15) {
  if (!settings.sound) return;
  audio ??= new AudioContext();
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = freq; g.gain.value = gain; o.connect(g).connect(audio.destination);
  o.start(); o.stop(audio.currentTime + dur);
}
async function requestWakeLock() { try { wakeLock = await navigator.wakeLock?.request('screen'); } catch { wakeLock = null; } }
function toast(text, kind = 'info', ms = 4000, onNotHelpful = null) {
  const el = document.createElement('div'); el.className = 'toast'; el.dataset.kind = kind;
  const p = document.createElement('span'); p.textContent = text; el.append(p);
  if (onNotHelpful) {
    const b = document.createElement('button'); b.className = 'linkbtn'; b.textContent = 'Không hữu ích';
    b.addEventListener('click', () => { onNotHelpful(); el.remove(); }); el.append(b);
  }
  $('toasts').append(el); setTimeout(() => el.remove(), ms);
}
function setLoading(text) { $('loading').hidden = !text; if (text) $('loadingText').textContent = text; }

function setMode(next) {
  if (running) stop();
  mode = next;
  document.body.dataset.mode = next; document.body.dataset.alarm = '0';
  document.querySelectorAll('[data-tab]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === next)));
  $('timerLabel').textContent = next === 'study' ? 'Thời gian mất tập trung' : 'Thời gian rời mắt khỏi đường';
  if (next === 'driver' && settings.camera === 'user' && matchMedia('(max-width: 600px)').matches) toast('Kẹp điện thoại sao cho camera trước nhìn thẳng vào mặt.', 'info', 6000);
}
document.querySelectorAll('[data-tab]').forEach(b => b.addEventListener('click', () => setMode(b.dataset.tab)));

function toggle(id, key, label) {
  $(id).addEventListener('click', () => {
    if (!S) return;
    S[key] = !S[key];
    $(id).setAttribute('aria-pressed', String(S[key]));
    $(id).textContent = S[key] ? `Tắt chế độ ${label}` : `Chế độ ${label}`;
  });
}
toggle('meetingBtn', 'meeting', 'họp');
toggle('notesBtn', 'notes', 'ghi chép');
function toggleSelfLabel() {
  if (!S) return;
  S.selfLabel = S.selfLabel ? 0 : 1;
  $('labelBtn').setAttribute('aria-pressed', String(!!S.selfLabel));
  $('labelBtn').textContent = S.selfLabel ? 'Đang ghi: MẤT TẬP TRUNG (phím D)' : 'Tự báo mất tập trung (phím D)';
}
$('labelBtn').addEventListener('click', toggleSelfLabel);
addEventListener('keydown', e => { if ((e.key === 'd' || e.key === 'D') && !/INPUT|SELECT|TEXTAREA/.test(e.target.tagName) && mode === 'study') toggleSelfLabel(); });

$('startBtn').addEventListener('click', start);
$('stopBtn').addEventListener('click', stop);
$('calibBtn').addEventListener('click', startCalibration);
applySettingsToForm(); renderConditionNote(); setMode('study'); loadPolicies();
if (!window.isSecureContext) toast('Camera chỉ chạy trên HTTPS hoặc localhost. Mở qua GitHub Pages hoặc `python -m http.server`.', 'warn', 12000);

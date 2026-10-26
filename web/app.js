// app.js — camera loop, study mode (DQN + rules) and driver mode (DMS rules).
import { FaceLandmarker, ObjectDetector, FilesetResolver }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import * as C from './core.js';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const OBJ_MODEL = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite';

const $ = id => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), octx = overlay.getContext('2d');

let face, objects, net, stream;
let mode = 'study';
let running = false;

// ── settings ──────────────────────────────────────────────────────────────
const settings = { age: 21, occupation: 'student', invertPitch: false, phone: true, sound: true };
function readSettings() {
  settings.age = Number($('age').value) || 21;
  settings.occupation = $('occupation').value;
  settings.invertPitch = $('invertPitch').checked;
  settings.phone = $('phoneToggle').checked;
  settings.sound = $('soundToggle').checked;
}
['age','occupation','invertPitch','phoneToggle','soundToggle'].forEach(id => $(id).addEventListener('change', readSettings));

// ── calibration: neutral head pose + open-eye EAR (helps users with glasses) ─
const calib = { active: false, until: 0, samples: [], pitch0: 0, yaw0: 0, ear0: 0.28, done: false };
function startCalibration() {
  calib.active = true; calib.samples = []; calib.until = performance.now() + 4000;
  toast('Nhìn thẳng vào màn hình / mặt đường trong 4 giây…', 'info', 4000);
}
function finishCalibration() {
  calib.active = false;
  if (calib.samples.length < 10) { toast('Chưa thấy khuôn mặt rõ, hãy thử hiệu chỉnh lại.', 'warn'); return; }
  const med = k => { const a = calib.samples.map(s => s[k]).sort((x, y) => x - y); return a[a.length >> 1]; };
  calib.pitch0 = med('pitch'); calib.yaw0 = med('yaw'); calib.ear0 = med('ear'); calib.done = true;
  $('calibInfo').textContent = `EAR mở mắt ${calib.ear0.toFixed(3)} · ngưỡng nhắm ${earThreshold().toFixed(3)}`;
  toast('Đã hiệu chỉnh.', 'ok');
}
const earThreshold = () => calib.done ? Math.min(0.2, calib.ear0 * 0.7) : 0.2;

// ── per-frame state ───────────────────────────────────────────────────────
const S = {
  t0: 0, lastT: 0, lastPolicyT: 0,
  distractedSec: 0, lookDownSec: 0, closedSec: 0, offRoadSec: 0, phoneSec: 0, missingSec: 0,
  last: { yaw: 0, pitch: 0, ear: 0 },
  window: [], phoneCount: 0, objPerson: 0, lastObjT: 0,
  perclos: [],                 // [t, closed] samples for the rolling PERCLOS window
  yawnOpenSec: 0, yawns: [],   // timestamps of completed yawns
  history: [],                 // [t, distracted 0/1] for the strip chart
  log: [],                     // feature rows, same schema as data/focus_dataset.csv
  sessionId: Math.random().toString(36).slice(2, 10),
  lastToastT: 0, hardShown: false, lastAlarmT: 0,
};

function resetState() {
  Object.assign(S, { t0: performance.now(), lastT: performance.now(), lastPolicyT: 0, distractedSec: 0, lookDownSec: 0,
    closedSec: 0, offRoadSec: 0, phoneSec: 0, missingSec: 0, window: [], perclos: [], yawnOpenSec: 0, yawns: [],
    history: [], hardShown: false });
}

// ── models ────────────────────────────────────────────────────────────────
async function loadModels() {
  setLoading('Đang tải mô hình nhận diện khuôn mặt…');
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  face = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' }, runningMode: 'VIDEO', numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });
  setLoading('Đang tải bộ phát hiện điện thoại…');
  objects = await ObjectDetector.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: OBJ_MODEL, delegate: 'GPU' }, runningMode: 'VIDEO', scoreThreshold: 0.45, maxResults: 5,
  });
  setLoading('Đang tải chính sách DQN…');
  net = new C.QNet(await (await fetch('./policy.bin')).arrayBuffer());
}

async function start() {
  try {
    $('startBtn').disabled = true;
    if (!face) await loadModels();
    setLoading('Đang xin quyền camera…');
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
    video.srcObject = stream; await video.play();
    overlay.width = video.videoWidth; overlay.height = video.videoHeight;
    setLoading(null);
    $('startBtn').hidden = true; $('stopBtn').hidden = false; $('calibBtn').disabled = false;
    resetState(); running = true; readSettings();
    startCalibration();
    requestAnimationFrame(loop);
  } catch (e) {
    setLoading(null); $('startBtn').disabled = false;
    toast(e.name === 'NotAllowedError' ? 'Trình duyệt chặn camera. Hãy cho phép camera cho trang này rồi thử lại.'
      : `Không khởi động được: ${e.message}`, 'warn', 8000);
  }
}
function stop() {
  running = false;
  stream?.getTracks().forEach(t => t.stop());
  $('startBtn').hidden = false; $('startBtn').disabled = false; $('stopBtn').hidden = true; $('calibBtn').disabled = true;
}

// ── main loop ─────────────────────────────────────────────────────────────
function loop() {
  if (!running) return;
  const now = performance.now();
  const dt = Math.min(0.5, (now - S.lastT) / 1000); S.lastT = now;

  const res = face.detectForVideo(video, now);
  if (settings.phone && now - S.lastObjT > 400) {           // object detector at ~2.5 Hz
    S.lastObjT = now;
    const det = objects.detectForVideo(video, now).detections;
    S.phoneCount = det.filter(d => d.categories[0].categoryName === 'cell phone').length;
    S.objPerson = det.some(d => d.categories[0].categoryName === 'person') ? 1 : 0;
    S.phoneBoxes = det.filter(d => d.categories[0].categoryName === 'cell phone').map(d => d.boundingBox);
  } else if (!settings.phone) { S.phoneCount = 0; S.phoneBoxes = []; }

  let m = { pitch: 0, yaw: 0, roll: 0, ear: 0, mar: 0, brow: 0, gaze: 0, person: 0 };
  const lm = res.faceLandmarks?.[0];
  if (lm) {
    const pose = C.poseFromMatrix(res.facialTransformationMatrixes[0].data, settings.invertPitch);
    m = { ...pose, ear: C.ear(lm), mar: C.mar(lm), brow: C.browDist(lm, video.videoWidth, video.videoHeight),
      gaze: C.gazeRatio(lm), person: 1 };
    if (calib.active) calib.samples.push({ pitch: m.pitch, yaw: m.yaw, ear: m.ear });
    m.pitch -= calib.pitch0; m.yaw -= calib.yaw0;
  } else if (S.objPerson) m.person = 1;
  if (calib.active && now > calib.until) finishCalibration();

  drawOverlay(lm);
  if (mode === 'study') studyStep(m, dt, now); else driverStep(m, dt, now);
  requestAnimationFrame(loop);
}

// ── STUDY MODE: same rules as FocusAgentInference, but in seconds, not frames ─
function studyStep(m, dt, now) {
  const faceSeen = m.ear > 0;
  S.lookDownSec = (m.pitch > 20 && Math.abs(m.yaw) < 20 && m.ear >= earThreshold()) ? S.lookDownSec + dt : 0;
  const noteTaking = m.pitch > 20 && Math.abs(m.yaw) < 20 && S.phoneCount === 0 && m.ear >= earThreshold() && S.lookDownSec < 30;
  const reasons = [];
  if (m.person === 0) reasons.push('Không thấy người');
  else if (!faceSeen) reasons.push('Không thấy khuôn mặt');
  else {
    if (m.ear < earThreshold()) reasons.push('Mắt nhắm');
    if (Math.abs(m.yaw) > 40) reasons.push('Quay đầu sang bên');
    if (Math.abs(m.pitch) > 30 && !noteTaking) reasons.push(m.pitch > 0 ? 'Cúi đầu quá lâu' : 'Ngửa đầu');
    if (m.pitch < -25 && !reasons.includes('Ngửa đầu')) reasons.push('Nhìn lên');
  }
  if (S.phoneCount > 0) reasons.push('Có điện thoại');
  const distracted = reasons.length > 0 ? 1 : 0;
  // Hysteresis (same idea as the local focus_env.py change): the longer the distraction,
  // the longer focus must be sustained before the counter resets (5/15/45 frames at 10 fps).
  if (distracted) { S.distractedSec += dt; S.focusedSec = 0; }
  else {
    S.focusedSec = (S.focusedSec || 0) + dt;
    const need = S.distractedSec >= 9 ? 4.5 : S.distractedSec >= 3 ? 1.5 : 0.5;
    if (S.focusedSec >= need) { S.distractedSec = 0; S.focusedSec = 0; }
  }

  // policy runs at 3 Hz so its window matches the training data (10 frames ≈ 3.3 s)
  let decision = S.decision;
  if (now - S.lastPolicyT >= 1000 / C.POLICY_HZ) {
    S.lastPolicyT = now;
    const elapsed = (now - S.t0) / 1000;
    const fatigue = Math.min(1, Math.min(1, elapsed / 3600) * (0.4 - Math.min(m.ear, 0.4)) * 2.5);
    const row = {
      head_pitch: m.pitch, head_yaw: m.yaw, head_roll: m.roll, ear_score: m.ear, mar_score: m.mar, brow_dist: m.brow,
      delta_yaw: m.yaw - S.last.yaw, delta_pitch: m.pitch - S.last.pitch, delta_ear: m.ear - S.last.ear, gaze_ratio: m.gaze,
      person_detected: m.person, phone_count: S.phoneCount,
      consecutive_frames: Math.round(S.distractedSec * C.POLICY_HZ), is_distracted_label: distracted, fatigue_score: fatigue,
    };
    S.last = { yaw: m.yaw, pitch: m.pitch, ear: m.ear };
    S.window.push(C.scaleRow(row)); if (S.window.length > C.WINDOW_SIZE) S.window.shift();
    S.log.push({ session_id: S.sessionId, timestamp: elapsed.toFixed(2), ...row });
    if (S.log.length > 20000) S.log.shift();
    S.history.push(distracted); if (S.history.length > 180) S.history.shift();

    if (S.window.length === C.WINDOW_SIZE) {
      const state = [...C.windowFeatures(S.window),
        ...C.encodeContext(settings.age, new Date().getHours() + new Date().getMinutes() / 60, settings.occupation, elapsed)];
      const { action, q } = net.act(state);
      decision = { raw: action, q, final: applyOverrides(action, S.distractedSec, fatigue), fatigue };
      S.decision = decision;
    }
  }
  renderStudy(m, reasons, decision);
  actStudy(decision);
}

// Overrides from focus_env.FocusAgentInference, translated from 10-fps frame counts to seconds.
function applyOverrides(raw, sec, fatigue) {
  let a = C.ACTION_NAMES[raw];
  if (sec < 5 && !(fatigue > 0.8 && a === 'Suggest Short Break')) a = 'Do Nothing';
  if (sec >= 90) a = 'Sound Alert';
  else if (sec >= 9 && a === 'Do Nothing') a = 'Gentle Visual Reminder';
  if ((a === 'Sound Alert' || a === 'Suggest Short Break') && sec < 90) a = sec >= 9 ? 'Gentle Visual Reminder' : 'Do Nothing';
  if (a === 'Play Focus Music') a = 'Do Nothing';
  return a;
}

const NUDGE_TEXT = {
  'Gentle Visual Reminder': 'Bạn đang mất tập trung một lúc rồi. Quay lại việc đang làm nhé.',
  'Dim Screen': 'Màn hình sẽ tối đi một chút để nhắc bạn quay lại.',
  'Suggest Short Break': 'Có vẻ bạn đã mệt. Nghỉ 5 phút rồi quay lại sẽ hiệu quả hơn.',
};
function actStudy(d) {
  const now = performance.now();
  $('dim').hidden = !(d && d.final === 'Dim Screen');
  if (!d) return;
  if (d.final === 'Sound Alert' && !S.hardShown) {
    S.hardShown = true; $('hardNudge').hidden = false; beep(660, 0.4);
  } else if (NUDGE_TEXT[d.final] && now - S.lastToastT > 15000) {
    S.lastToastT = now; toast(NUDGE_TEXT[d.final], 'warn', 5000); beep(520, 0.12, 0.05);
  }
}
$('hardAck').addEventListener('click', () => { $('hardNudge').hidden = true; S.distractedSec = 0; S.hardShown = false; });

// ── DRIVER MODE: deterministic DMS rules, no learned policy ───────────────
function driverStep(m, dt, now) {
  const D = C.DRIVER, t = now / 1000;
  const faceSeen = m.ear > 0;
  const closed = faceSeen && m.ear < earThreshold();
  S.closedSec = closed ? S.closedSec + dt : 0;
  S.perclos.push([t, closed ? 1 : 0]);
  while (S.perclos.length && S.perclos[0][0] < t - D.perclosWindowSec) S.perclos.shift();
  const perclos = S.perclos.reduce((s, p) => s + p[1], 0) / Math.max(1, S.perclos.length);

  const offRoad = !faceSeen || Math.abs(m.yaw) > 30 || m.pitch > 25 || m.pitch < -20;
  S.offRoadSec = offRoad && !closed ? S.offRoadSec + dt : 0;
  S.phoneSec = S.phoneCount > 0 ? S.phoneSec + dt : 0;

  if (m.mar > D.yawnMar) S.yawnOpenSec += dt;
  else { if (S.yawnOpenSec >= D.yawnMinSec) S.yawns.push(t); S.yawnOpenSec = 0; }
  S.yawns = S.yawns.filter(y => y > t - D.yawnWindowSec);

  const coverage = S.perclos.length && (S.perclos[S.perclos.length - 1][0] - S.perclos[0][0]) / D.perclosWindowSec;
  let level = 'ok', msg = 'Tỉnh táo, mắt hướng về phía trước';
  if (S.yawns.length >= D.yawnWarnCount) { level = 'warn'; msg = `Ngáp ${S.yawns.length} lần trong 5 phút. Cân nhắc dừng nghỉ.`; }
  if (coverage > 0.5 && perclos > D.perclosWarn) { level = 'warn'; msg = `Dấu hiệu buồn ngủ (PERCLOS ${(perclos * 100).toFixed(0)}%)`; }
  if (coverage > 0.5 && perclos > D.perclosSevere) { level = 'alarm'; msg = 'Buồn ngủ nặng. Hãy dừng xe ở nơi an toàn.'; }
  if (S.offRoadSec > D.offRoadSec) { level = 'alarm'; msg = faceSeen ? 'Mắt rời khỏi mặt đường!' : 'Không thấy khuôn mặt tài xế!'; }
  if (S.phoneSec > D.phoneSec) { level = 'alarm'; msg = 'Đang dùng điện thoại khi lái!'; }
  if (S.closedSec > D.microsleepSec) { level = 'alarm'; msg = 'NGỦ GẬT! Mở mắt ra!'; }

  if (level === 'alarm' && t - S.lastAlarmT > 0.6) { S.lastAlarmT = t; beep(880, 0.25, 0.25); }
  if (now - S.lastPolicyT > 333) { S.lastPolicyT = now; S.history.push(level === 'ok' ? 0 : 1); if (S.history.length > 180) S.history.shift(); }
  renderDriver(m, { perclos, coverage, level, msg });
}

// ── rendering ─────────────────────────────────────────────────────────────
function drawOverlay(lm) {
  const w = overlay.width, h = overlay.height;
  octx.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  if (lm) {
    octx.fillStyle = css.getPropertyValue('--mesh').trim();
    for (let i = 0; i < lm.length; i += 3) octx.fillRect(lm[i].x * w - 1, lm[i].y * h - 1, 2, 2);
    octx.fillStyle = css.getPropertyValue('--accent').trim();
    [33, 133, 362, 263, 468, 473, 13, 14, 61, 291].forEach(i => lm[i] && octx.fillRect(lm[i].x * w - 2, lm[i].y * h - 2, 4, 4));
  }
  octx.strokeStyle = css.getPropertyValue('--bad').trim(); octx.lineWidth = 3;
  (S.phoneBoxes || []).forEach(b => octx.strokeRect(b.originX, b.originY, b.width, b.height));
}

function setStatus(text, level) {
  const el = $('status'); el.textContent = text; el.dataset.level = level;
}
function fmt(v, d = 1) { return Number.isFinite(v) ? v.toFixed(d) : '–'; }

function renderCommon(m) {
  $('mPitch').textContent = fmt(m.pitch) + '°';
  $('mYaw').textContent = fmt(m.yaw) + '°';
  $('mEar').textContent = fmt(m.ear, 3);
  $('mMar').textContent = fmt(m.mar, 2);
  $('mGaze').textContent = fmt(m.gaze, 2);
  $('mPhone').textContent = S.phoneCount ? 'Có' : 'Không';
  const sec = Math.floor((performance.now() - S.t0) / 1000);
  $('elapsed').textContent = `${String(Math.floor(sec / 60)).padStart(2, '0')}:${String(sec % 60).padStart(2, '0')}`;
  drawStrip();
}

function renderStudy(m, reasons, d) {
  renderCommon(m);
  const s = S.distractedSec;
  if (m.person === 0) setStatus('Vắng mặt', 'alarm');
  else if (s < 3) setStatus('Tập trung', 'ok');
  else if (s < 9) setStatus('Cảnh báo sớm', 'warn');
  else setStatus('Mất tập trung', 'alarm');
  $('reason').textContent = reasons.length ? reasons.join(' · ') : 'Đang nhìn vào màn hình';
  $('timer').textContent = `${fmt(s)} s`;
  if (!d) { $('rawAction').textContent = 'Đang gom 10 mẫu…'; return; }
  $('rawAction').textContent = C.ACTION_NAMES[d.raw];
  $('finalAction').textContent = d.final;
  $('overridden').hidden = C.ACTION_NAMES[d.raw] === d.final;
  $('fatigue').textContent = fmt(d.fatigue, 2);
  const max = Math.max(...d.q), min = Math.min(...d.q);
  d.q.forEach((v, i) => {
    const bar = $('q' + i);
    bar.style.setProperty('--w', `${max === min ? 100 : 15 + 85 * (v - min) / (max - min)}%`);
    bar.dataset.best = i === d.raw ? '1' : '0';
    bar.querySelector('b').textContent = v.toFixed(2);
  });
}

function renderDriver(m, r) {
  renderCommon(m);
  setStatus(r.level === 'ok' ? 'Tỉnh táo' : r.level === 'warn' ? 'Cảnh báo' : 'NGUY HIỂM', r.level);
  $('reason').textContent = r.msg;
  $('timer').textContent = `${fmt(S.offRoadSec)} s`;
  $('dPerclos').textContent = r.coverage > 0.5 ? `${(r.perclos * 100).toFixed(0)}%` : `đang đo (${Math.round(r.coverage * 100)}%)`;
  $('dPerclosBar').style.setProperty('--w', `${Math.min(100, r.perclos / 0.4 * 100)}%`);
  $('dClosed').textContent = `${fmt(S.closedSec)} s`;
  $('dOff').textContent = `${fmt(S.offRoadSec)} s`;
  $('dYawn').textContent = String(S.yawns.length);
  document.body.dataset.alarm = r.level === 'alarm' ? '1' : '0';
}

function drawStrip() {
  const c = $('strip'), x = c.getContext('2d');
  const W = c.width = c.clientWidth * devicePixelRatio, H = c.height = c.clientHeight * devicePixelRatio;
  const css = getComputedStyle(document.documentElement);
  x.clearRect(0, 0, W, H);
  const n = 180, bw = W / n;
  S.history.forEach((v, i) => {
    x.fillStyle = css.getPropertyValue(v ? '--bad' : '--good').trim();
    x.globalAlpha = v ? 0.9 : 0.35;
    x.fillRect((n - S.history.length + i) * bw, v ? 0 : H * 0.6, Math.max(1, bw - 0.5), v ? H : H * 0.4);
  });
  x.globalAlpha = 1;
}

// ── small utilities ───────────────────────────────────────────────────────
let audio;
function beep(freq, dur, gain = 0.15) {
  if (!settings.sound) return;
  audio ??= new AudioContext();
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = freq; g.gain.value = gain; o.connect(g).connect(audio.destination);
  o.start(); o.stop(audio.currentTime + dur);
}
function toast(text, kind = 'info', ms = 4000) {
  const el = document.createElement('div'); el.className = 'toast'; el.dataset.kind = kind; el.textContent = text;
  $('toasts').append(el); setTimeout(() => el.remove(), ms);
}
function setLoading(text) { $('loading').hidden = !text; if (text) $('loadingText').textContent = text; }

function setMode(next) {
  mode = next;
  document.body.dataset.mode = next; document.body.dataset.alarm = '0';
  document.querySelectorAll('[data-tab]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === next)));
  $('dim').hidden = true; $('hardNudge').hidden = true;
  S.history = []; S.distractedSec = 0; S.decision = null;
  $('timerLabel').textContent = next === 'study' ? 'Thời gian mất tập trung' : 'Thời gian rời mắt khỏi đường';
}
document.querySelectorAll('[data-tab]').forEach(b => b.addEventListener('click', () => setMode(b.dataset.tab)));

function downloadCsv() {
  if (!S.log.length) { toast('Chưa có dữ liệu. Bật camera ở chế độ Học tập trước.', 'info'); return; }
  const cols = ['session_id','timestamp','head_pitch','head_yaw','head_roll','ear_score','mar_score','brow_dist','person_detected',
    'phone_count','consecutive_frames','is_distracted_label','delta_yaw','delta_pitch','delta_ear','gaze_ratio','fatigue_score'];
  const csv = [cols.join(','), ...S.log.map(r => cols.map(c => typeof r[c] === 'number' ? +r[c].toFixed(5) : r[c]).join(','))].join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  a.download = `focus_features_${S.sessionId}.csv`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

$('startBtn').addEventListener('click', start);
$('stopBtn').addEventListener('click', stop);
$('calibBtn').addEventListener('click', startCalibration);
$('csvBtn').addEventListener('click', downloadCsv);
setMode('study');
if (!window.isSecureContext) toast('Camera chỉ chạy trên HTTPS hoặc localhost. Hãy mở qua GitHub Pages hoặc `python -m http.server`.', 'warn', 12000);

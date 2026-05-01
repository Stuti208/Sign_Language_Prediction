/**
 * ISL Sign Language Recognition — app.js
 *
 * Flow:
 *  1. MediaPipe (JS) extracts landmarks every frame in the browser via GPU
 *  2. If hand moving  → buffer 30 frames → POST /api/predict/dynamic → LSTM
 *  3. If hand still   → POST /api/predict/static  → Landmark MLP
 *  4. Majority-vote over last N predictions to smooth output
 */

import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs";

// ── Config ─────────────────────────────────────────────────
const SEQ_LEN          = 30;
const POSE_IDX         = [0, 11, 12, 13, 14, 15, 16];
const MOTION_THRESH    = 0.008;
const PREDICT_EVERY    = 5;         // run prediction every N frames
const STATIC_MIN_CONF  = 0.60;
const DYNAMIC_MIN_CONF = 0.50;
const HOLD_MS          = 2500;      // keep result visible after hand leaves
const VOTE_WIN         = 5;         // majority vote window (static)
const MAX_MOTION_SCALE = 0.05;
const MAX_HIST         = 25;
const MAX_SENT         = 80;

// ── State ──────────────────────────────────────────────────
let handLM, poseLM;
let seqBuf    = [];       // ring buffer (147-dim per frame)
let voteBuf   = [];       // last VOTE_WIN static labels for smoothing
let frameIdx  = 0;
let motion    = 0;
let lastPredMs = 0;
let dispSign  = null;
let dispMode  = null;
let history   = [];
let sentence  = [];
let fetching  = false;
let camActive = false;
let stream    = null;

// ── DOM ────────────────────────────────────────────────────
const g = id => document.getElementById(id);
const video     = g('webcam');
const cvs       = g('overlay');
const ctx       = cvs.getContext('2d');
const loaderEl  = g('loading-overlay');
const loaderMsg = g('loader-msg');
const lpBar     = g('lp-bar');

// ── Hand skeleton ──────────────────────────────────────────
const HAND_CONN = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

// ══════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════
async function init() {
  try {
    setProgress(10); setMsg('Loading MediaPipe Vision…');
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    stepDone('ls-mp'); setProgress(35);

    setMsg('Loading hand detector…');
    handLM = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.4,
      minHandPresenceConfidence:  0.4,
      minTrackingConfidence:      0.4,
    });
    setProgress(65);

    setMsg('Loading pose detector…');
    poseLM = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      minPoseDetectionConfidence: 0.4,
      minPosePresenceConfidence:  0.4,
      minTrackingConfidence:      0.4,
    });
    setProgress(90);

    setMsg('Starting camera…');
    setProgress(90);
    await startCamera();

    stepDone('ls-cam'); stepDone('ls-go');
    setProgress(100);
    setTimeout(() => loaderEl.classList.add('hide'), 600);
  } catch (err) {
    loaderMsg.textContent = '⚠ ' + err.message;
    console.error(err);
  }
}

// ══════════════════════════════════════════════════════════
// CAMERA TOGGLE
// ══════════════════════════════════════════════════════════
g('btn-start').addEventListener('click', startCamera);
g('btn-stop').addEventListener('click',  stopCamera);

async function startCamera() {
  if (camActive) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    video.srcObject = stream;
    // Wait until the video is actually playing — most reliable trigger
    await video.play();

    g('cam-off').classList.add('hidden');
    video.classList.add('vis');
    g('cam-overlays').classList.add('vis');
    g('cam-card').classList.add('active');
    camActive = true;
    setLive(true);

    seqBuf = []; voteBuf = []; frameIdx = 0;
    dispSign = null; dispMode = null;
    resetResultCard();

    requestAnimationFrame(loop);
  } catch (err) {
    alert('Cannot access camera: ' + err.message);
  }
}

function stopCamera() {
  camActive = false;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;
  video.classList.remove('vis');
  g('cam-off').classList.remove('hidden');
  g('cam-overlays').classList.remove('vis');
  g('cam-card').classList.remove('active');
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  setLive(false);
  resetResultCard();
}

// ══════════════════════════════════════════════════════════
// MAIN LOOP
// ══════════════════════════════════════════════════════════
async function loop() {
  if (!camActive) return;
  if (video.readyState >= 2) {
    if (cvs.width !== video.videoWidth) {
      cvs.width  = video.videoWidth;
      cvs.height = video.videoHeight;
    }

    const ts  = performance.now();
    const hR  = handLM.detectForVideo(video, ts);
    const pR  = poseLM.detectForVideo(video, ts);

    const handsOn = hR.landmarks && hR.landmarks.length > 0;

    const kp = extractKP(hR, pR);
    seqBuf.push(kp);
    if (seqBuf.length > SEQ_LEN) seqBuf.shift();
    frameIdx++;

    motion = calcMotion(seqBuf);

    if (!handsOn && Date.now() - lastPredMs > HOLD_MS) {
      dispSign = null; dispMode = null;
    }

    // Predict
    if (!fetching && frameIdx % PREDICT_EVERY === 0) {
      const dyn = motion >= MOTION_THRESH;
      if (dyn && seqBuf.length === SEQ_LEN) {
        fetching = true;
        callDynamic([...seqBuf])
          .then(onResult).catch(console.error)
          .finally(() => (fetching = false));
      } else if (!dyn && handsOn) {
        fetching = true;
        callStatic(hR.landmarks[0])
          .then(onResult).catch(console.error)
          .finally(() => (fetching = false));
      }
    }

    drawScene(hR, pR);
    updateHUD(handsOn);
  }
  requestAnimationFrame(loop);
}

// ══════════════════════════════════════════════════════════
// FEATURE EXTRACTION
// ══════════════════════════════════════════════════════════
function extractKP(hR, pR) {
  let lh = new Array(63).fill(0);
  let rh = new Array(63).fill(0);
  if (hR.handednesses) {
    for (let i = 0; i < hR.handednesses.length; i++) {
      const label  = hR.handednesses[i][0].categoryName;
      const coords = hR.landmarks[i].flatMap(lm => [lm.x, lm.y, lm.z]);
      if (label === 'Left') lh = coords; else rh = coords;
    }
  }
  const pose = new Array(POSE_IDX.length * 3).fill(0);
  if (pR.landmarks && pR.landmarks.length > 0) {
    const lms = pR.landmarks[0];
    POSE_IDX.forEach((idx, j) => {
      if (lms[idx]) {
        pose[j*3] = lms[idx].x; pose[j*3+1] = lms[idx].y; pose[j*3+2] = lms[idx].z;
      }
    });
  }
  return [...pose, ...lh, ...rh];
}

function normLandmarks(lms) {
  const c = lms.map(lm => [lm.x, lm.y, lm.z]);
  const w = c[0];
  const t = c.map(p => [p[0]-w[0], p[1]-w[1], p[2]-w[2]]);
  const sc = Math.sqrt(t[9].reduce((s,v) => s+v*v, 0)) + 1e-6;
  return t.flat().map(v => v/sc);
}

function calcMotion(buf) {
  if (buf.length < 5) return 0;
  const frames = buf.slice(-SEQ_LEN);
  const n = frames.length, N = 126;
  const means = new Array(N).fill(0);
  for (const f of frames) for (let i=0;i<N;i++) means[i] += f[21+i];
  means.forEach((_,i) => (means[i]/=n));
  let total = 0;
  for (let i=0;i<N;i++){
    let v=0;
    for (const f of frames) v += (f[21+i]-means[i])**2;
    total += Math.sqrt(v/n);
  }
  return total/N;
}

// ══════════════════════════════════════════════════════════
// API
// ══════════════════════════════════════════════════════════
async function callStatic(lms) {
  const r = await fetch('/api/predict/static', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ landmarks: normLandmarks(lms) }),
  });
  return r.json();
}

async function callDynamic(seq) {
  const r = await fetch('/api/predict/dynamic', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ sequence: seq.flat() }),
  });
  return r.json();
}

// ══════════════════════════════════════════════════════════
// RESULT HANDLER
// ══════════════════════════════════════════════════════════
function onResult(res) {
  const { label, confidence, top3, mode } = res;
  const minConf = mode === 'static' ? STATIC_MIN_CONF : DYNAMIC_MIN_CONF;
  if (confidence < minConf) return;

  // Static majority vote for smoothing
  if (mode === 'static') {
    voteBuf.push(label);
    if (voteBuf.length > VOTE_WIN) voteBuf.shift();
    const freq = {};
    let best = label, bestN = 0;
    for (const v of voteBuf) { freq[v] = (freq[v]||0)+1; if(freq[v]>bestN){bestN=freq[v];best=v;} }
    if (bestN < Math.ceil(VOTE_WIN / 2)) return;   // not stable enough yet
    if (dispSign === best && dispMode === 'static') return; // no change
    dispSign = best;
  } else {
    voteBuf = [];
    if (dispSign === label && dispMode === 'dynamic') return;
    dispSign = label;
  }

  dispMode = mode;
  lastPredMs = Date.now();

  // History
  history.unshift({ label: dispSign, mode });
  if (history.length > MAX_HIST) history.pop();
  renderChips();

  // Sentence
  if (sentence[sentence.length-1] !== dispSign) {
    sentence.push(dispSign);
    if (sentence.length > MAX_SENT) sentence.shift();
    renderSentence();
  }

  // Update result card
  const card = g('result-card');
  const tag  = g('rc-mode-tag');

  card.className = `result-card ${mode === 'static' ? 's-mode' : 'd-mode'}`;
  tag.className  = `rc-mode-tag ${mode === 'static' ? 's-mode' : 'd-mode'}`;
  tag.textContent = mode === 'static' ? 'STATIC SIGN' : 'DYNAMIC GESTURE';
  g('rc-conf-tag').textContent = `${(confidence*100).toFixed(0)}%`;

  const sign = g('rc-sign');
  sign.textContent = dispSign;
  sign.className   = `rc-sign ${mode === 'static' ? 's-mode' : 'd-mode'}`;
  sign.classList.add('pop');
  void sign.offsetWidth;
  // pop class is removed by animation end listener (set up below)

  g('rc-glow').className = 'rc-glow lit';
  setTimeout(() => g('rc-glow').classList.remove('lit'), 600);

  g('rc-sub').textContent = mode === 'static' ? 'Static sign detected' : 'Dynamic gesture detected';

  const pct = confidence * 100;
  g('rc-conf-pct').textContent = `${pct.toFixed(0)} %`;
  const fill = g('rc-conf-fill');
  fill.style.width = `${pct}%`;
  fill.style.background = mode === 'static'
    ? 'linear-gradient(90deg,#00d4ff,#9b6dff)'
    : 'linear-gradient(90deg,#ff5f2e,#ffaa00)';

  g('rc-alts-body').innerHTML = top3.map(t =>
    `<div class="alt-item"><span>${t.label}</span><span>${(t.confidence*100).toFixed(0)} %</span></div>`
  ).join('');
}

g('rc-sign').addEventListener('animationend', () => g('rc-sign').classList.remove('pop'));

// ══════════════════════════════════════════════════════════
// HUD UPDATE (every frame)
// ══════════════════════════════════════════════════════════
function updateHUD(handsOn) {
  const pct = (seqBuf.length / SEQ_LEN) * 100;
  g('buf-bar').style.width = `${pct}%`;
  g('buf-txt').textContent = `Buffer ${seqBuf.length} / ${SEQ_LEN}`;

  const badge = g('mode-badge');
  const dyn   = motion >= MOTION_THRESH;
  if (handsOn) {
    badge.className        = `cam-mode-badge ${dyn ? 'd-mode' : 's-mode'}`;
    g('mb-icon').textContent = dyn ? '👋' : '✋';
    g('mb-txt').textContent  = dyn ? 'Dynamic' : 'Static';
  } else {
    badge.className          = 'cam-mode-badge';
    g('mb-icon').textContent = '👁';
    g('mb-txt').textContent  = 'No hand';
  }

  const mp = Math.min((motion / MAX_MOTION_SCALE) * 100, 100);
  g('motion-bar').style.width  = `${mp}%`;
  g('motion-val').textContent  = motion.toFixed(4);
  g('motion-needle').style.left = `${(MOTION_THRESH / MAX_MOTION_SCALE) * 100}%`;

  if (!dispSign) resetResultCard();
}

// ══════════════════════════════════════════════════════════
// DRAWING
// ══════════════════════════════════════════════════════════
function drawScene(hR, pR) {
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  const W = cvs.width, H = cvs.height;
  const col = dispMode === 'static' ? '#00d4ff'
            : dispMode === 'dynamic' ? '#ff5f2e'
            : '#44ee88';

  for (const lms of (hR.landmarks || [])) {
    ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.globalAlpha = .65;
    for (const [a,b] of HAND_CONN) {
      ctx.beginPath();
      ctx.moveTo(lms[a].x*W, lms[a].y*H);
      ctx.lineTo(lms[b].x*W, lms[b].y*H);
      ctx.stroke();
    }
    // Landmark dots with glow
    for (let i=0; i<lms.length; i++) {
      const x = lms[i].x*W, y = lms[i].y*H;
      const r = i===0 ? 6 : (i%4===0 ? 5 : 3.5);
      ctx.globalAlpha = .95;
      ctx.fillStyle = col;
      ctx.shadowColor = col; ctx.shadowBlur = 8;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2); ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  // Pose dots
  if (pR.landmarks && pR.landmarks.length > 0) {
    const lms = pR.landmarks[0];
    ctx.fillStyle = '#ffaa00'; ctx.globalAlpha = .7;
    for (const idx of POSE_IDX) {
      if (lms[idx]) {
        ctx.shadowColor = '#ffaa00'; ctx.shadowBlur = 6;
        ctx.beginPath(); ctx.arc(lms[idx].x*W, lms[idx].y*H, 5, 0, Math.PI*2); ctx.fill();
        ctx.shadowBlur = 0;
      }
    }
  }
  ctx.globalAlpha = 1;
}

// ══════════════════════════════════════════════════════════
// RENDER HELPERS
// ══════════════════════════════════════════════════════════
function resetResultCard() {
  g('rc-sign').textContent     = '—';
  g('rc-sign').className       = 'rc-sign';
  g('rc-sub').textContent      = camActive ? 'Show a sign to begin' : 'Start the camera and show a sign';
  g('rc-mode-tag').className   = 'rc-mode-tag';
  g('rc-mode-tag').textContent = 'WAITING';
  g('rc-conf-tag').textContent = '—';
  g('result-card').className   = 'result-card';
  g('rc-conf-pct').textContent = '0 %';
  g('rc-conf-fill').style.width = '0%';
  g('rc-alts-body').innerHTML  =
    '<div class="alt-item"><span>—</span><span>—</span></div>'.repeat(3);
}

function renderChips() {
  const el = g('chips-row');
  if (!history.length) { el.innerHTML = '<span class="placeholder-txt">No predictions yet</span>'; return; }
  el.innerHTML = history.map(h =>
    `<span class="chip ${h.mode==='static'?'s':'d'}">${h.label}</span>`
  ).join('');
}

function renderSentence() {
  const el = g('sentence-box');
  if (!sentence.length) {
    el.innerHTML = '<span class="placeholder-txt">Recognised words will appear here…</span>'; return;
  }
  el.textContent = sentence.join(' ');
  el.scrollTop = el.scrollHeight;
}

// ══════════════════════════════════════════════════════════
// BUTTONS
// ══════════════════════════════════════════════════════════
g('btn-clr-hist').addEventListener('click', () => { history = []; renderChips(); });
g('btn-clr-sent').addEventListener('click', () => { sentence = []; renderSentence(); });
g('btn-speak').addEventListener('click', () => {
  if (!sentence.length) return;
  const utt = new SpeechSynthesisUtterance(sentence.join(' '));
  utt.lang = 'en-IN';
  speechSynthesis.cancel();
  speechSynthesis.speak(utt);
});

// ══════════════════════════════════════════════════════════
// LOADER HELPERS
// ══════════════════════════════════════════════════════════
function setMsg(t)  { loaderMsg.textContent = t; }
function setProgress(p) { lpBar.style.width = `${p}%`; }
function stepDone(id) {
  const el = g(id);
  el.classList.add('done');
  el.querySelector('.lstep-dot').textContent = '';
}
function setLive(on) {
  const pill = g('live-pill'), txt = g('live-txt');
  if (on) { pill.classList.add('on'); txt.textContent = 'Live'; }
  else    { pill.classList.remove('on'); txt.textContent = 'Offline'; }
}

// ══════════════════════════════════════════════════════════
// BOOT
// ══════════════════════════════════════════════════════════
init();

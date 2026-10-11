import { createIcons, ScanEye, Radio, Radar, Maximize, Columns2, PictureInPicture2, Camera, Pause, Play, Navigation, Scan, VideoOff, X, Plug, Search, Cpu, ChevronDown, Crosshair, PanelRightClose, PanelRightOpen, Mic, MicOff, Send } from 'lucide';
import { RobotScene } from './scene.js';
import { normalizeState, normalizeScene, resolveTarget, displayName, distance, sourceUrl, cameraPose, memoryAgeLabel } from './state.js';
import './style.css';
import { ResourcePanel, metricsUrl } from './resources.js';
import { Voice } from './voice.js';

const $ = id => document.getElementById(id);
const icons = { ScanEye, Radio, Radar, Maximize, Columns2, PictureInPicture2, Camera, Pause, Play, Navigation, Scan, VideoOff, X, Plug, Search, Cpu, ChevronDown, Crosshair, PanelRightClose, PanelRightOpen, Mic, MicOff, Send };
const drawIcons = () => createIcons({ icons });
drawIcons();
const resources = new ResourcePanel($('resources'));
let resourceUrl = stored('robot-metrics-url', '');

let scene;
try { scene = new RobotScene($('scene'), $('labels')); }
catch { $('scene-empty').textContent = '3D view unavailable. WebGL is required.'; }
let state, target, selectedId = null, paused = false, source, events, generation = 0, lastStateAt = 0;
let frameController, frameTimer, currentImageUrl, lastImageAt = 0, lastSeq = null, listKey = '';
let mode = 'unknown', receivedValidState = false;
let framePath = '/frame.jpg';
let sceneRequest = 0, fileScene = false, sceneVersion, healVersion, healBusy = false;
let pendingState = null;
let objectFilter = 'all';

function inspector(open) {
  $('app').dataset.inspector = open ? 'open' : 'closed';
  $('memory-panel').hidden = !open;
  const button = $('inspector-toggle');
  button.setAttribute('aria-expanded', String(open));
  button.title = open ? 'Hide object memory' : 'Show object memory';
  button.setAttribute('aria-label', button.title);
  button.innerHTML = `<i data-lucide="${open ? 'panel-right-close' : 'panel-right-open'}"></i>`;
  drawIcons();
}
$('inspector-toggle').onclick = () => inspector($('memory-panel').hidden);
inspector(true);
$('resources-toggle').onclick = () => {
  const collapsed = !$('resource-metrics').hidden;
  $('resource-metrics').hidden = collapsed;
  $('resources-toggle').setAttribute('aria-expanded', String(!collapsed));
  $('resources-toggle').title = collapsed ? 'Expand compute metrics' : 'Collapse compute metrics';
  $('resources-toggle').setAttribute('aria-label', $('resources-toggle').title);
};
$('object-search').oninput = () => renderMemory();
$('clear-search').onclick = () => { $('object-search').value = ''; renderMemory(); $('object-search').focus(); };
document.querySelectorAll('[data-filter]').forEach(button => button.onclick = () => {
  objectFilter = button.dataset.filter;
  document.querySelectorAll('[data-filter]').forEach(item => {
    item.classList.toggle('active', item === button);
    item.setAttribute('aria-pressed', String(item === button));
  });
  renderMemory();
});
$('follow-target').onclick = () => { selectedId = null; display(true); };

// Commands for the robot (POST /prompt, like the Jetson's console): typed, or spoken after "hey rover".
let voiceNote = '', voiceNoteAt = 0;
function voiceStatus(text, kind = '') {
  voiceNote = text; voiceNoteAt = Date.now();
  $('voice-status').textContent = text;
  $('voice-status').dataset.kind = kind;
}
async function sendCommand(text) {
  text = text.trim();
  if (!text) return;
  if (!source) { voiceStatus(`Not connected: "${text}" not sent`, 'error'); return; }
  voiceStatus(`\u2192 "${text}"`, 'sent');
  try {
    const response = await fetch(`${source}/prompt`, { method: 'POST', body: text, signal: AbortSignal.timeout(4000) });
    if (!response.ok) throw new Error(String(response.status));
  } catch { voiceStatus(`Couldn't send "${text}": is the robot connected?`, 'error'); }
}
const VOICE_STATES = {
  off: 'Voice off', listening: 'Listening for "hey rover"', armed: 'Yes? Say the command',
  network: 'Speech recognition needs internet (Chrome sends audio to Google): retrying',
  blocked: 'Microphone blocked: allow it in the address bar, then click the mic again',
  insecure: 'The mic needs https or localhost: ssh -L 8080:localhost:8080 knighthacks@<jetson>, then http://localhost:8080',
  unsupported: 'No speech recognition in this browser: use Chrome (or type the command)',
};
const voice = new Voice({
  onCommand: text => sendCommand(text),
  onHeard: (text, final) => { if (!final || Date.now() - voiceNoteAt > 300) voiceStatus(final ? `Heard "${text}"` : `\u2026 ${text}`, final ? 'heard' : 'interim'); },
  onState: state => {
    $('voice').dataset.state = state;
    const on = state === 'listening' || state === 'armed' || state === 'network';
    $('voice-toggle').setAttribute('aria-pressed', String(on));
    $('voice-toggle').innerHTML = `<i data-lucide="${on ? 'mic' : 'mic-off'}"></i>`;
    drawIcons();
    voiceStatus(VOICE_STATES[state] || state, state === 'blocked' || state === 'insecure' || state === 'unsupported' ? 'error' : '');
  },
});
$('voice-toggle').onclick = () => (voice.enabled ? voice.stop() : voice.start());
$('command-form').onsubmit = event => { event.preventDefault(); sendCommand($('command-input').value); $('command-input').value = ''; };

function projection(view) {
  $('robot-view').parentElement.style.setProperty('--selected-index', view === 'robot' ? 0 : 1);
  scene?.setView(view);
  $('projection-label').textContent = view === 'map' ? 'MEMORY MAP' : 'ROBOT PERSPECTIVE';
  $('spatial-panel').setAttribute('aria-label', view === 'map' ? 'Top-down memory map' : 'Robot perspective visualization');
  for (const [id, name] of [['map-view', 'map'], ['robot-view', 'robot']]) {
    $(id).setAttribute('aria-pressed', String(view === name));
    $(id).classList.toggle('active', view === name);
  }
}
$('map-view').onclick = () => projection('map');
$('robot-view').onclick = () => projection('robot');
projection('robot');

async function fetchHeal(token, version) {
  healBusy = true;
  try {
    const response = await fetch(`${source}/scene/heal`, { cache: 'no-store', signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error('No heal');
    const data = await response.json();
    if (token !== generation || !Array.isArray(data.points)) return;
    healVersion = version;
    scene?.setHeal(data.points.filter(p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite)), data.age_s);
  } catch { /* the next message retries */ } finally { healBusy = false; }
}

async function fetchScene(token) {
  if (fileScene) return;
  const request = ++sceneRequest;
  try {
    const response = await fetch(`${source}/scene`, { cache: 'no-store', signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error(response.status === 404 ? 'No static scene' : 'Scene unavailable');
    const data = normalizeScene(await response.json());
    if (token !== generation || request !== sceneRequest) return;
    scene?.setScene(data);
    $('scene-status').textContent = `${data.cloud.points.length.toLocaleString()} scan points`;
    display();
  } catch (error) {
    if (token !== generation || request !== sceneRequest) return;
    scene?.setScene(null);
    $('scene-status').textContent = error.message;
    display();
  }
}

$('scene-file').onchange = async () => {
  const file = $('scene-file').files[0];
  if (!file) return;
  const request = ++sceneRequest;
  try {
    if (file.size > 25 * 1024 * 1024) throw new Error('Scene file exceeds 25 MB');
    const data = normalizeScene(JSON.parse(await file.text()));
    if (request !== sceneRequest) return;
    fileScene = true; scene?.setScene(data); display();
    $('scene-status').textContent = `${file.name}: ${data.cloud.points.length.toLocaleString()} points`;
  } catch (error) { $('scene-status').textContent = error.message; }
};

function stored(key, fallback) { try { return localStorage.getItem(key) || fallback; } catch { return fallback; } }
function persist(key, value) { try { localStorage.setItem(key, value); } catch { /* Storage is optional. */ } }

function layout(name) {
  if (!['split', 'spatial', 'camera'].includes(name)) name = 'split';
  $('workspace').dataset.layout = name;
  document.querySelector('[aria-label="View layout"]').style.setProperty('--selected-index', ['split', 'spatial', 'camera'].indexOf(name));
  document.querySelectorAll('[data-layout]').forEach(button => {
    if (button.tagName !== 'BUTTON') return;
    const active = button.dataset.layout === name;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  persist('robot-layout', name);
}
document.querySelectorAll('button[data-layout]').forEach(button => button.onclick = () => layout(button.dataset.layout));
layout(stored('robot-layout', 'spatial'));

function status(label, good = false) {
  $('connection-status').textContent = label;
  $('status-dot').className = `status-dot ${good ? 'good' : 'bad'}`;
}

// The newest Qwen-3D answer while questions are asked: what was asked, what it picked, how sure, how fast,
// and its top scores (the first one picked).
function decisionCard() {
  let card = $('qwen-card');
  const a = state?.phase === 'ask' ? state.qwen?.answers?.at(-1) : null;
  if (!a) { if (card) card.hidden = true; return; }
  if (!card) {
    card = document.createElement('div'); card.id = 'qwen-card';
    card.innerHTML = '<div class="qc-head">QWEN-3D <span class="qc-query"></span></div><div class="qc-pick"></div>'
      + '<div class="qc-info"></div><div class="qc-bars"></div>';
    $('spatial-panel').append(card);
  }
  card.hidden = false;
  card.querySelector('.qc-query').textContent = `"${a.query}"`;
  card.querySelector('.qc-pick').textContent = `\u2192 ${a.target || 'object'}  ${Math.round(100 * a.score)}%`;
  const info = [`${Number(a.seconds).toFixed(1)} s on the Jetson`, '8-bit 3B VLM'];
  if (Number.isFinite(a.ram_gb)) info.push(`${a.ram_gb.toFixed(1)} GB`);
  if (Number.isFinite(a.tokens)) info.push(`${a.tokens} tokens`);
  card.querySelector('.qc-info').textContent = info.join(' \u00b7 ');
  const top = (Array.isArray(a.top5) && a.top5.length ? a.top5 : [a.score]).slice(0, 5);
  card.querySelector('.qc-bars').replaceChildren(...top.map((score, i) => {
    const bar = document.createElement('div'); bar.className = `qc-bar${i ? '' : ' picked'}`;
    bar.innerHTML = `<span></span><i style="width:${Math.round(100 * Math.min(1, score))}%"></i>`;
    bar.firstChild.textContent = score.toFixed(2);
    return bar;
  }));
}

function robotNote() {           // what the robot is doing about the last command, under the command bar
  if (Date.now() - voiceNoteAt < 2500) return;
  const q = state?.qwen, drive = state?.drive;
  let note = '';
  if (Number.isFinite(q?.driving_in)) note = `Driving to Qwen-3D's answer in ${q.driving_in.toFixed(1)} s: say "hey rover, stop" to cancel`;
  else if (q?.thinking) note = `Qwen-3D thinking: "${q.thinking}"`;
  else if (drive && drive.state && !['idle', 'stopped'].includes(drive.state)) note = `Rover ${drive.state}${drive.target_id != null ? ` to ${drive.target_id}` : ''}`;
  else if (state?.phase === 'loading') note = `Loading: ${(state.loading || []).join(', ')}`;
  else if (state?.phase === 'ready') note = 'Ready: say "hey rover, scan" (or press Enter on the Jetson)';
  if (note && $('voice-status').textContent !== note) { $('voice-status').textContent = note; $('voice-status').dataset.kind = 'robot'; }
}

function display(force = false) {
  if (!state || (paused && !force)) return;
  decisionCard();
  robotNote();
  if (selectedId !== null && ![...state.objects, ...state.balls].some(object => String(object.id) === selectedId)) selectedId = null;
  target = resolveTarget(state, selectedId);
  const geometry = scene?.update(state, target);
  $('spatial-panel').dataset.stateIndex = state.replay?.index ?? '';
  if (scene) {
    $('scene-empty').hidden = geometry.ready;
    $('scene-empty').querySelector('span').textContent = 'Waiting for camera pose';
  }
  $('geometry-type').textContent = geometry?.clouds ? (mode === 'synthetic' ? 'SIMULATED POINTS' : 'POINT CLOUD') : 'BOUNDING BOXES';
  $('fps').textContent = Number.isFinite(state.fps) ? `${state.fps.toFixed(0)} fps` : '-- fps';
  $('phase').textContent = String(state.phase || (mode === 'replay' ? 'recording' : mode === 'synthetic' ? 'simulation' : 'observation')).toUpperCase();
  $('target-label').textContent = target ? displayName(target) : 'No target';
  $('target-distance').replaceChildren(document.createTextNode(target && cameraPose(state.camera) ? distance(state.camera.position, target.position).toFixed(2) : '--'));
  const unit = document.createElement('small'); unit.textContent = ' m'; $('target-distance').append(unit);
  $('target-state').textContent = target ? (target.state === 'INSIDE' ? `Inside cup ${target.inside}` : String(target.state || 'Unknown').replaceAll('_', ' ')) : 'Awaiting target';
  $('target-memory').textContent = target ? memoryAgeLabel(target, state.t) : 'Last seen unknown';
  $('follow-target').setAttribute('aria-pressed', String(selectedId === null));
  renderMemory();
}

function renderMemory() {
  const all = state ? [...state.objects, ...state.balls] : [];
  const observed = all.filter(object => object.state === 'VISIBLE').length;
  $('object-count').textContent = all.length;
  $('observed-count').textContent = state ? observed : '--';
  $('unseen-count').textContent = state ? all.length - observed : '--';
  const query = $('object-search').value.trim().toLowerCase();
  $('clear-search').hidden = !$('object-search').value;
  const filtered = all.filter(object => (objectFilter === 'all' || (object.state === 'VISIBLE') === (objectFilter === 'observed'))
    && `${displayName(object)} ${object.id}`.toLowerCase().includes(query));
  $('memory-empty').hidden = filtered.length > 0;
  $('memory-empty').textContent = all.length ? 'No matching objects' : 'Waiting for objects';
  const key = JSON.stringify(filtered.map(o => [o.id, o.color, o.label]));
  if (key !== listKey) {
    listKey = key;
    $('object-list').replaceChildren(...filtered.map(object => {
      const button = document.createElement('button');
      button.className = 'object-item';
      const dot = document.createElement('span'); dot.className = 'object-dot';
      const label = document.createElement('span'); label.textContent = displayName(object); label.title = label.textContent;
      label.className = 'object-name';
      const sub = document.createElement('small'); sub.className = 'object-state';
      const age = document.createElement('small'); age.className = 'memory-age';
      age.title = 'Time since last observation at the displayed frame';
      const range = document.createElement('span'); range.className = 'object-range';
      button.append(dot, label, sub, age, range);
      button.onclick = () => { selectedId = String(object.id); display(true); };
      return button;
    }));
  }
  Array.from($('object-list').children).forEach((button, index) => {
    const object = filtered[index];
    button.setAttribute('aria-pressed', String(object.id === target?.id));
    button.querySelector('.object-dot').style.background = { yellow: '#f4d06c', green: '#91dc83', red: '#ff8877', purple: '#d89fec' }[object.color] || (object.state === 'VISIBLE' ? '#73e6d3' : '#f4d06c');
    button.querySelector('.object-state').textContent = object.state === 'INSIDE' ? `Inside cup ${object.inside}` : String(object.state || 'Unknown').replaceAll('_', ' ').toLowerCase();
    button.querySelector('.memory-age').textContent = memoryAgeLabel(object, state.t);
    button.querySelector('.object-range').textContent = cameraPose(state.camera) ? `${distance(state.camera.position, object.position).toFixed(2)} m` : '-- m';
  });
}

async function pollFrame(token) {
  if (token !== generation) return;
  if (paused || document.hidden) { frameTimer = setTimeout(() => pollFrame(token), 250); return; }
  frameController = new AbortController();
  const timer = setTimeout(() => frameController?.abort(), 5000);
  let next = 100;
  const snapshot = pendingState;
  try {
    const url = new URL(`${source}${framePath}`);
    url.searchParams.set('_', Date.now());
    if (snapshot?.replay?.camera) url.searchParams.set('index', snapshot.replay.index);
    const response = await fetch(url, { cache: 'no-store', signal: frameController.signal });
    if (!response.ok || !response.headers.get('Content-Type')?.startsWith('image/')) throw new Error('Camera unavailable');
    if (snapshot?.replay?.camera && response.headers.get('X-State-Index') !== String(snapshot.replay.index)) throw new Error('Camera synchronization mismatch');
    const blob = await response.blob();
    if (token !== generation || paused) return;
    const imageUrl = URL.createObjectURL(blob);
    const image = new Image(); image.src = imageUrl;
    try { await image.decode(); } catch (error) { URL.revokeObjectURL(imageUrl); throw error; }
    if (token !== generation || paused) { URL.revokeObjectURL(imageUrl); return; }
    $('camera-frame').src = imageUrl;
    $('camera-frame').classList.add('ready');
    $('camera-frame').dataset.stateIndex = response.headers.get('X-State-Index') ?? '';
    $('camera-empty').hidden = true;
    if (currentImageUrl) URL.revokeObjectURL(currentImageUrl);
    currentImageUrl = imageUrl;
    if (snapshot?.replay?.camera) { state = snapshot; display(); }
    const seq = response.headers.get('X-Seq');
    if (seq === null || seq !== lastSeq) lastImageAt = performance.now();
    lastSeq = seq;
    $('frame-seq').textContent = `FRAME ${seq ?? '--'}`;
    const age = response.headers.get('X-Age-Ms');
    $('frame-age').textContent = snapshot?.replay?.camera ? `REC ${snapshot.replay.frame_t.toFixed(2)} s` : age === null ? 'Age unavailable' : `${age} ms`;
  } catch (error) {
    if (token !== generation) return;
    next = 1500;
    $('camera-empty').hidden = false;
    $('camera-empty').querySelector('span').textContent = 'Camera unavailable. Retrying...';
    $('camera-frame').classList.remove('ready');
  } finally {
    clearTimeout(timer);
    if (token === generation) frameTimer = setTimeout(() => pollFrame(token), next);
  }
}

async function connect(url) {
  const token = ++generation;
  ++sceneRequest; fileScene = false; sceneVersion = healVersion = undefined; scene?.setHeal([]); $('scene-file').value = '';
  source = sourceUrl(url);
  try { resources.connect(metricsUrl(resourceUrl, source)); }
  catch { resourceUrl = ''; resources.connect(`${source}/metrics`); }
  events?.close(); clearTimeout(frameTimer); frameController?.abort();
  lastStateAt = 0; lastImageAt = 0; lastSeq = null; receivedValidState = false; selectedId = null;
  state = null; pendingState = null; listKey = ''; mode = 'unknown';
  if (scene) { scene.lastState = null; scene.setScene(null); scene.clear(); }
  $('object-list').replaceChildren(); $('object-count').textContent = '0';
  renderMemory();
  $('follow-target').setAttribute('aria-pressed', 'true');
  $('target-label').textContent = 'No target'; $('target-distance').textContent = '-- m'; $('target-state').textContent = 'Awaiting state';
  $('target-memory').textContent = 'Last seen unknown';
  $('camera-frame').classList.remove('ready'); $('camera-frame').removeAttribute('src');
  if (currentImageUrl) URL.revokeObjectURL(currentImageUrl); currentImageUrl = null;
  $('camera-empty').hidden = false; $('scene-empty').hidden = false;
  $('camera-empty').querySelector('span').textContent = 'Waiting for camera';
  $('source-name').textContent = new URL(source).host;
  $('mode-label').textContent = 'CONNECTING'; $('geometry-type').textContent = 'WAITING'; $('fps').textContent = '-- fps';
  $('phase').textContent = 'WAITING';
  $('stale-banner').hidden = true;
  persist('robot-source', source); persist('robot-frame-path', framePath);
  status('CONNECTING');
  events = new EventSource(`${source}/state`);
  fetchScene(token);
  events.onmessage = message => {
    if (token !== generation) return;
    try {
      const previousPhase = pendingState?.phase, wasReady = !!pendingState?.qwen?.ready;
      pendingState = normalizeState(JSON.parse(message.data));
      if (!pendingState.replay?.camera && !paused) state = pendingState;
      // The live server counts /scene versions: fetch whenever it changes (it is built while tracking starts).
      // Servers without it: fetch when tracking starts.
      const version = pendingState.scene_version;
      if (Number.isFinite(version) ? version > 0 && version !== sceneVersion : pendingState.phase === 'track' && previousPhase !== 'track') {
        sceneVersion = version; fetchScene(token);
      }
      // The memory graph marks Qwen-3D arriving and leaving (its memory handed to YOLOE).
      if (pendingState.qwen?.ready && !wasReady) resources.mark('Qwen-3D loaded');
      if (previousPhase === 'ask' && pendingState.phase !== 'ask') resources.mark('Qwen-3D freed for YOLOE');
      // The scan's holes (where objects stood) fill in from live depth: fetch the patch when it grows.
      if (Number.isFinite(pendingState.heal_version) && pendingState.heal_version !== healVersion && !healBusy) fetchHeal(token, pendingState.heal_version);
      lastStateAt = performance.now(); receivedValidState = true;
      if (pendingState.source === 'synthetic' || pendingState.source === 'replay') mode = pendingState.source;
      const label = mode === 'synthetic' ? 'SIMULATION' : mode === 'replay' ? 'REPLAY' : 'CONNECTED';
      status(paused ? 'FROZEN' : label, !paused);
      $('mode-label').textContent = label;
      $('camera-type').textContent = pendingState.replay?.camera ? (framePath === '/frame-raw.jpg' ? 'RECORDED RGB' : 'REPLAY OVERLAY') : mode === 'synthetic' ? 'SIMULATED' : mode === 'replay' ? 'NO RECORDING' : 'X-RAY';
      $('camera-frame').alt = pendingState.replay?.camera ? 'Recorded front camera with replay telemetry' : 'Robot front camera with x-ray overlay';
      $('stream-detail').textContent = mode === 'synthetic' ? 'Local simulation' : mode === 'replay' ? 'Recorded telemetry' : 'Robot stream';
      $('stale-banner').hidden = true;
      display();
    } catch { status('INVALID STATE'); }
  };
  events.onerror = () => { if (token === generation) status('RECONNECTING'); };
  pollFrame(token);
  try {
    const response = await fetch(`${source}/viewer-info`, { signal: AbortSignal.timeout(2500) });
    if (response.ok) {
      const info = await response.json();
      if (token === generation && ['synthetic', 'replay', 'live'].includes(info.mode)) { mode = info.mode; display(); }
    }
  } catch { /* The Jetson's original server need not expose viewer-info. */ }
}

$('pause').onclick = () => {
  paused = !paused;
  $('pause').setAttribute('aria-pressed', String(paused));
  $('pause').setAttribute('aria-label', paused ? 'Resume display' : 'Freeze display');
  $('pause').title = paused ? 'Resume display' : 'Freeze display';
  $('pause').innerHTML = `<i data-lucide="${paused ? 'play' : 'pause'}"></i>`; drawIcons();
  if (paused) status('FROZEN');
  else {
    if (pendingState && !pendingState.replay?.camera) state = pendingState;
    display();
  }
};
$('fullscreen').onclick = async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
  catch { $('fullscreen').title = 'Fullscreen unavailable'; }
};
$('source-button').onclick = () => {
  $('stream-url').value = source; $('frame-path').value = framePath; $('connection-error').textContent = '';
  $('metrics-url').value = resourceUrl;
  $('connection-dialog').showModal();
};
$('close-dialog').onclick = () => $('connection-dialog').close();
$('local-source').onclick = () => { $('stream-url').value = location.origin; $('frame-path').value = '/frame.jpg'; };
$('connection-form').onsubmit = event => {
  event.preventDefault();
  try {
    const value = sourceUrl($('stream-url').value);
    const resourceValue = $('metrics-url').value.trim();
    const resourceEndpoint = metricsUrl(resourceValue, value);
    if (location.protocol === 'https:' && resourceEndpoint.startsWith('http:')) throw new Error('Open the viewer over HTTP for HTTP Jetson metrics.');
    const path = $('frame-path').value.trim();
    if (!/^\/(?!\/)[^?#\s]*$/.test(path)) throw new Error('Use an endpoint path such as /frame.jpg.');
    if (location.protocol === 'https:' && value.startsWith('http:')) throw new Error('Open this viewer over HTTP to connect to an HTTP robot.');
    resourceUrl = resourceValue; persist('robot-metrics-url', resourceUrl);
    framePath = path; paused = false;
    $('pause').setAttribute('aria-pressed', 'false'); $('pause').setAttribute('aria-label', 'Freeze display');
    $('pause').title = 'Freeze display'; $('pause').innerHTML = '<i data-lucide="pause"></i>'; drawIcons();
    connect(value); $('connection-dialog').close();
  } catch (error) { $('connection-error').textContent = error.message; }
};

setInterval(() => {
  const age = lastStateAt ? (performance.now() - lastStateAt) / 1000 : Infinity;
  $('last-update').textContent = Number.isFinite(age) ? `STATE ${age.toFixed(1)} s ago${paused ? ' / DISPLAY FROZEN' : ''}` : 'No telemetry';
  if (age > 3) {
    $('stale-banner').hidden = false;
    $('stale-banner').textContent = receivedValidState ? 'Connection interrupted. Showing last received state.' : 'Waiting for a state stream. Check the robot connection.';
    status('OFFLINE');
  }
  if (!paused && lastImageAt && performance.now() - lastImageAt > 5000) $('frame-age').textContent = 'STALE FRAME';
}, 500);

let initial = stored('robot-source', location.origin);
try { initial = sourceUrl(initial); } catch { initial = location.origin; }
const savedPath = stored('robot-frame-path', '/frame.jpg');
framePath = /^\/(?!\/)[^?#\s]*$/.test(savedPath) ? savedPath : '/frame.jpg';
connect(initial);

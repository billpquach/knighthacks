import { createIcons, ScanEye, Radio, Maximize, Columns2, PictureInPicture2, Camera, Pause, Play, Navigation, Scan, VideoOff, Layers3, X, Plug } from 'lucide';
import { RobotScene } from './scene.js';
import { normalizeState, resolveTarget, displayName, distance, sourceUrl, cameraPose } from './state.js';
import './style.css';

const $ = id => document.getElementById(id);
const icons = { ScanEye, Radio, Maximize, Columns2, PictureInPicture2, Camera, Pause, Play, Navigation, Scan, VideoOff, Layers3, X, Plug };
const drawIcons = () => createIcons({ icons });
drawIcons();

let scene;
try { scene = new RobotScene($('scene'), $('labels')); }
catch { $('scene-empty').textContent = '3D view unavailable. WebGL is required.'; }
let state, target, selectedId = null, paused = false, source, events, generation = 0, lastStateAt = 0;
let frameController, frameTimer, currentImageUrl, lastImageAt = 0, lastSeq = null, listKey = '';
let mode = 'unknown', receivedValidState = false;
let framePath = '/frame.jpg';

function stored(key, fallback) { try { return localStorage.getItem(key) || fallback; } catch { return fallback; } }
function persist(key, value) { try { localStorage.setItem(key, value); } catch { /* Storage is optional. */ } }

function layout(name) {
  if (!['split', 'spatial', 'camera'].includes(name)) name = 'split';
  $('workspace').dataset.layout = name;
  document.querySelectorAll('[data-layout]').forEach(button => {
    if (button.tagName !== 'BUTTON') return;
    const active = button.dataset.layout === name;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  persist('robot-layout', name);
}
document.querySelectorAll('button[data-layout]').forEach(button => button.onclick = () => layout(button.dataset.layout));
layout(stored('robot-layout', 'split'));

function status(label, good = false) {
  $('connection-status').textContent = label;
  $('status-dot').className = `status-dot ${good ? 'good' : 'bad'}`;
}

function display() {
  if (!state || paused) return;
  target = resolveTarget(state, selectedId);
  const geometry = scene?.update(state, target);
  if (scene) {
    $('scene-empty').hidden = geometry.ready;
    $('scene-empty').querySelector('span').textContent = 'Waiting for camera pose';
  }
  $('geometry-type').textContent = geometry?.clouds ? (mode === 'synthetic' ? 'SIMULATED POINTS' : 'POINT CLOUD') : 'BOUNDING BOXES';
  $('object-count').textContent = state.objects.length + state.balls.length;
  $('fps').textContent = Number.isFinite(state.fps) ? `${state.fps.toFixed(0)} fps` : '-- fps';
  $('phase').textContent = String(state.phase || 'unknown').toUpperCase();
  $('target-label').textContent = target ? displayName(target) : 'No target';
  $('target-distance').replaceChildren(document.createTextNode(target && cameraPose(state.camera) ? distance(state.camera.position, target.position).toFixed(2) : '--'));
  const unit = document.createElement('small'); unit.textContent = ' m'; $('target-distance').append(unit);
  $('target-state').textContent = target ? (target.state === 'INSIDE' ? `Inside cup ${target.inside}` : String(target.state || 'Unknown').replaceAll('_', ' ')) : 'Awaiting target';
  const all = [...state.objects, ...state.balls];
  const key = JSON.stringify(all.map(o => [o.id, o.state, o.inside, o.color, o.label, o.id === target?.id]));
  if (key !== listKey) {
    listKey = key;
    $('object-list').replaceChildren(...all.map(object => {
      const button = document.createElement('button');
      button.className = 'object-item';
      button.setAttribute('aria-pressed', String(object.id === target?.id));
      const dot = document.createElement('span'); dot.className = 'object-dot';
      dot.style.background = { yellow: '#f4d06c', green: '#91dc83', red: '#ff8877', purple: '#d89fec' }[object.color] || (object.state === 'VISIBLE' ? '#73e6d3' : '#f4d06c');
      const label = document.createElement('span'); label.textContent = displayName(object);
      const sub = document.createElement('small');
      sub.textContent = object.state === 'INSIDE' ? `Inside cup ${object.inside}` : String(object.state || 'Unknown').replaceAll('_', ' ');
      button.append(dot, label, sub);
      button.onclick = () => { selectedId = String(object.id); display(); };
      return button;
    }));
  }
}

async function pollFrame(token) {
  if (token !== generation) return;
  if (paused || document.hidden) { frameTimer = setTimeout(() => pollFrame(token), 250); return; }
  frameController = new AbortController();
  const timer = setTimeout(() => frameController?.abort(), 5000);
  let next = 100;
  try {
    const url = new URL(`${source}${framePath}`);
    url.searchParams.set('_', Date.now());
    const response = await fetch(url, { cache: 'no-store', signal: frameController.signal });
    if (!response.ok || !response.headers.get('Content-Type')?.startsWith('image/')) throw new Error('Camera unavailable');
    const blob = await response.blob();
    if (token !== generation || paused) return;
    const imageUrl = URL.createObjectURL(blob);
    const image = new Image(); image.src = imageUrl;
    try { await image.decode(); } catch (error) { URL.revokeObjectURL(imageUrl); throw error; }
    if (token !== generation || paused) { URL.revokeObjectURL(imageUrl); return; }
    $('camera-frame').src = imageUrl;
    $('camera-frame').classList.add('ready');
    $('camera-empty').hidden = true;
    if (currentImageUrl) URL.revokeObjectURL(currentImageUrl);
    currentImageUrl = imageUrl;
    const seq = response.headers.get('X-Seq');
    if (seq === null || seq !== lastSeq) lastImageAt = performance.now();
    lastSeq = seq;
    $('frame-seq').textContent = `FRAME ${seq ?? '--'}`;
    const age = response.headers.get('X-Age-Ms');
    $('frame-age').textContent = age === null ? 'Age unavailable' : `${age} ms`;
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
  source = sourceUrl(url);
  events?.close(); clearTimeout(frameTimer); frameController?.abort();
  lastStateAt = 0; lastImageAt = 0; lastSeq = null; receivedValidState = false; selectedId = null;
  state = null; listKey = ''; mode = 'unknown';
  scene?.clear(); $('object-list').replaceChildren(); $('object-count').textContent = '0';
  $('target-label').textContent = 'No target'; $('target-distance').textContent = '-- m'; $('target-state').textContent = 'Awaiting state';
  $('camera-frame').classList.remove('ready'); $('camera-frame').removeAttribute('src');
  if (currentImageUrl) URL.revokeObjectURL(currentImageUrl); currentImageUrl = null;
  $('camera-empty').hidden = false; $('scene-empty').hidden = false;
  $('camera-empty').querySelector('span').textContent = 'Waiting for camera';
  $('source-name').textContent = new URL(source).host;
  $('mode-label').textContent = 'CONNECTING'; $('geometry-type').textContent = 'WAITING'; $('fps').textContent = '-- fps';
  $('stale-banner').hidden = true;
  persist('robot-source', source); persist('robot-frame-path', framePath);
  status('CONNECTING');
  events = new EventSource(`${source}/state`);
  events.onmessage = message => {
    if (token !== generation) return;
    try {
      state = normalizeState(JSON.parse(message.data));
      lastStateAt = performance.now(); receivedValidState = true;
      if (state.source === 'synthetic' || state.source === 'replay') mode = state.source;
      const label = mode === 'synthetic' ? 'SIMULATION' : mode === 'replay' ? 'REPLAY' : 'CONNECTED';
      status(paused ? 'FROZEN' : label, !paused);
      $('mode-label').textContent = label;
      $('camera-type').textContent = mode === 'synthetic' ? 'SIMULATED' : mode === 'replay' ? 'NO RECORDING' : 'X-RAY';
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
  if (paused) status('FROZEN'); else display();
};
$('fullscreen').onclick = async () => {
  try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
  catch { $('fullscreen').title = 'Fullscreen unavailable'; }
};
$('source-button').onclick = () => {
  $('stream-url').value = source; $('frame-path').value = framePath; $('connection-error').textContent = '';
  $('connection-dialog').showModal();
};
$('close-dialog').onclick = () => $('connection-dialog').close();
$('local-source').onclick = () => { $('stream-url').value = location.origin; $('frame-path').value = '/frame.jpg'; };
$('connection-form').onsubmit = event => {
  event.preventDefault();
  try {
    const value = sourceUrl($('stream-url').value);
    const path = $('frame-path').value.trim();
    if (!/^\/(?!\/)[^?#\s]*$/.test(path)) throw new Error('Use an endpoint path such as /frame.jpg.');
    if (location.protocol === 'https:' && value.startsWith('http:')) throw new Error('Open this viewer over HTTP to connect to an HTTP robot.');
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

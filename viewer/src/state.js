export function vector(value) {
  return Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
}

export function normalizeState(raw) {
  if (!raw || typeof raw !== 'object' || !Array.isArray(raw.objects)) throw new Error('Invalid state message');
  const objects = raw.objects.filter(o => o && (typeof o.id === 'string' || Number.isFinite(o.id)) && vector(o.position));
  const balls = raw.ball?.balls?.length ? raw.ball.balls : raw.ball ? [raw.ball] : [];
  return { ...raw, objects, balls: balls.filter(b => b && vector(b.position)),
    occluders: Array.isArray(raw.occluders) ? raw.occluders.filter(o => o && vector(o.position) && vector(o.size)) : [] };
}

export function cameraPose(camera) {
  if (!camera || !vector(camera.position)) return null;
  let forward = camera.forward;
  const matrix = camera.T_world_cam;
  if (!vector(forward) && Array.isArray(matrix) && matrix.length === 12) forward = [matrix[2], matrix[6], matrix[10]];
  if (!vector(forward) || Math.hypot(...forward) < 1e-6) return null;
  const length = Math.hypot(...forward);
  return { position: camera.position, forward: forward.map(v => v / length) };
}

export function resolveTarget(state, selectedId) {
  const all = [...state.objects, ...state.balls];
  return all.find(o => String(o.id) === selectedId) ?? state.objects.find(o => o.id === state.target_id) ?? null;
}

export function distance(a, b) { return Math.hypot(...a.map((v, i) => v - b[i])); }

export function displayName(object) {
  const label = String(object.label || 'Object');
  return object.color ? `${object.color[0].toUpperCase()}${object.color.slice(1)} ball` : `${label[0].toUpperCase()}${label.slice(1)} ${object.id}`;
}

// Point clouds are explicitly world-space or translation-only object-space.
// Never manufacture points from a bounding box and present them as LiDAR.
export function cloudPositions(cloud, origin = [0, 0, 0], limit = 50000) {
  if (!cloud || !Array.isArray(cloud.points) || !['world', 'object'].includes(cloud.frame)) return [];
  const result = [];
  const stride = Math.max(1, Math.ceil(cloud.points.length / limit));
  for (let i = 0; i < cloud.points.length; i += stride) {
    const point = cloud.points[i];
    if (!vector(point)) continue;
    result.push(...point.map((v, axis) => v + (cloud.frame === 'object' ? origin[axis] : 0)));
  }
  return result;
}

export function sourceUrl(value) {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('Use an HTTP or HTTPS robot URL without credentials.');
  url.search = ''; url.hash = '';
  return url.href.replace(/\/$/, '');
}

import * as THREE from 'three';

export const G8_LENS = { width: 480, height: 640, fx: 452, fy: 452, cx: 240.7, cy: 319.6 };

export function validLens(lens) {
  return lens && ['width', 'height', 'fx', 'fy', 'cx', 'cy'].every(k => Number.isFinite(lens[k]))
    && ['width', 'height', 'fx', 'fy'].every(k => lens[k] > 0);
}

export function calibrate(camera, lens, transform) {
  const { width, height, fx, fy, cx, cy } = lens;
  const near = camera.near;
  camera.projectionMatrix.makePerspective(-cx * near / fx, (width - cx) * near / fx,
    cy * near / fy, -(height - cy) * near / fy, near, camera.far);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  if (Array.isArray(transform) && [12, 16].includes(transform.length) && transform.every(Number.isFinite)) {
    // Row-major OpenCV camera-to-world: negate the local down and forward axes.
    const t = transform;
    const matrix = new THREE.Matrix4().set(t[0], -t[1], -t[2], t[3], t[4], -t[5], -t[6], t[7],
      t[8], -t[9], -t[10], t[11], 0, 0, 0, 1);
    matrix.decompose(camera.position, camera.quaternion, camera.scale);
    camera.updateMatrixWorld();
  }
}

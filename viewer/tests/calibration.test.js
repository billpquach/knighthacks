import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { calibrate, G8_LENS } from '../src/calibration.js';
import { normalizeScene, normalizeState, resolveTarget } from '../src/state.js';

test('OpenCV extrinsics and off-center intrinsics project to the same pixels', () => {
  const camera = new THREE.PerspectiveCamera(66, 1, .01, 100);
  const transform = [0, 0, 1, 3, 1, 0, 0, 4, 0, 1, 0, 5];
  calibrate(camera, G8_LENS, transform);
  // Camera point [0.1, 0.2, 2] transformed by this nontrivial pose.
  const ndc = new THREE.Vector3(5, 4.1, 5.2).project(camera);
  const u = (ndc.x + 1) / 2 * G8_LENS.width;
  const v = (1 - ndc.y) / 2 * G8_LENS.height;
  assert.ok(Math.abs(u - (452 * .1 / 2 + 240.7)) < 1e-9);
  assert.ok(Math.abs(v - (452 * .2 / 2 + 319.6)) < 1e-9);
});

test('reveal falls back to yellow ball, while explicit selection wins', () => {
  const state = normalizeState({ target_id: null, objects: [{ id: 42, position: [0, 0, .7] }],
    ball: { balls: [{ id: 'y', color: 'yellow', position: [0, 0, .8] }, { id: 'r', color: 'red', position: [0, 0, .8] }] } });
  assert.equal(resolveTarget(state, null).id, 'y');
  assert.equal(resolveTarget(state, '42').id, 42);
  state.target_id = 99;
  assert.equal(resolveTarget(state, null), null);
});

test('scene rejects incompatible frames, malformed geometry and excessive budgets', () => {
  const valid = { version: 1, frame: 'world', up: 'z', units: 'm', table: { z: .67, outline: [[0, 0], [1, 0], [1, 1]] }, cloud: { points: [[0, 0, .7]] }, objects: [] };
  assert.equal(normalizeScene(valid), valid);
  for (const change of [{ up: 'y' }, { cloud: { points: [[NaN, 0, 0]] } },
    { cloud: { points: Array(50001).fill([0, 0, 0]) } },
    { objects: [{ id: 1, position: [0, 0, 0], points: Array(2001).fill([0, 0, 0]) }] }]) {
    assert.throws(() => normalizeScene({ ...valid, ...change }));
  }
});

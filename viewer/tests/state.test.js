import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeState, cameraPose, resolveTarget, cloudPositions, sourceUrl } from '../src/state.js';

test('target IDs are identifiers, not array offsets, and legacy ball is supported', () => {
  const state = normalizeState({ target_id: 42, objects: [{ id: 42, position: [1, 2, 3] }], ball: { id: 'yellow ball', position: [1, 2, 3], balls: [] } });
  assert.equal(resolveTarget(state, null).id, 42);
  assert.equal(resolveTarget(state, 'yellow ball').id, 'yellow ball');
  assert.equal(resolveTarget(state, 'removed').id, 42);
});
test('invalid coordinates never reach rendering', () => {
  assert.throws(() => normalizeState(null));
  assert.equal(normalizeState({ objects: [{ id: 1, position: [NaN, 2, 3] }] }).objects.length, 0);
  assert.equal(cameraPose({ position: [1, 2, 3], forward: [0, 0, 0] }), null);
  assert.deepEqual(cameraPose({ position: [1, 2, 3], T_world_cam: [1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3] }).forward, [0, 0, 1]);
});
test('cloud coordinates require an explicit frame and respect point budget', () => {
  assert.deepEqual(cloudPositions({ frame: 'object', points: [[1, 2, 3], [Infinity, 0, 0]] }, [10, 0, 0]), [11, 2, 3]);
  assert.deepEqual(cloudPositions({ frame: 'world', points: [[1, 2, 3]] }, [10, 0, 0]), [1, 2, 3]);
  assert.deepEqual(cloudPositions({ points: [[1, 2, 3]] }), []);
  assert.equal(cloudPositions({ frame: 'world', points: Array(100).fill([1, 2, 3]) }, [0, 0, 0], 10).length, 30);
});
test('stream URLs reject unexpected protocols and embedded credentials', () => {
  assert.equal(sourceUrl('http://robot:8080/?x=1#hello'), 'http://robot:8080');
  assert.throws(() => sourceUrl('javascript:alert(1)'));
  assert.throws(() => sourceUrl('http://user:secret@robot'));
});

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { memoryAgeSeconds, memoryAgeLabel, displayName } from '../src/state.js';

test('memory age uses telemetry time and supports arbitrary tracked objects', () => {
  const object = { id: 42, label: 'backpack', state: 'OCCLUDED', since_seen_s: 18.7, last_seen_t: 1 };
  assert.equal(memoryAgeSeconds(object, 100), 18.7);
  assert.equal(memoryAgeLabel(object, 100), 'Last seen 18s ago');
  assert.equal(memoryAgeLabel({ ...object, state: 'VISIBLE' }, 100), 'Seen in frame');
  assert.equal(memoryAgeLabel({ state: 'OUT_OF_VIEW', last_seen_t: 20 }, 83), 'Last seen 1m 3s ago');
  assert.equal(displayName({ id: 4, color: 'red', label: 'backpack' }), 'Red backpack');
  assert.equal(displayName({ id: 'yellow', color: 'yellow' }), 'Yellow ball');
});

test('missing, malformed, or future observation timestamps remain unknown', () => {
  for (const object of [null, {}, { since_seen_s: -1 }, { since_seen_s: '4' }, { since_seen_s: Infinity }, { last_seen_t: 101 }, { last_seen_t: -1 }, { last_seen_t: null }]) {
    assert.equal(memoryAgeSeconds(object, 100), null);
    assert.equal(memoryAgeLabel(object, 100), 'Last seen unknown');
  }
  assert.equal(memoryAgeSeconds({ last_seen_t: 3 }, undefined), null);
  assert.equal(memoryAgeSeconds({ since_seen_s: null, last_seen_t: 0 }, 4), 4);
});

test('age formatting handles subsecond observations and long memories', () => {
  for (const [age, expected] of [[0, '<1s'], [.9, '<1s'], [1, '1s'], [59.9, '59s'], [60, '1m 0s'], [3599, '59m 59s'], [3600, '1h 0m'], [7320, '2h 2m']]) {
    assert.equal(memoryAgeLabel({ since_seen_s: age }), `Last seen ${expected} ago`);
  }
});

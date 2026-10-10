import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeMetrics, metricsUrl } from '../src/resources.js';

const sample = { version: 1, memory_kind: 'shared', device: 'Jetson', available: true,
  sample_id: '1:1', age_ms: 100, metrics: { ram_used_gb: 4, ram_total_gb: 8, gpu_pct: 0, cpu_pct: 20 } };
test('shared memory and real zeros are retained; missing fields are unavailable', () => {
  const value = normalizeMetrics(sample);
  assert.equal(value.metrics.gpu_pct, 0);
  assert.equal(value.metrics.ram_used_gb, 4);
  assert.equal(value.metrics.power_w, null);
  assert.equal(normalizeMetrics({ ...sample, available: false }), null);
});
test('bad units, stale values and invalid measurements never masquerade as live metrics', () => {
  assert.throws(() => normalizeMetrics({ ...sample, memory_kind: 'dedicated' }));
  assert.throws(() => normalizeMetrics({ ...sample, age_ms: 6000 }));
  assert.throws(() => normalizeMetrics({ ...sample, sample_id: null }));
  const invalid = normalizeMetrics({ ...sample, metrics: { ram_used_gb: 9, ram_total_gb: 8, gpu_pct: 150, power_w: -2 } });
  assert.equal(invalid.metrics.ram_used_gb, null);
  assert.equal(invalid.metrics.gpu_pct, null);
  assert.equal(invalid.metrics.power_w, null);
});
test('endpoint defaults to the robot, accepts a separate monitor, rejects credentials', () => {
  assert.equal(metricsUrl('', 'http://robot:8080'), 'http://robot:8080/metrics');
  assert.equal(metricsUrl('http://robot:8083/metrics', 'http://local'), 'http://robot:8083/metrics');
  assert.throws(() => metricsUrl('http://a:b@robot/metrics', 'http://local'));
});

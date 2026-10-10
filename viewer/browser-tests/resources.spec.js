import { test, expect } from '@playwright/test';

test('resource panel shows disconnected, live, missing and stale values honestly', async ({ page }, info) => {
  let mode = 'offline', seq = 0;
  await page.route('**/metrics', route => route.fulfill({ headers: { 'Access-Control-Allow-Origin': '*' }, json: {
    version: 1, memory_kind: 'shared', device: 'NVIDIA Jetson Orin Nano', available: mode !== 'offline',
    reason: 'not_jetson', sample_id: String(seq++), age_ms: mode === 'stale' ? 8000 : 100,
    metrics: { ram_used_gb: 4.2, ram_total_gb: 7.4, gpu_pct: 65, cpu_pct: 24,
      temp_c: 51.2, power_w: mode === 'partial' ? null : 9.8, swap_used_mb: 0 },
  } }));
  await page.goto('/');
  await expect(page.locator('.resource-status')).toHaveText('NOT CONNECTED');
  await expect(page.locator('[data-metric=memory] strong')).toHaveText('--');
  mode = 'live';
  await expect(page.locator('.resource-status')).toHaveText('LIVE');
  await expect(page.locator('[data-metric=memory] strong')).toHaveText('4.2 / 7.4 GiB');
  await expect(page.locator('[data-metric=gpu] progress')).toHaveAttribute('aria-valuetext', '65 percent');
  await expect(page.locator('[data-metric=swap] strong')).toHaveText('0 MiB');
  await expect.poll(() => page.locator('[data-metric=gpu] canvas').evaluate(c => {
    const pixels = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    return pixels.some((v, i) => i % 4 === 3 && v > 0);
  }), { timeout: 10000 }).toBe(true);
  await page.screenshot({ path: info.outputPath('resources-live.png'), fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  mode = 'partial';
  await expect(page.locator('.resource-status')).toHaveText('PARTIAL');
  await expect(page.locator('[data-metric=power] strong')).toHaveText('--');
  await page.getByRole('button', { name: 'Stream connection' }).click();
  await page.getByLabel('Jetson metrics URL').fill('http://jetson.test:8083/metrics');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await expect(page.locator('.resource-status')).toHaveText('PARTIAL');
  await expect(page.locator('.resource-heading')).toHaveAttribute('title', /jetson.test:8083/);
  mode = 'stale';
  await expect(page.locator('.resource-status')).toHaveText('STALE');
  await expect(page.locator('[data-metric=gpu] strong')).toHaveText('--');
});

test('an unchanged metrics sample expires even if the server reports age zero', async ({ page }) => {
  await page.route('**/metrics', route => route.fulfill({ json: {
    version: 1, memory_kind: 'shared', device: 'Jetson', available: true, sample_id: 'stuck', age_ms: 0,
    metrics: { ram_used_gb: 4, ram_total_gb: 8, gpu_pct: 25, cpu_pct: 10, temp_c: 40, power_w: 8, swap_used_mb: 0 },
  } }));
  await page.goto('/');
  await expect(page.locator('.resource-status')).toHaveText('LIVE');
  await expect(page.locator('.resource-status')).toHaveText('STALE', { timeout: 10000 });
  await expect(page.locator('[data-metric=memory] strong')).toHaveText('--');
});

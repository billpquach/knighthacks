import { test, expect } from '@playwright/test';

const cloud = Array.from({ length: 400 }, (_, i) => [(i % 20) * .02 - .2, Math.floor(i / 20) * .02 + .3, .67]);
const scene = { version: 1, frame: 'world', up: 'z', units: 'm',
  table: { z: .67, outline: [[-.3, .2], [.3, .2], [.3, .8], [-.3, .8]] },
  cloud: { points: cloud, colors: cloud.map(() => [120, 180, 140]) },
  objects: [{ id: 42, label: 'cup', position: [0, .5, .73], points: [[0, 0, 0], [.01, 0, .01]] }] };
const state = { source: 'replay', target_id: null,
  camera: { position: [0, 0, 1], forward: [0, 1, -.3] },
  objects: [{ id: 42, label: 'cup', position: [0, .5, .73], state: 'VISIBLE' }],
  ball: { id: 'yellow', color: 'yellow', position: [.1, .5, .7], state: 'VISIBLE' } };

test('static scene loads on connect and track, supports local files and both projections', async ({ page }, info) => {
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  let requests = 0;
  await page.route('**/scene', route => { requests++; return route.fulfill({ json: scene }); });
  await page.route('**/state', route => route.fulfill({ contentType: 'text/event-stream',
    body: ['scan', 'track'].map(phase => `data: ${JSON.stringify({ ...state, phase })}\n\n`).join('') }));
  await page.goto('/');
  await expect(page.locator('#geometry-type')).toHaveText('POINT CLOUD');
  await expect.poll(() => requests).toBeGreaterThanOrEqual(2);
  await expect(page.locator('#projection-label')).toHaveText('ROBOT PERSPECTIVE');
  await expect(page.locator('#robot-view')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#map-view')).toHaveAttribute('aria-pressed', 'false');
  await expect(page.locator('#target-label')).toHaveText('Yellow ball');
  for (const name of ['Calibrated robot perspective', 'Memory map']) {
    await page.getByRole('button', { name, exact: true }).click();
    await expect(page.getByRole('button', { name, exact: true })).toHaveAttribute('aria-pressed', 'true');
    await page.waitForTimeout(200);
    const pixels = await page.locator('#scene canvas').evaluate(canvas => {
      const gl = canvas.getContext('webgl2'), data = new Uint8Array(canvas.width * canvas.height * 4);
      gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
      let bright = 0;
      for (let i = 0; i < data.length; i += 4) if (Math.max(data[i], data[i + 1], data[i + 2]) > 90) bright++;
      return bright;
    });
    expect(pixels).toBeGreaterThan(100);
  }
  await page.screenshot({ path: info.outputPath('static-scene.png'), fullPage: true });
  await page.reload();
  await expect(page.locator('#projection-label')).toHaveText('ROBOT PERSPECTIVE');
  await expect(page.locator('#robot-view')).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Stream connection' }).click();
  await page.getByLabel('Scene preview file').setInputFiles({ name: 'scene.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(scene)) });
  await expect(page.locator('#scene-status')).toContainText('scene.json: 400 points');
  await page.getByLabel('Scene preview file').setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{"up":"y"}') });
  await expect(page.locator('#scene-status')).toContainText('Invalid scene');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

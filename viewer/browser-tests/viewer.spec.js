import { test, expect } from '@playwright/test';

test('scene, camera, and responsive layouts', async ({ page }, info) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#connection-status')).toHaveText('SIMULATION');
  await expect(page.locator('#camera-frame')).toHaveClass('ready');
  await expect(page.locator('#object-list button')).toHaveCount(6);
  await expect(page.locator('#geometry-type')).toHaveText('SIMULATED POINTS');
  const labelsOverlap = await page.locator('#labels').evaluate(container => {
    const labels = [...container.children].filter(el => getComputedStyle(el).visibility !== 'hidden').map(el => el.getBoundingClientRect());
    return labels.some((a, i) => labels.slice(i + 1).some(b => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top));
  });
  expect(labelsOverlap).toBe(false);
  const pixels = await page.locator('#scene canvas').evaluate(canvas => {
    const gl = canvas.getContext('webgl2');
    const data = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
    let bright = 0;
    for (let i = 0; i < data.length; i += 4) if (Math.max(data[i], data[i + 1], data[i + 2]) > 90) bright++;
    return bright;
  });
  expect(pixels).toBeGreaterThan(500);
  const first = await page.locator('#scene canvas').evaluate(c => c.toDataURL());
  await page.waitForTimeout(650);
  expect(await page.locator('#scene canvas').evaluate(c => c.toDataURL())).not.toBe(first);
  for (const [name, layout] of [['Spatial view with camera inset', 'spatial'], ['Camera view with spatial inset', 'camera'], ['Split view', 'split']]) {
    await page.getByRole('button', { name, exact: true }).click();
    await expect(page.locator('#workspace')).toHaveAttribute('data-layout', layout);
    const scene = await page.locator('#spatial-panel').boundingBox();
    const camera = await page.locator('#camera-panel').boundingBox();
    expect(scene.width).toBeGreaterThan(150); expect(scene.height).toBeGreaterThan(100);
    expect(camera.width).toBeGreaterThan(150); expect(camera.height).toBeGreaterThan(100);
    await page.screenshot({ path: info.outputPath(`${layout}.png`), fullPage: true });
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test('freeze, target selection, and connection settings', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#camera-frame')).toHaveClass('ready');
  await expect(page.locator('#object-list button')).toHaveCount(6);
  await page.getByRole('button', { name: /Yellow ball/ }).click();
  await expect(page.locator('#target-label')).toHaveText('Yellow ball');
  await page.getByRole('button', { name: 'Freeze display', exact: true }).click();
  const frame = await page.locator('#camera-frame').getAttribute('src');
  await page.waitForTimeout(600);
  expect(await page.locator('#camera-frame').getAttribute('src')).toBe(frame);
  await page.getByRole('button', { name: 'Resume display' }).click();
  await expect.poll(() => page.locator('#camera-frame').getAttribute('src')).not.toBe(frame);
  await page.getByRole('button', { name: 'Stream connection' }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.getByLabel('Front camera endpoint').fill('//bad-host');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await expect(page.locator('#connection-error')).toContainText('endpoint path');
  await page.getByRole('button', { name: 'Close connection settings' }).click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test('legacy state uses bounding boxes and missing source shows offline state', async ({ page }) => {
  await page.route('**/state', route => route.fulfill({ contentType: 'text/event-stream', body: `data: ${JSON.stringify({ target_id: 42, objects: [{ id: 42, label: 'cup', state: 'OUT_OF_VIEW', position: [0, -1, 0.7], size: [0.1, 0.1, 0.1] }], camera: { position: [0, 0, 1], forward: [0, 1, -0.3] } })}\n\n` }));
  await page.goto('/');
  await expect(page.locator('#geometry-type')).toHaveText('BOUNDING BOXES');
  await expect(page.locator('#target-label')).toHaveText('Cup 42');
  await page.getByRole('button', { name: 'Calibrated robot perspective' }).click();
  await expect(page.locator('.object-label.edge')).toBeVisible();
  await page.getByRole('button', { name: 'Stream connection' }).click();
  await page.getByLabel('Robot URL').fill('http://127.0.0.1:1');
  await page.unroute('**/state');
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await expect(page.locator('#connection-status')).toHaveText('OFFLINE', { timeout: 8000 });
  await expect(page.locator('#stale-banner')).toBeVisible();
  await expect(page.locator('#object-list button')).toHaveCount(0);
});

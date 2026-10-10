import { test, expect } from '@playwright/test';

test.afterEach(async ({ page }) => {
  await page.unrouteAll({ behavior: 'ignoreErrors' });
});

test('recorded camera and scene commit the same snapshot, including freeze and resume', async ({ page }) => {
  let index = 0;
  await page.route('**/state', route => {
    const i = index++;
    return route.fulfill({ contentType: 'text/event-stream', body: `retry: 100\ndata: ${JSON.stringify({
      source: 'replay', t: i, objects: [{ id: 42, position: [0, .5, .7], state: 'VISIBLE' }],
      camera: { position: [0, 0, 1], forward: [0, 1, -.3] },
      replay: { camera: true, index: i, frame: i, frame_t: i / 15 },
    })}\n\n` });
  });
  await page.route('**/frame.jpg?*', async route => {
    const i = new URL(route.request().url()).searchParams.get('index');
    const response = await route.fetch();
    await new Promise(resolve => setTimeout(resolve, 180));
    await route.fulfill({ response, headers: { ...response.headers(), 'x-state-index': i ?? '', 'x-seq': i ?? '' } });
  });
  await page.goto('/');
  await expect(page.locator('#camera-type')).toHaveText('REPLAY OVERLAY');
  await expect.poll(() => page.locator('#spatial-panel').getAttribute('data-state-index')).toMatch(/^\d+$/);
  await page.getByRole('button', { name: 'Freeze display', exact: true }).click();
  const frozen = await page.locator('#spatial-panel').getAttribute('data-state-index');
  await expect(page.locator('#camera-frame')).toHaveAttribute('data-state-index', frozen);
  await page.waitForTimeout(600);
  await expect(page.locator('#spatial-panel')).toHaveAttribute('data-state-index', frozen);
  await page.getByRole('button', { name: 'Resume display' }).click();
  await expect.poll(() => page.locator('#spatial-panel').getAttribute('data-state-index')).not.toBe(frozen);
  expect(await page.evaluate(() => document.querySelector('#spatial-panel').dataset.stateIndex === document.querySelector('#camera-frame').dataset.stateIndex)).toBe(true);
});

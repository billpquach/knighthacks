import { test, expect } from '@playwright/test';

test('memory ages follow displayed telemetry, preserve focus, and reset on source changes', async ({ page }) => {
  await page.addInitScript(() => {
    window.EventSource = class {
      constructor() { window.testStream = this; }
      close() {}
    };
  });
  await page.route('**/scene', route => route.fulfill({ status: 404, body: '' }));
  await page.goto('/');
  const send = async (age, state = 'OCCLUDED') => page.evaluate(({ age, state }) => {
    window.testStream.onmessage({ data: JSON.stringify({ source: 'replay', t: 100, target_id: 42,
      camera: { position: [0, 0, 1], forward: [0, 1, -.3] },
      objects: [{ id: 42, label: 'backpack', position: [0, .5, .7], state, since_seen_s: age },
        { id: 43, label: 'chair', position: [.2, .5, .7], state: 'OUT_OF_VIEW' }] }) });
  }, { age, state });
  await send(18);
  const object = page.locator('.object-item').first();
  await expect(object.locator('.memory-age')).toHaveText('Last seen 18s ago');
  await expect(page.locator('.object-item').nth(1).locator('.memory-age')).toHaveText('Last seen unknown');
  await expect(page.locator('#target-memory')).toHaveText('Last seen 18s ago');
  await object.focus();
  await send(19);
  await expect(object).toBeFocused();
  await expect(object.locator('.memory-age')).toHaveText('Last seen 19s ago');
  await page.getByRole('button', { name: 'Freeze display', exact: true }).click();
  await send(25);
  await expect(page.locator('#target-memory')).toHaveText('Last seen 19s ago');
  await page.getByRole('button', { name: 'Resume display', exact: true }).click();
  await expect(page.locator('#target-memory')).toHaveText('Last seen 25s ago');
  await expect(page.locator('#connection-status')).toHaveText('OFFLINE', { timeout: 6000 });
  await expect(page.locator('#target-memory')).toHaveText('Last seen 25s ago');
  await send(0, 'VISIBLE');
  await expect(page.locator('#target-memory')).toHaveText('Seen in frame');
  await send(3);
  await expect(page.locator('#target-memory')).toHaveText('Last seen 3s ago');
  await page.getByRole('button', { name: 'Stream connection', exact: true }).click();
  await page.getByRole('button', { name: 'Connect', exact: true }).click();
  await expect(page.locator('#target-memory')).toHaveText('Last seen unknown');
  await expect(page.locator('.object-item')).toHaveCount(0);
});

test('object inspector searches, filters, and inspects the frozen frame', async ({ page }) => {
  await page.addInitScript(() => {
    window.EventSource = class {
      constructor() { window.testStream = this; }
      close() {}
    };
  });
  await page.route('**/scene', route => route.fulfill({ status: 404, body: '' }));
  await page.goto('/');
  const send = async (later = false) => page.evaluate(later => {
    window.testStream.onmessage({ data: JSON.stringify({ source: 'replay', t: later ? 200 : 100, target_id: later ? 43 : 42,
      camera: { position: [0, 0, 1], forward: [0, 1, -.3] },
      objects: [{ id: 42, label: 'backpack', position: [0, .5, .7], state: later ? 'OCCLUDED' : 'VISIBLE', since_seen_s: later ? 20 : 0 },
        { id: 43, label: 'suitcase', position: [.2, .5, .7], state: 'OUT_OF_VIEW', since_seen_s: later ? 112 : 45 }] }) });
  }, later);
  await send();
  await expect(page.locator('#observed-count')).toHaveText('1');
  await expect(page.locator('#unseen-count')).toHaveText('1');
  await page.getByRole('searchbox', { name: 'Search objects' }).fill('BACK');
  await expect(page.locator('.object-item')).toHaveCount(1);
  await expect(page.locator('.object-name')).toHaveText('Backpack 42');
  await page.getByRole('searchbox', { name: 'Search objects' }).fill('missing');
  await expect(page.locator('#memory-empty')).toHaveText('No matching objects');
  await page.getByRole('button', { name: 'Clear object search' }).click();
  await expect(page.getByRole('searchbox', { name: 'Search objects' })).toBeFocused();
  await page.getByRole('button', { name: 'Unseen', exact: true }).click();
  await expect(page.locator('.object-item')).toHaveCount(1);
  await expect(page.locator('.object-name')).toHaveText('Suitcase 43');
  await page.getByRole('button', { name: 'Freeze display', exact: true }).click();
  await send(true);
  await page.locator('.object-item').click();
  await expect(page.locator('#target-label')).toHaveText('Suitcase 43');
  await expect(page.locator('#target-memory')).toHaveText('Last seen 45s ago');
  await expect(page.locator('#follow-target')).toHaveAttribute('aria-pressed', 'false');
  await page.getByRole('button', { name: 'All', exact: true }).click();
  await expect(page.locator('.object-item')).toHaveCount(2);
  await page.getByRole('button', { name: 'Follow automatic target' }).click();
  await expect(page.locator('#target-label')).toHaveText('Backpack 42');
  await page.getByRole('button', { name: 'Resume display', exact: true }).click();
  await expect(page.locator('#target-label')).toHaveText('Suitcase 43');
  await expect(page.locator('#target-memory')).toHaveText('Last seen 1m 52s ago');
  await expect(page.locator('#observed-count')).toHaveText('0');
  await expect(page.locator('#unseen-count')).toHaveText('2');
});

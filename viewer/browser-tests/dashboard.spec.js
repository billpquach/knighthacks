import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.route('**/state', route => route.fulfill({ contentType: 'text/event-stream', body: 'retry: 600000\n\n' }));
  await page.route('**/scene', route => route.fulfill({ status: 404, body: '' }));
  await page.goto('/');
});

test('dashboard keeps controls and resource metrics within the responsive layout', async ({ page }) => {
  const resourceBox = await page.locator('#resources').boundingBox();
  const workspaceBox = await page.locator('#workspace').boundingBox();
  if (page.viewportSize().width > 980) {
    expect(resourceBox.x + resourceBox.width).toBeLessThanOrEqual(workspaceBox.x + 1);
  } else {
    expect(resourceBox.y).toBeGreaterThanOrEqual(workspaceBox.y + workspaceBox.height);
  }
  for (const layout of ['spatial', 'camera', 'split']) {
    const button = page.locator(`button[data-layout=${layout}]`);
    await button.click();
    await expect(button).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#workspace')).toHaveAttribute('data-layout', layout);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await page.getByRole('button', { name: 'Stream connection', exact: true }).click();
  await expect(page.locator('dialog')).toBeVisible();
  expect(await page.locator('dialog').evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
  await page.getByRole('button', { name: 'Close connection settings' }).click();
  await expect(page.locator('dialog')).not.toBeVisible();
});

test('buttons provide brief press feedback and honor reduced motion', async ({ page }) => {
  const button = page.locator('#source-button');
  const box = await button.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await expect.poll(() => button.evaluate(el => getComputedStyle(el).transform)).not.toBe('none');
  await page.mouse.move(0, 0);
  await page.mouse.up();
  await expect.poll(() => button.evaluate(el => getComputedStyle(el).transform)).toBe('none');
  await page.emulateMedia({ reducedMotion: 'reduce' });
  expect(await button.evaluate(el => getComputedStyle(el).transitionDuration)).toBe('0s');
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  expect(await button.evaluate(el => getComputedStyle(el).transform)).toBe('none');
  await page.mouse.up();
  await expect(page.locator('dialog')).toBeVisible();
  expect(await page.locator('dialog').evaluate(el => getComputedStyle(el).animationName)).toBe('none');
});

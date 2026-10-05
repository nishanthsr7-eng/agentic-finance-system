// @ts-check
const { test, expect } = require('@playwright/test');

// js/flux-config.js points localhost at the API on :8000.
const API = 'http://localhost:8000';

test.beforeEach(async ({ page }) => {
  // Anything not mocked by a test fails fast instead of waiting on a real API.
  await page.route(`${API}/**`, (route) => route.fulfill({ status: 503, json: {} }));
});

test('landing page renders without script errors', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/');
  await expect(page).toHaveTitle(/FLUX|Agentic/);
  await expect(page.locator('a[href*="login"]').first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('demo account signs in and opens the dashboard', async ({ page }) => {
  await page.route(`${API}/auth/login`, async (route) => {
    const body = route.request().postDataJSON();
    expect(body.email).toBe('nishanth@flux.app');
    await route.fulfill({
      json: { token: 'test-token', user: { id: 1, name: 'Demo', email: body.email } },
    });
  });
  await page.goto('/pages/login.html');
  await page.getByRole('button', { name: 'Use demo account' }).click();
  await expect(page).toHaveURL(/dashboard\.html/);
  expect(await page.evaluate(() => localStorage.getItem('flux_token'))).toBe('test-token');
});

test('a rejected login shows the server message', async ({ page }) => {
  await page.route(`${API}/auth/login`, (route) =>
    route.fulfill({ status: 401, json: { detail: 'Invalid email or password' } }),
  );
  await page.goto('/pages/login.html');
  await page.fill('#login-email', 'someone@example.com');
  await page.fill('#login-password', 'wrong-password');
  await page.click('#login-btn');
  await expect(page.locator('#login-error')).toHaveText('Invalid email or password');
  await expect(page).toHaveURL(/login\.html/);
});

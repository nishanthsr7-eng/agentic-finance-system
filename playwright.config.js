// Frontend smoke tests. The API is mocked per test, so no backend is needed.
const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: { baseURL: 'http://localhost:3000' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'npx live-server --port=3000 --no-browser --quiet',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});

/**
 * FLUX — static site build for Cloudflare Pages.
 *
 * There is no bundler here; the frontend is plain HTML/CSS/JS. The only job is
 * to copy the frontend into dist/ and leave everything else behind.
 *
 * That separation is the point. Pages serves its output directory verbatim, so
 * pointing it at the repo root would publish backend/*.py, scripts/seed_mysql.py and
 * the training scripts as downloadable static files. Copying an explicit
 * allowlist means a new top-level file is never published by accident.
 *
 *   npm run build      → dist/
 *
 * Cloudflare Pages settings:
 *   Build command:            npm run build
 *   Build output directory:   dist
 */

import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// fileURLToPath, not URL.pathname: the latter keeps the percent-encoding and a
// leading slash, so a project path containing a space breaks on Windows.
const ROOT = fileURLToPath(new URL("..", import.meta.url));
const DIST = join(ROOT, "dist");

// Everything the browser actually requests. Anything not listed stays private.
const INCLUDE = [
  "index.html", "css", "js", "pages", "assets",
  // A root 404.html also switches off the Pages SPA fallback, so unknown
  // paths (and these two files) stop returning the landing page with a 200.
  "404.html", "robots.txt", "sitemap.xml",
];

// Cloudflare reads these from the output directory root.
const HEADERS = `# Security headers for the static frontend. The API sends its own.
/*
  X-Frame-Options: DENY
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: geolocation=(), microphone=(), camera=()

# The config carries the API origin and changes on redeploy, so it must not be
# served from a stale edge cache while the rest of the site updates.
/js/flux-config.js
  Cache-Control: no-cache
`;

await rm(DIST, { recursive: true, force: true });
await mkdir(DIST, { recursive: true });

for (const entry of INCLUDE) {
  const src = join(ROOT, entry);
  if (!existsSync(src)) {
    console.error(`missing: ${entry}`);
    process.exit(1);
  }
  await cp(src, join(DIST, entry), { recursive: true });
  console.log(`copied  ${entry}`);
}

await writeFile(join(DIST, "_headers"), HEADERS, "utf8");
console.log("wrote   _headers");
console.log(`\nstatic site ready in dist/`);

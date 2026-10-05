# Keeping the API warm

The live demo's API runs on Render's free plan, which spins a service down after
**15 minutes without traffic**. The next request then waits about **50 seconds**
for a cold start, which is what a recruiter or visitor would see as a blank
dashboard. This page explains what keeps the service awake, how to check it is
working, and what to do when it is not.

## Why staying up all month is allowed

Render's free plan gives **750 instance-hours per month per workspace**. A month
is at most 744 hours, so one service can run continuously and still fit.

> This only holds while `flux-api` is the **only** free service in the Render
> workspace. Two services together exceed 750 hours, and both get suspended for
> the rest of the month.

## What pings the service

All pingers hit `GET /health` (it also answers `HEAD`, because many monitors
default to it). `/health` is cheap: it reports cache, ChromaDB and scheduler
state and does no market-data calls, so pinging it costs no API quota.

| Pinger | Schedule | Status | Where it is configured |
|---|---|---|---|
| Cloudflare Worker `flux-warm` | every 10 min (cron `*/10 * * * *`) | **Primary.** Firing reliably since 2026-09-18 | [`keep-warm-worker/`](../keep-warm-worker/) in this repo |
| UptimeRobot HTTP monitor | every 5 min | **Second layer**, and alerts on real downtime | UptimeRobot dashboard (not in the repo) |
| GitHub Actions "Keep the API warm" | manual only | Fallback; its schedule is switched off | [`.github/workflows/keep-warm.yml`](../.github/workflows/keep-warm.yml) |

Two independent pingers mean one can fail without the demo going cold.

### Cloudflare Worker (primary)

[`keep-warm-worker/src/index.js`](../keep-warm-worker/src/index.js) has two
handlers:

- `scheduled` runs on the cron trigger. It logs `cron fired: ...` first, then
  fetches `${FLUX_API_URL}/health` with a 90-second timeout (long enough to
  ride out a cold start) and logs `ping 200 awake` or the failure.
- `fetch` runs when you open the Worker's URL in a browser. It pings the API
  immediately and returns `Service is awake.` (HTTP 200) or `Ping failed.`
  (HTTP 502), so you can test it without waiting for the next trigger.

Configuration lives in
[`keep-warm-worker/wrangler.toml`](../keep-warm-worker/wrangler.toml):

| Setting | Value | Why |
|---|---|---|
| `name` | `flux-warm` | Worker name in the Cloudflare dashboard |
| `crons` | `*/10 * * * *` | Inside the 15-minute idle timeout with margin |
| `FLUX_API_URL` | `https://flux-api-vono.onrender.com` | The Render service |
| `[observability] enabled` | `true` | Without logs a silent cron and a broken one look identical |

The Worker sits in its own folder on purpose: Cloudflare Pages looks for a
Wrangler config at the repository root, and finding one there makes it treat
the whole repo as a Worker instead of building the static site.

Cost: about 144 invocations a day against the free plan's 100,000.

**Deploy or redeploy** (from inside the folder, never the repo root):

```bash
cd keep-warm-worker
npx wrangler login
npx wrangler deploy
```

### UptimeRobot (second layer)

A free HTTP(s) monitor pointed at
`https://flux-api-vono.onrender.com/health` with a 5-minute interval. Besides
keeping the instance up, it emails when the API is genuinely down, which the
Worker does not.

### GitHub Actions (manual fallback)

The workflow used to run on a 10-minute cron. It was switched to
`workflow_dispatch` only because:

- GitHub queues scheduled jobs on shared runners and they drift; delays of over
  two hours happened on this repository.
- It doubled the Worker's pings for about 4,300 Actions runs a month.
- GitHub disables scheduled workflows in public repos after 60 days without a
  commit, so it would stop silently if the project sat untouched.

To run it by hand: **Actions -> Keep the API warm -> Run workflow**. It needs a
repository variable (not a secret, the URL is public) named `FLUX_API_URL`.

## Checking that it works

1. **The API itself:**
   ```bash
   curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" https://flux-api-vono.onrender.com/health
   ```
   A warm service answers `200` in well under a second. Several seconds or a
   timeout means it was asleep.
2. **The Worker's logs:** Cloudflare dashboard -> Workers & Pages ->
   `flux-warm` -> Logs. Expect a `cron fired` and a `ping 200 awake` line every
   10 minutes.
3. **The Worker on demand:** open its `*.workers.dev` URL (printed by
   `wrangler deploy`); it should say `Service is awake.`
4. **UptimeRobot:** the monitor should show 100% recent uptime.

## When the demo is cold anyway

| Symptom | Likely cause | Fix |
|---|---|---|
| No `cron fired` lines in Worker logs | Cron trigger not registered | Redeploy with `npx wrangler deploy`; check Triggers in the dashboard |
| `cron fired` but `ping failed` | API crashed or URL changed | Check Render logs; update `FLUX_API_URL` in `wrangler.toml` and redeploy |
| Render shows the service **suspended** | 750-hour quota used up | Make sure no second free service exists in the workspace; it resets next month |
| Service restarts often | Out of memory (512 MB limit) | See the memory notes in [DEPLOYMENT.md](DEPLOYMENT.md) |

The full history of how this setup was reached, including the period when the
Worker did not fire at all, is in
[DEPLOYMENT_RECORD.md](DEPLOYMENT_RECORD.md#6-keeping-the-instance-warm).

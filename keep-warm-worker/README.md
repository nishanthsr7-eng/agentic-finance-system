# flux-warm

A Cloudflare Worker that pings the FLUX backend every 10 minutes so Render's
free instance never spins down.

It replaced the scheduled run of `.github/workflows/keep-warm.yml`, which is now manual-only. GitHub's scheduler queues cron
on shared runners and routinely drifts - delays of over two hours have happened
on this repo - which is too unreliable for a 15-minute idle timeout. Workers
cron triggers fire from Cloudflare's edge and hold their schedule.

## Deploy

From inside this directory, not the repository root:

    cd keep-warm-worker
    npx wrangler login
    npx wrangler deploy

`wrangler login` opens a browser to authorise against your Cloudflare account.

## Verify

The worker also answers HTTP, so you can test it without waiting for the next
trigger: open the `*.workers.dev` URL that `deploy` prints. It should return
`Service is awake.`

Scheduled runs appear under Workers & Pages -> flux-warm -> Logs.

## Cost

Free plan: 100,000 requests/day. This uses 144.

## If the backend URL changes

Edit `FLUX_API_URL` in `wrangler.toml` and redeploy.

## Why this is not at the repository root

Cloudflare Pages looks for a Wrangler config at the root of the repo. One there
would make Pages treat the whole project as a Worker rather than building the
static site, which is exactly the wrong turn that cost a deploy earlier. Keep
this config in its own folder and always run wrangler from inside it.

The full keep-warm setup, including the UptimeRobot monitor and troubleshooting, is in [docs/KEEP_WARM.md](../docs/KEEP_WARM.md).

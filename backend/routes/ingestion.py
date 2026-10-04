"""Ingestion scheduler status and manual job triggers."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_admin
from ..db import (
    get_ingestion_log,
)
from . import common
from .common import _insight_job, _now_iso, log

router = APIRouter()


@router.get("/ingestion/status")
async def ingestion_status_endpoint():
    """Scheduler status and per-job last-run metadata."""
    from ..ingestion import get_status

    jobs = []
    if common.scheduler and common.scheduler.running:
        jobs = [
            {
                "id": j.id,
                "next_run": j.next_run_time.isoformat() if j.next_run_time else None,
            }
            for j in common.scheduler.get_jobs()
        ]
    log_rows = await get_ingestion_log(20)
    return {
        "scheduler_running": bool(common.scheduler and common.scheduler.running),
        "job_status": get_status(),
        "scheduled_jobs": jobs,
        "recent_log": log_rows,
        "timestamp": _now_iso(),
    }


@router.post("/ingestion/trigger/{job}", dependencies=[Depends(require_admin)])
async def ingestion_trigger(job: str):
    """
    Manually trigger a specific ingestion job immediately.
    job: crypto | stocks | ohlcv | news | market | insights | predictions

    predictions is the manual lever for the heavy FLUX-X cycle, which no longer
    runs itself on startup (HEAVY_JOBS_ON_STARTUP) — on a 512 MB host that run
    was the difference between one OOM kill and a restart loop.
    """
    from ..ingestion import (
        full_market_cycle,
        ingest_crypto,
        ingest_news,
        ingest_ohlcv,
        ingest_stocks,
    )

    async def _predictions():
        from ..prediction.flux_x import run_flux_x

        try:
            res = await run_flux_x()
            log.info("manual prediction cycle: %s", res)
        except Exception as exc:
            log.warning("manual prediction cycle failed: %s", exc)

    job_map = {
        "crypto": ingest_crypto,
        "stocks": ingest_stocks,
        "ohlcv": ingest_ohlcv,
        "news": ingest_news,
        "market": full_market_cycle,
        "insights": _insight_job,
        "predictions": _predictions,
    }
    if job not in job_map:
        raise HTTPException(400, f"Unknown job '{job}'. Valid: {list(job_map)}")
    asyncio.create_task(job_map[job]())
    return {"triggered": job, "timestamp": _now_iso()}

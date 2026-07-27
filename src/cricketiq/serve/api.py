"""
Phase 4.2 — FastAPI service for the win-probability replay.

Serves the precomputed timelines + match metadata via serve.data. It never trains
and (in this step) never runs the model — it streams precomputed rows, so nothing
expensive can fail during a live demo.

Endpoints:
  GET /health                    — liveness + how many matches are replayable
  GET /matches?league=&limit=    — matches for the picker (newest first)
  GET /match/{id}/timeline       — full win-prob curve + metadata (for the chart)
  GET /live/{id}?interval=1.0    — SSE: replay one ball every `interval` seconds

Run from the repo root:
    uvicorn cricketiq.serve.api:app --reload
Then open http://127.0.0.1:8000/docs to try it interactively.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from cricketiq.serve import data


@asynccontextmanager
async def lifespan(app: FastAPI):
    data.load()          # preload artifacts once — fail fast if missing, fast first request
    yield


app = FastAPI(title="CricketIQ", version="0.1.0", lifespan=lifespan)

# Dev CORS so a browser/Streamlit front end on another port can call this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten before any public deploy
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "matches": data.match_count()}


@app.get("/matches")
def matches(league: str | None = None, limit: int = Query(100, ge=1, le=2000)):
    return data.list_matches(league=league, limit=limit)


@app.get("/match/{match_id}/timeline")
def timeline(match_id: str):
    balls = data.get_timeline(match_id)
    if not balls:
        raise HTTPException(404, f"no replayable timeline for match {match_id}")
    return {"meta": data.get_meta(match_id), "balls": balls}


def _sse(event: str, payload: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.get("/live/{match_id}")
async def live(match_id: str, interval: float = Query(1.0, ge=0.05, le=10.0)):
    balls = data.get_timeline(match_id)      # fetched once, then streamed
    if not balls:
        raise HTTPException(404, f"no replayable timeline for match {match_id}")
    meta = data.get_meta(match_id)

    async def stream():
        yield _sse("meta", meta)             # match card first
        for ball in balls:
            yield _sse("ball", ball)         # then one delivery per tick
            await asyncio.sleep(interval)
        yield _sse("end", {"match_id": match_id, "balls": len(balls)})

    return StreamingResponse(stream(), media_type="text/event-stream")
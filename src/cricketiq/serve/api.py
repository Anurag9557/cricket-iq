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
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from cricketiq.serve import data, explain

STATIC_DIR = Path(__file__).parent / "static"


class MatchState(BaseModel):
    """The 11 features B1 needs to score an arbitrary chase state."""
    over: float
    innings_runs: float
    wickets_in_hand: float
    balls_remaining: float
    runs_needed: float
    target: float
    current_rr: float
    required_rr: float
    rr_diff: float
    runs_last30: float
    wkts_last30: float


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


@app.get("/")
def index():
    """Serve the single-page replay UI."""
    return FileResponse(STATIC_DIR / "index.html")


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


@app.get("/winprob/{match_id}/{ball_seq}")
def winprob_ball(match_id: str, ball_seq: int):
    """Explain a real delivery: win prob + ranked TreeSHAP drivers."""
    result = explain.explain_ball(match_id, ball_seq)
    if result is None:
        raise HTTPException(404, f"no state for match {match_id} ball {ball_seq}")
    return result


@app.post("/winprob")
def winprob_state(state: MatchState):
    """Score + explain an arbitrary chase state (the what-if endpoint)."""
    return explain.explain_features(state.model_dump())


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
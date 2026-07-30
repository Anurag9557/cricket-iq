# CricketIQ — Deployment (Phase 7.2): LIVE

*Companion to `claude/cricket-step-by-step.md` (the full execution log). This doc records the deploy milestone so any future session sees deploy status at a glance without re-reading the whole guide. Written 2026-07-30.*

## 🟢 LIVE — https://anurag9557.github.io/cricket-iq/

Verified working end-to-end (see "Verification" below). Repo: https://github.com/Anurag9557/cricket-iq.

## What shipped: a fully STATIC demo on GitHub Pages

**The decision (why not the FastAPI app on a host):** the plan's HF Spaces / Render route hit a wall — **Hugging Face's Docker SDK is a PAID feature** on this account (the free SDKs are Static / Gradio / Streamlit, none of which run a FastAPI + SSE app). Rather than pay, or chase a cold-start-prone free host, the live demo ships as a **fully static site**: the SSE/FastAPI replay is replaced by **precomputed JSON the browser fetches**, so the win-probability replay, verified commentary, and SHAP "why" panels run **entirely client-side — no server, no cold start, nothing that can fail live.** This is the project's "precompute, serve reads" principle taken to its conclusion. The FastAPI app stays in `src/` for local development; only *hosting* changed.

**The deploy kit (all under `deploy/`):**

- **`build_static.py`** — turns the serve layer's artifacts into static JSON: `deploy/site/data/matches.json` (dropdown) + `deploy/site/data/match/<id>.json` (`{meta, balls, commentary[], drivers{}}`). Reads local parquet + `b1.pkl` only (no LLM, no network), idempotent. Computes SHAP drivers for **every key moment** (not just the ~10 carded ones) so every clickable "why ▸" resolves. Run: `python deploy\build_static.py` (from repo root, `.venv` active).
- **`site/index.html`** — the static twin of the replay UI. Fetches the JSON, runs the replay with a client-side `setInterval` timer, and `showWhy` reads a **preloaded drivers map** instead of calling `/winprob`. Same chart, cards, badges, and SHAP panel as the served UI.
- **`DEPLOY_STATIC.md`** — the runbook: build → local smoke-test → assemble `docs/` → `git push` → enable Pages (Settings → Pages → main /docs) → verify → link in README. Includes troubleshooting + a `gh-pages`-branch alternative.
- (Superseded, kept for the FastAPI-host option: `deploy/space/` Dockerfile + `subset_for_deploy.py`, and `DEPLOY.md` with a banner pointing here.)

**Pre-flight test:** the static `index.html` was validated headless (Playwright vs stubbed JSON in the exact contract shape) — 15/16 checks pass (boot + auto-select, skip-to-end renders all key-moment cards, verified/​fallback badges, LLM-vs-generated bodies, FINAL/winner state, chart draw, click-to-open SHAP rows + facts, graceful "drivers unavailable"). The one non-pass was the browser's automatic `/favicon.ico` 404, unrelated.

## README

Rewritten from scratch earlier this phase; now carries the live link: a clickable **▶ Live Demo** badge in the hero, the hero screenshot wrapped in the demo link, and the `TODO: deployment URL` placeholder replaced with `https://anurag9557.github.io/cricket-iq/`. Updated `README.md` delivered to Anurag to commit.

## Verification (drove the live site in Anurag's own browser via Claude-in-Chrome)

- Booted onto the **RCB–KKR 1-run thriller** (match 1426274, IPL 2024) with the model report card showing real numbers (Brier 0.1144 · ECE 2.6% · AUC 0.978) → every data file loads over GitHub's CDN.
- **Skip to end** drew the full win-probability curve with six/wicket markers, the "Kolkata Knight Riders won by 1 runs" final line, the ball feed, and the key-moment cards — correct mix of ✓ Verified LLM lines (19.5, 19.4, 19.3, 19.2, 19.1) and a generated body for the uncarded moment (19.6).
- Clicking a **why ▸** opened the SHAP panel: "every number in the line traces to the data" facts line + signed driver bars (wickets in hand −1.27, runs needed +1.22, rate gap +0.96, runs scored +0.44, required run rate −0.43). All from static JSON.

## Gotchas banked (for future sessions)

- **The cloud sandbox can't reach `github.io`** — its network egress is allowlisted to package registries only. Consequence: the live Pages site can be driven **only from the user's connected browser** (Claude-in-Chrome), not headless from the sandbox. To render/screenshot the live site in the sandbox you'd first need to stage its data files locally.
- **In-session automated hero GIF was not achievable this session:** `gif_creator`'s frame capture was unreliable (returned 5 → 1 → 0 frames across attempts), and `javascript_tool` truncates its result at ~2 KB, so the 57 KB match JSON couldn't be pulled in one call to render locally. → Deferred to a quick self-record (Windows **ScreenToGif**) or a bridge-staged local render.

## Deploy-related TODO

- **Hero GIF** — record the RCB–KKR replay (Play → full curve → click a "why") → `assets/replay.gif`, swap the README hero `<img>` (the README already has a comment marking where). Easiest path: ScreenToGif on the live URL.
- **Ensure Phase 6.1 code is on `main`** — `serve/{data.py, api.py, static/index.html}` + `serve/build_commentary.py` (commentary.parquet stays gitignored).
- **Optional +1 — a live "API" badge:** if a hosted FastAPI instance is wanted alongside the static demo, **Google Cloud Run** is the best of the Docker-friendly free tiers, BUT it needs a billing account (card on file) and accepts cold starts (a recruiter can hit a spinner) — verify each platform's *current* free-tier terms first, since those shift. Not a blocker: the static demo already covers the recruiter-click case. (ChatGPT's comparison table rated Render ⭐⭐ for "insufficient RAM" — that's moot here: we'd deploy the 9-match subset, a few MB, not the full 782k-row data; the real free-tier catch across all of them is cold-starts.)

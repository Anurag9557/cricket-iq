"""
Deterministic cricket stat tools (Phase 5.1).

Typed polars functions over deliveries.parquet + matches.parquet. Every tool returns
the value(s) AND a sample size `n` — an economy off 12 balls is noise, and reporting
n is what makes this an analyst tool rather than a toy. The LLM agent NEVER computes
these numbers itself: it calls these, and the verifier RE-RUNS them to check the
agent's claims. Correctness here is the foundation of the whole verified-agent idea.

Conventions (documented because a few are approximations of the strict rule):
- balls faced/bowled = legal deliveries (is_legal). This excludes no-balls as well as
  wides, so it very slightly undercounts a batter's strict "balls faced".
- bowler runs conceded = sum(runs_total). parse.py stored total extras without
  splitting byes/leg-byes, so economy is a hair generous on the rare bye. Documented.
- a wicket is credited to the bowler only for: bowled, caught, caught and bowled, lbw,
  stumped, hit wicket (run-outs etc. are not the bowler's).
- `batter`/`bowler`/`player_out` are Cricsheet registry IDs — use resolve()/pid() to
  turn a player name into an ID.
"""
from __future__ import annotations

import polars as pl

from cricketiq.core import config

BOWLER_WICKET_KINDS = ["bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"]

_deliveries: pl.DataFrame | None = None
_matches: pl.DataFrame | None = None
_register: pl.DataFrame | None = None
_reg_index: list[dict] | None = None
_activity: dict | None = None


def _deliv() -> pl.DataFrame:
    global _deliveries
    if _deliveries is None:
        _deliveries = pl.read_parquet(config.PROCESSED_DIR / "deliveries.parquet")
    return _deliveries


def _matchtbl() -> pl.DataFrame:
    global _matches
    if _matches is None:
        _matches = pl.read_parquet(config.PROCESSED_DIR / "matches.parquet")
    return _matches


# ---------- name resolution ----------

def _register_tbl() -> pl.DataFrame:
    global _register
    if _register is None:
        path = config.RAW_DIR / "people.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Cricsheet register not found at {path}. Point _register_tbl() at your "
                "people.csv (needs columns: identifier, name, unique_name)."
            )
        _register = pl.read_csv(path)
    return _register


def _register_index() -> list[dict]:
    """Precompute (surname, first, initial) per register entry — once — so name
    matching is a fast, format-aware Python scan rather than a naive substring."""
    global _reg_index
    if _reg_index is None:
        _reg_index = []
        for ident, nm in _register_tbl().select(["identifier", "name"]).iter_rows():
            if not nm:
                continue
            toks = str(nm).lower().split()
            if not toks:
                continue
            _reg_index.append({
                "id": ident, "name": nm,
                "surname": toks[-1],
                "first": toks[0] if len(toks) > 1 else "",
                "initial": toks[0][0] if len(toks) > 1 and toks[0] else "",
            })
    return _reg_index


def _activity_map() -> dict:
    """id -> number of deliveries the player appears in (batting + bowling), built once.
    The disambiguator: among namesakes, the famous player has by far the most data."""
    global _activity
    if _activity is None:
        d = _deliv()
        _activity = {}
        for col in ("batter", "bowler"):
            for ident, n in d.group_by(col).len().iter_rows():
                _activity[ident] = _activity.get(ident, 0) + int(n)
    return _activity


def resolve(name: str, limit: int = 5) -> list[dict]:
    """Ranked {id, name} candidates for a player, robust to Cricsheet's formats.
    Matches surname + first-name/initial — handles 'Virat Kohli'->'V Kohli',
    'Rohit Sharma'->'RG Sharma', 'MS Dhoni', and 'Kohli' — then breaks ties by how much
    data each namesake has, so the famous player wins. Two DIFFERENT spelled-out first
    names (e.g. 'Vivek' vs 'Virat') are treated as different people, not a match."""
    toks = name.strip().lower().split()
    if not toks:
        return []
    surname, qfirst = toks[-1], (toks[0] if len(toks) > 1 else "")
    qinit = qfirst[:1]
    act = _activity_map()

    scored = []
    for e in _register_index():
        if e["surname"] != surname:
            continue
        ef = e["first"]
        if not qfirst:
            score = 1                             # surname-only query
        elif ef == qfirst:
            score = 3                             # exact first name / same initials ('MS')
        elif len(ef) <= 2 and ef[:1] == qinit:
            score = 2                             # register is initials ('V','RG') matching query
        elif len(qfirst) <= 2 and qfirst[:1] == ef[:1]:
            score = 2                             # query is initials matching register's first
        else:
            continue                              # different spelled-out first names -> different player
        scored.append((score, act.get(e["id"], 0), e))

    # among valid name matches, prefer the player with the richest data
    scored.sort(key=lambda s: (-s[1], -s[0]))   # activity, then string-match score
    seen, out = set(), []
    for _, _, e in scored:
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        out.append({"id": e["id"], "name": e["name"]})
        if len(out) >= limit:
            break
    return out


def pid(name: str) -> str:
    """The single best-matching registry id for a name (convenience for tests/demos)."""
    hits = resolve(name, limit=1)
    if not hits:
        raise ValueError(f"no player matches {name!r}")
    return hits[0]["id"]


# ---------- filters + helpers ----------

def _apply(d: pl.DataFrame, phase: str | None, season: int | None) -> pl.DataFrame:
    if phase:
        if phase not in config.PHASES:
            raise ValueError(f"phase must be one of {list(config.PHASES)}")
        d = d.filter(pl.col("over").is_in(list(config.PHASES[phase])))
    if season:
        ids = _matchtbl().filter(pl.col("season") == season)["match_id"]
        d = d.filter(pl.col("match_id").is_in(ids))
    return d


def _rate(num: float, den: float):
    return round(num / den, 2) if den else None


# ---------- the tools (each returns value(s) + n) ----------

def batter_stats(batter_id: str, phase: str | None = None, season: int | None = None) -> dict:
    d = _apply(_deliv().filter(pl.col("batter") == batter_id), phase, season)
    runs = int(d["runs_batter"].sum() or 0)
    balls = int(d.filter(pl.col("is_legal")).height)
    outs = int(d.filter(pl.col("is_wicket") & (pl.col("player_out") == batter_id)).height)
    return {
        "runs": runs, "balls": balls, "dismissals": outs,
        "average": _rate(runs, outs), "strike_rate": _rate(runs * 100, balls),
        "n": balls,
    }


def bowler_stats(bowler_id: str, phase: str | None = None, season: int | None = None) -> dict:
    d = _apply(_deliv().filter(pl.col("bowler") == bowler_id), phase, season)
    balls = int(d.filter(pl.col("is_legal")).height)
    runs = int(d["runs_total"].sum() or 0)
    wkts = int(d.filter(pl.col("is_wicket") & pl.col("wicket_kind").is_in(BOWLER_WICKET_KINDS)).height)
    dots = int(d.filter(pl.col("is_legal") & (pl.col("runs_total") == 0)).height)
    return {
        "balls": balls, "runs": runs, "wickets": wkts,
        "economy": _rate(runs * 6, balls), "dot_pct": _rate(dots * 100, balls),
        "n": balls,
    }


def matchup(batter_id: str, bowler_id: str, phase: str | None = None) -> dict:
    d = _apply(_deliv().filter((pl.col("batter") == batter_id) & (pl.col("bowler") == bowler_id)), phase, None)
    balls = int(d.filter(pl.col("is_legal")).height)
    runs = int(d["runs_batter"].sum() or 0)
    outs = int(d.filter(
        pl.col("is_wicket") & (pl.col("player_out") == batter_id)
        & pl.col("wicket_kind").is_in(BOWLER_WICKET_KINDS)
    ).height)
    return {"balls": balls, "runs": runs, "dismissals": outs,
            "strike_rate": _rate(runs * 100, balls), "n": balls}


def venue_par(venue: str, season: int | None = None) -> dict:
    m = _matchtbl().filter(pl.col("venue") == venue)
    if season:
        m = m.filter(pl.col("season") == season)
    par = m["first_innings_runs"].mean()
    return {"par": round(float(par), 1) if par is not None else None, "n": int(m.height)}
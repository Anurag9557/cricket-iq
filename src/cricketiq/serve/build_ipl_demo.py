"""
Pick the most exciting IPL matches in YOUR dataset and precompute verified
commentary for them (plus any you want to keep) in one pass.

Two-step by design — preview first, spend LLM calls only when you're happy:
    python build_ipl_demo.py            # DRY RUN: print the picks, no API calls
    python build_ipl_demo.py --build    # actually generate commentary
    python build_ipl_demo.py 6 --build  # top 6 instead of 8

Excitement = number of key moments (|win-prob swing| >= 0.08) plus how dramatic
the last four overs were (biggest late swing), so blowouts sink and nail-biters float.

NOTE: build() OVERWRITES commentary.parquet with exactly the matches it is given.
ALSO_BUILD keeps matches you don't want dropped (default: the game you already
verified). After building, RESTART uvicorn — data.py caches commentary once at
load, so a running server won't see the new file until it restarts.
"""
import sys

import polars as pl

from cricketiq.core import config
from cricketiq.serve.build_commentary import build, TOP_K

args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 8
BUILD = "--build" in sys.argv
ALSO_BUILD = ["1512844"]          # always keep these; set to [] to drop the Derbyshire game

tl = pl.read_parquet(config.PROCESSED_DIR / "timelines.parquet")
mt = pl.read_parquet(config.PROCESSED_DIR / "matches.parquet")

# --- self-diagnosis: surface the real column names / league labels up front ------
print("timelines columns:", tl.columns)
print("matches columns   :", mt.columns)
print("leagues in data   :", sorted(mt["league"].unique().to_list()))

# --- excitement per match --------------------------------------------------------
ev = (tl.with_columns(pl.col("wp_delta").abs().alias("d"))
        .group_by("match_id")
        .agg([
            (pl.col("d") >= 0.08).sum().alias("n_key"),
            pl.col("d").filter(pl.col("balls_remaining") <= 24).max().alias("late_swing"),
        ]))

ranked = (ev.join(mt, on="match_id", how="left")
            .filter(pl.col("league").str.to_lowercase().str.contains("ipl"))
            .with_columns(
                (pl.col("n_key").cast(pl.Float64)
                 + 20.0 * pl.col("late_swing").fill_null(0.0)).alias("score"))
            .sort("score", descending=True)
            .head(N))

if ranked.height == 0:
    print("\nNo IPL matches matched. Look at 'leagues in data' above and adjust the "
          "  .str.contains('ipl')  filter to whatever your IPL label actually is.")
    sys.exit(1)

print(f"\nTop {ranked.height} exciting IPL matches:")
for r in ranked.iter_rows(named=True):
    late = r.get("late_swing") or 0.0
    print(f"  {r['match_id']}  {r.get('team_bat_first')} v {r.get('team_chase')}"
          f"  {r.get('date')}  {r.get('city')}"
          f"  | key_moments={r['n_key']}  late_swing={late:.2f}"
          f"  | won by {r.get('win_margin')} {r.get('win_by')}")

ids = list(dict.fromkeys([str(m) for m in ALSO_BUILD]
                         + [str(r["match_id"]) for r in ranked.iter_rows(named=True)]))

if not BUILD:
    print(f"\n(dry run) would build {len(ids)} matches: {ids}")
    print("re-run with  --build  to generate commentary (this makes LLM calls).")
    sys.exit(0)

print(f"\nBuilding {len(ids)} matches x up to {TOP_K} cards each "
      f"(~{len(ids) * TOP_K} LLM calls). Overwrites commentary.parquet.\n")
build(ids)
print("\nDone. RESTART uvicorn so the server reloads commentary.parquet.")
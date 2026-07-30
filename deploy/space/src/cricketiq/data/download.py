"""Download Cricsheet match data (JSON zips) + player register into data/raw/.

Run:  python -m cricketiq.data.download

Idempotent: competitions already extracted are skipped, so re-running is cheap.
Source: https://cricsheet.org/downloads/  (free, updated ~daily)
URLs verified 2026-07-20.
"""
from __future__ import annotations

import io
import zipfile

import requests

from cricketiq.core.config import RAW_DIR

# Key = local folder name under data/raw/.  Value = Cricsheet JSON bundle URL.
LEAGUES: dict[str, str] = {
    "ipl":   "https://cricsheet.org/downloads/ipl_json.zip",
    "t20i":  "https://cricsheet.org/downloads/t20s_male_json.zip",
    "bbl":   "https://cricsheet.org/downloads/bbl_male_json.zip",
    "blast": "https://cricsheet.org/downloads/ntb_male_json.zip",  # NatWest/Vitality T20 Blast
    "psl":   "https://cricsheet.org/downloads/psl_male_json.zip",
    "cpl":   "https://cricsheet.org/downloads/cpl_male_json.zip",
}

REGISTER_URL = "https://cricsheet.org/register/people.csv"
HEADERS = {"User-Agent": "cricket-iq/0.1 (student project)"}


def download_zip(name: str, url: str) -> int:
    """Download one competition's zip and extract its .json files into data/raw/<name>/.

    Returns the number of match .json files present after extraction.
    """
    dest = RAW_DIR / name
    dest.mkdir(parents=True, exist_ok=True)

    existing = list(dest.glob("*.json"))
    if existing:
        print(f"  {name:6s} already has {len(existing)} json files - skipping")
        return len(existing)

    print(f"  {name:6s} downloading {url} ...")
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        members = [m for m in zf.namelist() if m.endswith(".json")]
        zf.extractall(dest, members=members)

    size_mb = len(resp.content) // 1_000_000
    print(f"  {name:6s} extracted {len(members)} json files ({size_mb} MB)")
    return len(members)


def download_register() -> None:
    """Download the player-identifier register (people.csv) into data/raw/."""
    dest = RAW_DIR / "people.csv"
    if dest.exists():
        print("  register already present - skipping")
        return
    print("  downloading player register ...")
    resp = requests.get(REGISTER_URL, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  register saved ({len(resp.content) // 1000} KB)")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Cricsheet data into {RAW_DIR}")

    counts: dict[str, int] = {}
    for name, url in LEAGUES.items():
        counts[name] = download_zip(name, url)

    download_register()

    print("\nSummary - matches per competition:")
    for name, n in counts.items():
        print(f"  {name:6s} {n:>5d}")
    print(f"  {'TOTAL':6s} {sum(counts.values()):>5d}")


if __name__ == "__main__":
    main()
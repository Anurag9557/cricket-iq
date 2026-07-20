"""Project-wide constants and paths. Import from here — never hardcode."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # core -> cricketiq -> src -> repo root
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Locked decisions (see docs/cricket-blueprint.md — do not reopen)
ERA_START_YEAR = 2008      # modern T20 era only
TRAIN_END_YEAR = 2023      # temporal split: train <= 2023
VAL_YEAR = 2024            # calibration / validation year
TEST_START_YEAR = 2025     # 2025-26 held out until final eval

PHASES = {"powerplay": range(1, 7), "middle": range(7, 16), "death": range(16, 21)}
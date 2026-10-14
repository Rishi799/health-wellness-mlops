"""Live demo: validation catching bad data.

    python scripts/validation_demo.py                 # uses data/interim/merged_2017-2018.csv
    python scripts/validation_demo.py --cycle 2021-2023

1. validates the real merged file (should pass)
2. copies it, inserts ONE 9999 and ONE duplicate seqn
3. validates the broken copy (fails, names the column and the bad values)

If the merged file does not exist yet it falls back to a synthetic one so the
demo can still be rehearsed.  The real files in data/interim are never touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.data import validate  # noqa: E402

DEMO_DATA_DIR = REPO_ROOT / "data" / "interim" / "demo"
DEMO_REPORT_DIR = REPO_ROOT / "reports" / "demo"


def load_merged(cycle: str) -> tuple[pd.DataFrame, str]:
    path = REPO_ROOT / "data" / "interim" / f"merged_{cycle}.csv"
    if path.exists():
        return pd.read_csv(path), str(path.relative_to(REPO_ROOT))
    from tests.synthetic import make_merged

    lifestyle = REPO_ROOT / "data" / "interim" / f"lifestyle_{cycle}.csv"
    if not lifestyle.exists():
        raise SystemExit(f"Neither {path.name} nor {lifestyle.name} exists yet")
    return make_merged(pd.read_csv(lifestyle)), f"SYNTHETIC merge built on {lifestyle.name}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", default="2017-2018")
    args = parser.parse_args()
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    merged, source = load_merged(args.cycle)
    good_path = DEMO_DATA_DIR / f"merged_good_{args.cycle}.csv"
    bad_path = DEMO_DATA_DIR / f"merged_broken_{args.cycle}.csv"
    merged.to_csv(good_path, index=False)

    broken = merged.copy()
    broken.loc[broken.index[2], "sedentary_min_day"] = 9999  # a refused/unknown code that slipped through
    broken = pd.concat([broken, broken.iloc[[0]]], ignore_index=True)  # duplicate seqn
    broken.to_csv(bad_path, index=False)

    common = ["--cycle", args.cycle, "--schema", "merged"]
    print(f"\n=== 1. GOOD DATA ({source}) ===")
    validate.main([*common, "--path", str(good_path), "--report", str(DEMO_REPORT_DIR / "report_good.json")])

    print("\n=== 2. SAME DATA with one 9999 and one duplicate seqn inserted ===")
    code = validate.main([*common, "--path", str(bad_path), "--report", str(DEMO_REPORT_DIR / "report_broken.json")])
    print(f"\nexit code = {code}  (non-zero = dvc repro would stop here)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

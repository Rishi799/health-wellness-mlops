"""Download raw NHANES files (CDC, public domain) into data/raw/nhanes/<cycle>/.

Usage:  python download_nhanes.py            # both cycles
        python download_nhanes.py 2017-2018  # one cycle

2017-2018 = training data (dataset v1).
2021-2023 = later population, used in Phase 3 as real drift.
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
OUT_ROOT = Path(__file__).resolve().parent / "data" / "raw" / "nhanes"

CYCLES = {
    "2017-2018": ("2017", "J", "BPX"),   # blood pressure by manual cuff (BPX)
    "2021-2023": ("2021", "L", "BPXO"),  # blood pressure by automatic device (BPXO)
}

# file -> what it contains
FILES = {
    "DEMO": "demographics: age, sex, income, education",
    "BMX": "body measures: weight, height, BMI, waist",
    "BP": "blood pressure readings + resting pulse",
    "BPQ": "questionnaire: told high BP / high cholesterol",
    "TCHOL": "lab: total cholesterol",
    "HDL": "lab: HDL cholesterol",
    "GHB": "lab: glycohemoglobin (HbA1c)",
    "SMQ": "smoking",
    "ALQ": "alcohol use",
    "PAQ": "physical activity",
    "SLQ": "sleep",
    "DPQ": "PHQ-9 depression screener (stress / mental wellbeing)",
    "DBQ": "diet behaviour: self-rated diet quality",
    "MCQ": "medical conditions: heart disease, heart attack, stroke",
    "DIQ": "diabetes",
}


def download(cycle: str) -> None:
    year, suffix, bp_name = CYCLES[cycle]
    out = OUT_ROOT / cycle
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        fname = f"{bp_name if name == 'BP' else name}_{suffix}.xpt"
        target = out / fname
        if target.exists():
            print(f"skip  {cycle}/{fname} (already there)")
            continue
        url = f"{BASE}/{year}/DataFiles/{fname}"
        print(f"get   {url}")
        urllib.request.urlretrieve(url, target)
    print(f"done  {cycle}: {len(FILES)} files in {out}")


if __name__ == "__main__":
    for c in sys.argv[1:] or CYCLES:
        download(c)

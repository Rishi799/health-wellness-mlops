"""Lifestyle topic table (Person 3).

Turns the raw NHANES questionnaires SMQ, ALQ, PAQ, SLQ and DBQ into the seven
contract columns:

    seqn, smoking_status, alcohol_drinks_week, exercise_min_week,
    sedentary_min_day, sleep_hours, diet_quality

Run from the repo root:

    python -m src.data.lifestyle --cycle 2017-2018
    python -m src.data.lifestyle --cycle 2021-2023
    python -m src.data.lifestyle --cycle all

Reads only from data/raw/nhanes/<cycle>/ and writes
data/interim/lifestyle_<cycle>.csv.  Nothing is imputed here (contract rule 7).

Every cleaning rule below has a row in the decision log
(docs/p3_decision_log.md and the notebook).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "nhanes"
INTERIM_DIR = REPO_ROOT / "data" / "interim"

CYCLE_SUFFIX = {"2017-2018": "J", "2021-2023": "L"}
STEMS = ("SMQ", "ALQ", "PAQ", "SLQ", "DBQ")

CONTRACT_COLUMNS = [
    "seqn",
    "smoking_status",
    "alcohol_drinks_week",
    "exercise_min_week",
    "sedentary_min_day",
    "sleep_hours",
    "diet_quality",
]

# --------------------------------------------------------------------------
# Decisions (see the decision log). Change them here, nowhere else.
# --------------------------------------------------------------------------

# ALQ121: how often drank alcohol in the past 12 months -> days per year.
# Midpoint of each range in the codebook wording.  CONFIRM THE WORDING against
# https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/ALQ_J.htm before you present.
ALQ121_TABLE = {
    # code: (codebook wording, days per year used)
    0: ("Never in the last year", 0.0),
    1: ("Every day", 365.0),
    2: ("Nearly every day (5-6 days a week)", 286.0),  # 5.5 * 52
    3: ("3 to 4 times a week", 182.0),  # 3.5 * 52
    4: ("2 times a week", 104.0),  # 2 * 52
    5: ("Once a week", 52.0),
    6: ("2 to 3 times a month", 30.0),  # 2.5 * 12
    7: ("Once a month", 12.0),
    8: ("7 to 11 times in the last year", 9.0),
    9: ("3 to 6 times in the last year", 4.5),
    10: ("1 to 2 times in the last year", 1.5),
}
ALQ121_DAYS_PER_YEAR = {code: days for code, (_, days) in ALQ121_TABLE.items()}
WEEKS_PER_YEAR = 52

# 2021-2023 activity: unit letter -> multiplier that turns "times per unit"
# into "times per week".
UNIT_TO_PER_WEEK = {
    "D": 7.0,  # times per day   -> x 7
    "W": 1.0,  # times per week  -> x 1
    "M": 12.0 / 52.0,  # times per month -> x 12 / 52
    "Y": 1.0 / 52.0,  # times per year  -> x 1 / 52
}

VIGOROUS_WEIGHT = 2.0  # WHO: 1 vigorous minute ~ 2 moderate minutes
CAP_MIN_WEEK = 3000.0  # moderate-equivalent minutes/week; larger totals are capped
SLEEP_WEIGHTS = (5.0, 2.0)  # (workday, weekend) -> (5*SLD012 + 2*SLD013) / 7

FAKE_TINY = 1e-6  # |x| below this is a float-noise zero (5.4e-79)


def _say(msg: str) -> None:
    print(f"[lifestyle] {msg}")


# --------------------------------------------------------------------------
# Loading and generic cleaning
# --------------------------------------------------------------------------


def _find_raw_file(raw_dir: Path, cycle: str, stem: str) -> Path:
    suffix = CYCLE_SUFFIX[cycle]
    folder = raw_dir / cycle
    for ext in (".csv", ".CSV", ".xpt", ".XPT"):
        path = folder / f"{stem}_{suffix}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {stem}_{suffix}.csv or .xpt in {folder}")


def load_raw(raw_dir: Path, cycle: str, stem: str) -> pd.DataFrame:
    """Read one raw file, upper-case the columns, fix float-noise zeros."""
    path = _find_raw_file(raw_dir, cycle, stem)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_sas(path, format="xport")
    df.columns = [str(c).upper() for c in df.columns]
    df["SEQN"] = df["SEQN"].astype(int)
    if df["SEQN"].duplicated().any():
        raise ValueError(f"{path.name}: SEQN is not unique in the raw file")

    # 5.4e-79 is a stored zero (decision log #1).
    for col in df.columns:
        if col != "SEQN" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].mask(df[col].abs() < FAKE_TINY, 0.0)
    return df.set_index("SEQN")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """A numeric column, or all-NaN when the cycle does not have it."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _codes_to_nan(s: pd.Series, codes) -> pd.Series:
    """Refused / don't know codes are not real values."""
    return s.mask(s.isin(list(codes)))


def decode_unit(s: pd.Series) -> pd.Series:
    """b'W', "b'W'", 'W', '' -> 'W' or NaN (2021-2023 activity units)."""

    def one(value):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        text = re.sub(r"^b?['\"]|['\"]$", "", str(value).strip()).strip().upper()
        return text if text else np.nan

    return s.map(one).astype(object)


# --------------------------------------------------------------------------
# Smoking
# --------------------------------------------------------------------------


def clean_smoking(smq: pd.DataFrame) -> pd.Series:
    """never / former / current from SMQ020 (100 cigarettes ever) + SMQ040.

    SMQ020 = 2                   -> never   (SMQ040 is blank by the skip pattern)
    SMQ020 = 1, SMQ040 = 1 or 2  -> current (every day / some days)
    SMQ020 = 1, SMQ040 = 3       -> former
    anything else (7, 9, blank)  -> NaN
    """
    ever = _codes_to_nan(_num(smq, "SMQ020"), [7, 9])
    now = _codes_to_nan(_num(smq, "SMQ040"), [7, 9])
    status = np.select(
        [ever == 2, (ever == 1) & now.isin([1, 2]), (ever == 1) & (now == 3)],
        ["never", "current", "former"],
        default="",
    )
    return pd.Series(status, index=smq.index).replace("", np.nan).astype(object)


# --------------------------------------------------------------------------
# Alcohol
# --------------------------------------------------------------------------


def clean_alcohol(alq: pd.DataFrame) -> pd.Series:
    """Average drinks per week.

    drinks_week = days_per_year(ALQ121) * ALQ130 / 52
    Never drinkers (ALQ111 = 2) and people who drank 0 times last year -> 0.
    ALQ121 is a frequency *category*; it is converted, never averaged.
    """
    ever = _codes_to_nan(_num(alq, "ALQ111"), [7, 9])
    freq = _codes_to_nan(_num(alq, "ALQ121"), [77, 99])
    per_day = _codes_to_nan(_num(alq, "ALQ130"), [777, 999])

    unknown = freq.dropna()
    unknown = unknown[~unknown.isin(list(ALQ121_DAYS_PER_YEAR))]
    if len(unknown):
        _say(f"  ALQ121: {len(unknown)} rows have a code not in the table -> NaN")

    days_year = freq.map(ALQ121_DAYS_PER_YEAR)
    drinks = days_year * per_day / WEEKS_PER_YEAR
    drinks[days_year == 0] = 0.0  # no drinking in last 12 months: ALQ130 skipped
    drinks[ever == 2] = 0.0  # never had a drink: gate question 'no'
    return drinks


# --------------------------------------------------------------------------
# Physical activity: two separate code paths (never mixed)
# --------------------------------------------------------------------------


def _domain_2017(paq: pd.DataFrame, gate: str, days: str, mins: str) -> pd.Series:
    """Minutes per week for one 2017-2018 domain (gate -> days -> minutes/day)."""
    g = _codes_to_nan(_num(paq, gate), [7, 9])
    d = _codes_to_nan(_num(paq, days), [77, 99])
    m = _codes_to_nan(_num(paq, mins), [7777, 9999])
    out = d * m
    out[g.isna()] = np.nan
    out[g == 2] = 0.0  # gate 'no' -> follow-ups are blank -> 0 minutes
    return out


def activity_2017(paq: pd.DataFrame) -> pd.Series:
    vigorous = _domain_2017(paq, "PAQ650", "PAQ655", "PAD660")
    moderate = _domain_2017(paq, "PAQ665", "PAQ670", "PAD675")
    return vigorous * VIGOROUS_WEIGHT + moderate  # NaN if either part is unknown


def _domain_2021(paq: pd.DataFrame, count: str, unit: str, mins: str) -> pd.Series:
    """Minutes per week for one 2021-2023 domain (times per unit x minutes each)."""
    n = _codes_to_nan(_num(paq, count), [77777, 99999, 7777, 9999])
    m = _codes_to_nan(_num(paq, mins), [77777, 99999, 7777, 9999])
    if unit in paq.columns:
        letter = decode_unit(paq[unit])
    else:
        letter = pd.Series(np.nan, index=paq.index, dtype=object)
    times_per_week = n * letter.map(UNIT_TO_PER_WEEK).astype(float)
    out = times_per_week * m
    out[n == 0] = 0.0  # count 0 -> unit and minutes are blank -> 0 minutes
    return out


def activity_2021(paq: pd.DataFrame) -> pd.Series:
    moderate = _domain_2021(paq, "PAD790Q", "PAD790U", "PAD800")
    vigorous = _domain_2021(paq, "PAD810Q", "PAD810U", "PAD820")
    return vigorous * VIGOROUS_WEIGHT + moderate


ACTIVITY_PATH = {"2017-2018": activity_2017, "2021-2023": activity_2021}


def clean_activity(paq: pd.DataFrame, cycle: str) -> pd.Series:
    total = ACTIVITY_PATH[cycle](paq)
    n_cap = int((total > CAP_MIN_WEEK).sum())
    if n_cap:
        _say(f"  exercise_min_week: {n_cap} totals above {CAP_MIN_WEEK:.0f} capped")
    return total.clip(upper=CAP_MIN_WEEK)


# --------------------------------------------------------------------------
# Sedentary, sleep, diet
# --------------------------------------------------------------------------


def clean_sedentary(paq: pd.DataFrame) -> pd.Series:
    return _codes_to_nan(_num(paq, "PAD680"), [7777, 9999])


def clean_sleep(slq: pd.DataFrame) -> pd.Series:
    """(5 * workday + 2 * weekend) / 7; falls back to whichever one exists."""
    code_list = [77, 99, 7777, 9999]
    workday = _codes_to_nan(_num(slq, "SLD012"), code_list)
    weekend = _codes_to_nan(_num(slq, "SLD013"), code_list)
    w_work, w_end = SLEEP_WEIGHTS
    sleep = workday.fillna(weekend)
    both = workday.notna() & weekend.notna()
    sleep[both] = (w_work * workday[both] + w_end * weekend[both]) / (w_work + w_end)
    return sleep.round(2)


def clean_diet(dbq: pd.DataFrame) -> pd.Series:
    """1 excellent ... 5 poor.  DBQ700 does not exist in 2021-2023 -> all NaN."""
    return _codes_to_nan(_num(dbq, "DBQ700"), [7, 9])


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------


def build_lifestyle(cycle: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    if cycle not in CYCLE_SUFFIX:
        raise ValueError(f"Unknown cycle {cycle!r}; use one of {list(CYCLE_SUFFIX)}")
    _say(f"=== cycle {cycle} ===")

    raw = {stem: load_raw(raw_dir, cycle, stem) for stem in STEMS}
    for stem, table in raw.items():
        _say(f"  raw {stem}: {len(table):,} rows, {table.shape[1]} columns")

    seqn = sorted(set().union(*(t.index for t in raw.values())))
    out = pd.DataFrame(index=pd.Index(seqn, name="seqn"))
    _say(f"  people across the five files (union of SEQN): {len(out):,}")

    out["smoking_status"] = clean_smoking(raw["SMQ"])
    out["alcohol_drinks_week"] = clean_alcohol(raw["ALQ"])
    out["exercise_min_week"] = clean_activity(raw["PAQ"], cycle)
    out["sedentary_min_day"] = clean_sedentary(raw["PAQ"])
    out["sleep_hours"] = clean_sleep(raw["SLQ"])
    out["diet_quality"] = clean_diet(raw["DBQ"])

    out = out.reset_index()
    out["seqn"] = out["seqn"].astype(int)
    out = out[CONTRACT_COLUMNS]
    for col in ("alcohol_drinks_week", "exercise_min_week"):
        out[col] = out[col].round(2)

    assert out["seqn"].is_unique, "seqn must be unique"
    assert list(out.columns) == CONTRACT_COLUMNS
    _summarise(out)
    return out


def _summarise(df: pd.DataFrame) -> None:
    _say(f"  output: {len(df):,} rows")
    _say("  smoking_status counts:")
    for label, n in df["smoking_status"].value_counts(dropna=False).items():
        _say(f"    {label}: {n:,}")
    _say("  % missing per column:")
    for col, pct in (df.drop(columns="seqn").isna().mean() * 100).round(1).items():
        _say(f"    {col}: {pct}%")
    numeric = df.drop(columns=["seqn", "smoking_status"])
    desc = numeric.describe().loc[["min", "50%", "mean", "max"]].round(2)
    _say("  numeric summary:\n" + desc.to_string())
    if "exercise_min_week" in df and df["exercise_min_week"].notna().any():
        share = (df["exercise_min_week"].dropna() >= 150).mean() * 100
        _say(f"  share meeting WHO 150 min/week (of those known): {share:.1f}%")


def run(cycle: str, raw_dir: Path = RAW_DIR, out_dir: Path = INTERIM_DIR) -> Path:
    df = build_lifestyle(cycle, raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lifestyle_{cycle}.csv"
    df.to_csv(path, index=False)
    _say(f"  saved {path}")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the lifestyle topic table")
    parser.add_argument("--cycle", default="2017-2018", help="2017-2018, 2021-2023 or all")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=INTERIM_DIR)
    args = parser.parse_args(argv)
    cycles = list(CYCLE_SUFFIX) if args.cycle == "all" else [args.cycle]
    for cycle in cycles:
        run(cycle, args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()

"""Synthetic stand-ins for the raw NHANES files and the other topic tables.

Only used for testing the pipeline code when the real data is not present.
The shapes mimic the real files: skip patterns, codes (7/9, 77/99 ...), the
5.4e-79 zeros and the byte units in 2021-2023.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TINY = 5.4e-79


def _pick(rng, n, values, probs):
    return rng.choice(values, size=n, p=probs)


def make_raw_lifestyle(raw_dir: Path, cycle: str, n: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    folder = Path(raw_dir) / cycle
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "J" if cycle == "2017-2018" else "L"
    base = 93703 if cycle == "2017-2018" else 130378
    seqn = np.arange(base, base + n)

    # SMQ: gate SMQ020, follow-up SMQ040 only when SMQ020 == 1
    smq020 = _pick(rng, n, [1, 2, 7, 9], [0.4, 0.57, 0.02, 0.01])
    smq040 = np.where(smq020 == 1, _pick(rng, n, [1, 2, 3], [0.3, 0.1, 0.6]), np.nan)
    pd.DataFrame({"SEQN": seqn, "SMQ020": smq020, "SMQ040": smq040}).to_csv(
        folder / f"SMQ_{suffix}.csv", index=False
    )

    # ALQ: adults only (drop a few rows), skip patterns, codes
    keep = rng.random(n) > 0.15
    a = seqn[keep]
    m = len(a)
    alq111 = _pick(rng, m, [1, 2], [0.85, 0.15])
    alq121 = np.where(alq111 == 1, rng.integers(0, 11, size=m).astype(float), np.nan)
    alq121 = np.where(rng.random(m) < 0.03, 77, alq121)
    alq121 = np.where(alq121 == 0, TINY, alq121)  # stored zero
    alq130 = np.where((alq111 == 1) & (alq121 != TINY), rng.integers(1, 8, size=m).astype(float), np.nan)
    alq130 = np.where(rng.random(m) < 0.03, 999, alq130)
    pd.DataFrame({"SEQN": a, "ALQ111": alq111, "ALQ121": alq121, "ALQ130": alq130}).to_csv(
        folder / f"ALQ_{suffix}.csv", index=False
    )

    # PAQ
    sed = rng.integers(60, 900, size=n).astype(float)
    sed = np.where(rng.random(n) < 0.05, 9999, sed)
    sed = np.where(rng.random(n) < 0.01, TINY, sed)
    if cycle == "2017-2018":
        cols = {"SEQN": seqn, "PAD680": sed}
        for gate, days, mins, p in (("PAQ650", "PAQ655", "PAD660", 0.25), ("PAQ665", "PAQ670", "PAD675", 0.4)):
            g = _pick(rng, n, [1, 2, 7], [p, 1 - p - 0.01, 0.01])
            cols[gate] = g
            cols[days] = np.where(g == 1, rng.integers(1, 8, size=n).astype(float), np.nan)
            mins_v = np.where(g == 1, rng.integers(10, 180, size=n).astype(float), np.nan)
            cols[mins] = np.where((g == 1) & (rng.random(n) < 0.03), 9999, mins_v)
        pd.DataFrame(cols).to_csv(folder / f"PAQ_{suffix}.csv", index=False)
    else:
        cols = {"SEQN": seqn, "PAD680": sed}
        for q, u, mn, p in (("PAD790Q", "PAD790U", "PAD800", 0.5), ("PAD810Q", "PAD810U", "PAD820", 0.25)):
            count = np.where(rng.random(n) < p, rng.integers(1, 8, size=n).astype(float), 0.0)
            unit = np.where(count > 0, rng.choice(["D", "W", "M", "Y"], size=n, p=[0.1, 0.7, 0.15, 0.05]), "")
            minutes = np.where(count > 0, rng.integers(10, 120, size=n).astype(float), np.nan)
            cols[q] = np.where(count == 0, TINY, count)
            cols[u] = [f"b'{x}'" for x in unit]
            cols[mn] = minutes
        pd.DataFrame(cols).to_csv(folder / f"PAQ_{suffix}.csv", index=False)

    # SLQ
    s = seqn[rng.random(n) > 0.1]
    pd.DataFrame(
        {
            "SEQN": s,
            "SLD012": np.round(rng.uniform(4, 11, size=len(s)) * 2) / 2,
            "SLD013": np.round(rng.uniform(4, 12, size=len(s)) * 2) / 2,
        }
    ).to_csv(folder / f"SLQ_{suffix}.csv", index=False)

    # DBQ (DBQ700 only in 2017-2018)
    d = {"SEQN": seqn}
    if cycle == "2017-2018":
        d["DBQ700"] = _pick(rng, n, [1, 2, 3, 4, 5, 7, 9], [0.1, 0.2, 0.35, 0.2, 0.1, 0.03, 0.02])
    else:
        d["DBQ010"] = rng.integers(1, 3, size=n)
    pd.DataFrame(d).to_csv(folder / f"DBQ_{suffix}.csv", index=False)


def make_other_topics(seqn: np.ndarray, seed: int = 1) -> dict[str, pd.DataFrame]:
    """Plausible people / body / mental tables that follow the contract."""
    rng = np.random.default_rng(seed)
    n = len(seqn)
    binary = lambda p: (rng.random(n) < p).astype(float)  # noqa: E731
    people = pd.DataFrame(
        {
            "seqn": seqn,
            "age": rng.integers(20, 81, size=n),
            "sex": rng.choice(["male", "female"], size=n),
            "education": rng.integers(1, 6, size=n),
            "income_ratio": np.round(rng.uniform(0, 5, size=n), 2),
            "diabetes": binary(0.12),
            "heart_failure": binary(0.03),
            "coronary_heart_disease": binary(0.04),
            "angina": binary(0.03),
            "heart_attack": binary(0.04),
            "stroke": binary(0.03),
        }
    )
    cm = people[["diabetes", "heart_failure", "coronary_heart_disease", "angina", "heart_attack", "stroke"]].max(axis=1)
    people["cardiometabolic"] = cm.astype(float)

    body = pd.DataFrame(
        {
            "seqn": seqn,
            "height_cm": np.round(rng.normal(168, 10, n), 1),
            "weight_kg": np.round(rng.normal(80, 18, n).clip(35, 200), 1),
            "waist_cm": np.round(rng.normal(98, 14, n).clip(60, 160), 1),
            "sbp": np.round(rng.normal(122, 16, n).clip(80, 220)),
            "dbp": np.round(rng.normal(72, 10, n).clip(40, 120)),
            "pulse": np.round(rng.normal(72, 11, n).clip(45, 130)),
            "told_high_bp": binary(0.3),
            "total_chol": np.where(rng.random(n) < 0.4, np.nan, np.round(rng.normal(190, 38, n).clip(100, 400))),
            "hdl": np.where(rng.random(n) < 0.4, np.nan, np.round(rng.normal(53, 15, n).clip(20, 120))),
            "hba1c": np.where(rng.random(n) < 0.4, np.nan, np.round(rng.normal(5.7, 0.9, n).clip(4, 14), 1)),
        }
    )
    body["bmi"] = np.round(body["weight_kg"] / (body["height_cm"] / 100) ** 2, 1).clip(14, 70)

    score = rng.integers(0, 28, size=n)
    mental = pd.DataFrame(
        {
            "seqn": seqn,
            "phq9_score": np.where(rng.random(n) < 0.1, np.nan, score),
        }
    )
    mental["phq9_category"] = pd.cut(
        mental["phq9_score"], [-1, 4, 9, 14, 19, 27], labels=["none", "mild", "moderate", "moderately severe", "severe"]
    ).astype(object)
    return {"people": people, "body": body, "mental": mental}


def make_merged(lifestyle: pd.DataFrame) -> pd.DataFrame:
    topics = make_other_topics(lifestyle["seqn"].to_numpy())
    merged = topics["people"]
    for name in ("body", "lifestyle", "mental"):
        table = lifestyle if name == "lifestyle" else topics[name]
        merged = merged.merge(table, on="seqn", how="left", validate="one_to_one")
    return merged

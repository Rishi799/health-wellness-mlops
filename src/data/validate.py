"""Data validation for the whole team (Person 3, pandera).

Automatic tests for data.  One schema per topic file (people, body, lifestyle,
mental) plus one for the merged file.  Every failure is reported together
(lazy=True), a JSON report is written, and the process exits with code 1 when
any check fails so that `dvc repro` stops at the validate stage.

Run from the repo root:

    python -m src.data.validate --cycle 2017-2018
    python -m src.data.validate --cycle 2017-2018 --only lifestyle
    python -m src.data.validate --schema merged --path some_file.csv --cycle 2017-2018

Ranges, categories and missing-% limits live in the tables at the top of this
file.  They are the written agreement with each file owner - change them only
after a team decision, never just to make a failing check pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = REPO_ROOT / "data" / "interim"
REPORTS_DIR = REPO_ROOT / "reports"

ID = "seqn"


# --------------------------------------------------------------------------
# The agreement: what every column may contain
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    kind: str  # "num" | "binary" | "cat" | "any"
    lo: float | None = None
    hi: float | None = None
    categories: tuple | None = None


NUM = lambda lo, hi: Spec("num", lo, hi)  # noqa: E731
BINARY = Spec("binary", 0, 1)

TOPIC_COLUMNS: dict[str, dict[str, Spec]] = {
    # Person 1.  TO CONFIRM with Person 1: sex / education encodings.
    "people": {
        "age": NUM(0, 80),
        "sex": Spec("any"),
        "education": Spec("any"),
        "income_ratio": NUM(0, 5),
        "diabetes": BINARY,
        "heart_failure": BINARY,
        "coronary_heart_disease": BINARY,
        "angina": BINARY,
        "heart_attack": BINARY,
        "stroke": BINARY,
        "cardiometabolic": BINARY,
    },
    # Person 2.  TO CONFIRM with Person 2.
    "body": {
        "height_cm": NUM(50, 230),
        "weight_kg": NUM(2, 300),
        "bmi": NUM(12, 80),
        "waist_cm": NUM(30, 200),
        "sbp": NUM(60, 260),
        "dbp": NUM(20, 150),
        "pulse": NUM(30, 200),
        "told_high_bp": BINARY,
        "total_chol": NUM(50, 600),
        "hdl": NUM(10, 200),
        "hba1c": NUM(3, 20),
    },
    # Person 3 (mine).
    "lifestyle": {
        "smoking_status": Spec("cat", categories=("never", "former", "current")),
        "alcohol_drinks_week": NUM(0, 110),
        "exercise_min_week": NUM(0, 3000),  # lifestyle.py caps at 3000
        "sedentary_min_day": NUM(0, 1440),
        "sleep_hours": NUM(2, 14),
        "diet_quality": NUM(1, 5),
    },
    # Person 4.  TO CONFIRM category words with Person 4.
    "mental": {
        "phq9_score": NUM(0, 27),
        "phq9_category": Spec("any"),
    },
}
TOPICS = tuple(TOPIC_COLUMNS)

# The merged file is validated against every contract column, but the people
# table may include children while the model population is adults 20-80.
MERGED_RANGE_OVERRIDES = {"age": (20, 80)}
# No label -> unusable row.  Person 1 drops unknown-target rows before the
# merged file is written; if that is not agreed, change it as a team decision.
MERGED_TARGET_NULLABLE = False
TARGET = "cardiometabolic"

# Missing-% limits (fraction).  Tuned on the real NHANES topic files.
DEFAULT_MAX_MISSING = 0.40
MAX_MISSING_OVERRIDES: dict[str, float] = {
    "total_chol": 0.70,  # lab tests were done on a subsample
    "hdl": 0.70,
    "hba1c": 0.70,
    # ALQ is asked of adults only; many SEQN in the union never appear, or skip/refuse.
    "alcohol_drinks_week": 0.60,
}
# (column, cycle) -> limit.  diet_quality does not exist in 2021-2023.
MAX_MISSING_BY_CYCLE: dict[tuple[str, str], float] = {
    ("diet_quality", "2021-2023"): 1.0,
}

# Raw NHANES codes that must never survive cleaning.
ALL_CODES = (7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999)
TINY = 1e-6  # nonzero values smaller than this are float noise (5.4e-79)


def max_missing(column: str, cycle: str) -> float:
    if (column, cycle) in MAX_MISSING_BY_CYCLE:
        return MAX_MISSING_BY_CYCLE[(column, cycle)]
    return MAX_MISSING_OVERRIDES.get(column, DEFAULT_MAX_MISSING)


# --------------------------------------------------------------------------
# Building schemas
# --------------------------------------------------------------------------


def _check(name: str, fn, error: str) -> pa.Check:
    # pandera reports `error` instead of `name` in failure_cases, so the human
    # wording goes in `description` and is looked up when the report is built.
    return pa.Check(fn, name=name, description=error)


def _numeric_checks(lo: float, hi: float) -> list[pa.Check]:
    checks = [
        _check(f"range[{lo:g},{hi:g}]", lambda s, lo=lo, hi=hi: s.between(lo, hi), f"value outside {lo:g}..{hi:g}"),
        _check("no_tiny_values", lambda s: ~((s != 0) & (s.abs() < TINY)), "float-noise value such as 5.4e-79"),
    ]
    # Raw codes that cannot be real values for this column.
    codes = [c for c in ALL_CODES if c > hi]
    if codes:
        checks.append(
            _check(
                "no_leftover_codes",
                lambda s, codes=tuple(codes): ~s.isin(codes),
                f"leftover NHANES code ({', '.join(str(c) for c in codes)})",
            )
        )
    return checks


def _column(spec: Spec, lo_hi: tuple[float, float] | None = None, nullable: bool = True) -> pa.Column:
    if spec.kind == "any":
        return pa.Column(nullable=nullable, required=True)
    if spec.kind == "cat":
        allowed = list(spec.categories)
        return pa.Column(
            str,
            checks=[_check("allowed_categories", lambda s, a=allowed: s.isin(a), f"not one of {allowed}")],
            nullable=nullable,
        )
    if spec.kind == "binary":
        checks = [
            _check("binary_0_1", lambda s: s.isin([0, 1]), "must be 0 or 1"),
            _check("no_leftover_codes", lambda s: ~s.isin([7, 9, 77, 99]), "leftover NHANES code (7, 9, 77, 99)"),
        ]
        return pa.Column(float, checks=checks, nullable=nullable, coerce=True)
    lo, hi = lo_hi if lo_hi else (spec.lo, spec.hi)
    return pa.Column(float, checks=_numeric_checks(lo, hi), nullable=nullable, coerce=True)


def _id_column() -> pa.Column:
    return pa.Column(
        int,
        checks=[_check("positive_id", lambda s: s > 0, "seqn must be a positive integer")],
        nullable=False,
        unique=True,
        coerce=True,
    )


def schema_columns(name: str) -> dict[str, tuple[Spec, tuple[float, float] | None, bool]]:
    """column -> (spec, range override, nullable) for a schema name."""
    if name == "merged":
        cols = {}
        for topic in TOPICS:
            for col, spec in TOPIC_COLUMNS[topic].items():
                nullable = not (col == TARGET and not MERGED_TARGET_NULLABLE)
                cols[col] = (spec, MERGED_RANGE_OVERRIDES.get(col), nullable)
        return cols
    if name in TOPIC_COLUMNS:
        return {c: (s, None, True) for c, s in TOPIC_COLUMNS[name].items()}
    raise KeyError(f"Unknown schema {name!r}; use one of {[*TOPICS, 'merged']}")


def build_schema(name: str) -> pa.DataFrameSchema:
    columns = {ID: _id_column()}
    for col, (spec, rng, nullable) in schema_columns(name).items():
        columns[col] = _column(spec, rng, nullable)
    # strict=False: extra columns are allowed (contract rule 8), renamed or
    # missing contract columns are not.
    return pa.DataFrameSchema(columns, strict=False, name=name)


# --------------------------------------------------------------------------
# Running a schema and building the report
# --------------------------------------------------------------------------


def _implicit_name(check: str) -> str:
    c = str(check)
    if c.startswith("column_in_dataframe"):
        return "column_present"
    if c.startswith(("dtype", "coerce_dtype")):
        return "type"
    if c.startswith("not_nullable"):
        return "not_null"
    if c.startswith("field_uniqueness"):
        return "unique"
    return c


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        return int(value) if float(value).is_integer() else float(value)
    return value if isinstance(value, (str, int, bool)) or value is None else str(value)


def validate_dataframe(df: pd.DataFrame, schema_name: str, cycle: str, source: str = "") -> dict:
    """Validate one DataFrame.  Returns a report dict (never raises on bad data)."""
    schema = build_schema(schema_name)
    checks: list[dict] = []
    failed_keys: set[tuple[str, str]] = set()
    rules = {
        (col_name, chk.name): chk.description
        for col_name, col in schema.columns.items()
        for chk in col.checks
    }

    # 1. pandera schema, all failures at once
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        fc = exc.failure_cases.copy()
        fc["column"] = fc["column"].fillna("(dataframe)")
        # for a missing column pandera puts the schema name in `column` and the
        # missing column's name in `failure_case`
        absent = fc["check"].astype(str).str.startswith("column_in_dataframe")
        fc.loc[absent, "column"] = fc.loc[absent, "failure_case"]
        fc["check_name"] = fc["check"].map(_implicit_name)
        for (column, check), group in fc.groupby(["column", "check_name"], sort=False):
            values = group["failure_case"].drop_duplicates().head(5).map(_json_safe).tolist()
            checks.append(
                {
                    "column": column,
                    "check": check,
                    "rule": rules.get((column, check), ""),
                    "status": "failed",
                    "failed_count": int(len(group)),
                    "bad_values": values,
                }
            )
            failed_keys.add((column, check))

    # 2. the passed checks (everything declared that did not fail)
    declared: list[tuple[str, str]] = []
    for col_name, col in schema.columns.items():
        declared.append((col_name, "column_present"))
        declared.append((col_name, "type"))
        if not col.nullable:
            declared.append((col_name, "not_null"))
        if col.unique:
            declared.append((col_name, "unique"))
        declared.extend((col_name, chk.name) for chk in col.checks)
    for column, check in declared:
        if (column, check) not in failed_keys:
            checks.append({"column": column, "check": check, "status": "passed", "failed_count": 0, "bad_values": []})

    # 3. missingness limits (manual: pandera would dump the whole series)
    missing = {}
    for col_name in schema.columns:
        if col_name not in df.columns:
            continue
        frac = float(df[col_name].isna().mean()) if len(df) else 0.0
        limit = max_missing(col_name, cycle)
        missing[col_name] = round(frac * 100, 2)
        ok = frac <= limit + 1e-12
        checks.append(
            {
                "column": col_name,
                "check": f"missing<={limit * 100:g}%",
                "status": "passed" if ok else "failed",
                "failed_count": 0 if ok else int(df[col_name].isna().sum()),
                "bad_values": [] if ok else [f"{frac * 100:.1f}% missing"],
            }
        )

    # 4. not empty
    checks.append(
        {
            "column": "(dataframe)",
            "check": "not_empty",
            "status": "passed" if len(df) else "failed",
            "failed_count": 0 if len(df) else 1,
            "bad_values": [],
        }
    )

    n_failed = sum(c["status"] == "failed" for c in checks)
    return {
        "schema": schema_name,
        "cycle": cycle,
        "source": source,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "passed": n_failed == 0,
        "n_checks": len(checks),
        "n_failed": n_failed,
        "pct_missing": missing,
        "checks": checks,
    }


def format_failures(report: dict) -> list[str]:
    """Clear messages: schema, column, rule, how many, which bad values."""
    lines = []
    for c in report["checks"]:
        if c["status"] == "failed":
            values = ", ".join(str(v) for v in c["bad_values"])
            rule = f" ({c['rule']})" if c.get("rule") else ""
            lines.append(
                f"  [FAIL] {report['schema']}: column '{c['column']}' rule '{c['check']}'{rule} - "
                f"{c['failed_count']} row(s); bad values: {values}"
            )
    return lines


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _targets(args) -> list[tuple[str, Path, bool]]:
    """(schema, path, required) to validate."""
    if args.schema:
        if not args.path:
            raise SystemExit("--schema needs --path")
        return [(args.schema, args.path, True)]
    names = [args.only] if args.only else [*TOPICS, "merged"]
    out = []
    for name in names:
        path = INTERIM_DIR / f"{name}_{args.cycle}.csv"
        # the merged file is required; topic files are validated if present
        out.append((name, path, name == "merged" or bool(args.only)))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the interim data files")
    parser.add_argument("--cycle", default="2017-2018")
    parser.add_argument("--only", choices=[*TOPICS, "merged"], help="validate just one file")
    parser.add_argument("--schema", choices=[*TOPICS, "merged"], help="schema to use with --path")
    parser.add_argument("--path", type=Path, help="validate this file instead of data/interim")
    parser.add_argument("--report", type=Path, help="report path (default reports/validation_<cycle>.json)")
    args = parser.parse_args(argv)

    reports, skipped = [], []
    for name, path, required in _targets(args):
        if not path.exists():
            if required:
                reports.append(
                    {
                        "schema": name,
                        "cycle": args.cycle,
                        "source": str(path),
                        "rows": 0,
                        "columns": 0,
                        "passed": False,
                        "n_checks": 1,
                        "n_failed": 1,
                        "pct_missing": {},
                        "checks": [
                            {
                                "column": "(file)",
                                "check": "file_exists",
                                "status": "failed",
                                "failed_count": 1,
                                "bad_values": [f"{path} not found"],
                            }
                        ],
                    }
                )
            else:
                skipped.append(name)
            continue
        reports.append(validate_dataframe(_read(path), name, args.cycle, str(path)))

    ok = all(r["passed"] for r in reports)
    result = {
        "cycle": args.cycle,
        "passed": ok,
        "schemas_validated": [r["schema"] for r in reports],
        "schemas_skipped_not_found": skipped,
        "reports": reports,
    }
    report_path = args.report or REPORTS_DIR / f"validation_{args.cycle}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, default=str))

    for r in reports:
        status = "PASSED" if r["passed"] else "FAILED"
        print(f"[validate] {r['schema']:<9} {status}  ({r['rows']:,} rows, {r['n_checks']} checks, {r['n_failed']} failed)")
        for line in format_failures(r):
            print(line)
    if skipped:
        print(f"[validate] skipped (file not found): {', '.join(skipped)}")
    print(f"[validate] report written to {report_path}")
    if not ok:
        print("[validate] VALIDATION FAILED - stopping the pipeline", file=sys.stderr)
        return 1
    print("[validate] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Generate a deterministic synthetic clinical dataset + golden expected output.

The dataset is intentionally small (~30 rows) and includes:

  * patients of varied ages (including <18 and >=65),
  * mixed response categories,
  * a couple of rows with malformed dates and missing labs,
  * one row where visit_date < treatment_start_date (negative duration —
    exercises the invariant + refiner path).

The golden file is computed by the same canonical fallbacks the system uses,
so a clean run on this dataset should score 100% correctness against it.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "samples"
GOLDEN_DIR = PROJECT / "data" / "golden"
DATA_DIR.mkdir(parents=True, exist_ok=True)
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


# ---- 30 deterministic synthetic patients -------------------------------------
ROWS: list[dict] = []
for i in range(1, 31):
    age = [12, 17, 18, 24, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
           22, 29, 33, 41, 48, 52, 58, 63, 68, 72, 77, 82, 19, 25, 90][i - 1]
    sex = "F" if i % 2 == 0 else "M"
    start = date(2024, 1, 1) + timedelta(days=(i * 3) % 60)
    visit = start + timedelta(days=[10, 30, 60, 90, 120, 5, 0, 15, 45, 75,
                                    20, 50, 80, 110, 7, 11, 25, 35, 55,
                                    65, 100, 14, 40, 22, 99, 17, 28, 42,
                                    -3, 60][i - 1])
    if i == 29:  # explicit negative duration row
        visit = start - timedelta(days=3)
    lab = [4.1, 5.2, 6.8, 7.9, 3.2, 8.4, 5.7, 6.1, 4.4, 7.0,
           5.5, 6.9, 7.3, 8.1, 4.9, 5.0, 6.2, 5.8, 6.6, 7.1,
           4.3, 5.9, 6.4, 7.7, 8.5, 4.7, 5.3, 6.7, None, 9.0][i - 1]
    response = ["CR", "PR", "SD", "PD", "NE", "CR", "PR", "SD",
                "PD", "CR", "PR", "SD", "PD", "CR", "NE",
                "PR", "SD", "CR", "PD", "PR", "SD", "CR", "PR",
                "PD", "CR", "SD", "PR", "CR", None, "PD"][i - 1]
    ROWS.append({
        "patient_id": f"P{i:03d}",
        "age": age,
        "sex": sex,
        "treatment_start_date": start.isoformat(),
        "visit_date": visit.isoformat(),
        "lab_value": "" if lab is None else lab,
        "response": "" if response is None else response,
    })


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


write_csv(DATA_DIR / "clinical_sample.csv", ROWS)


# ---- Golden expected output -------------------------------------------------
def parse_int(x):
    try:
        return int(x)
    except Exception:
        return None


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def age_group(age):
    if age is None:
        return None
    if age < 18:
        return "<18"
    if age < 65:
        return "18-64"
    return ">=65"


def days_between(later: str, earlier: str):
    try:
        a = date.fromisoformat(later)
        b = date.fromisoformat(earlier)
        return (a - b).days
    except Exception:
        return None


def treatment_duration(start, visit):
    d = days_between(visit, start)
    if d is None or d < 0:
        return None
    return int(d)


def response_flag(r):
    if not r:
        return None
    r = r.strip().upper()
    if r in ("CR", "PR"):
        return "RESPONDER"
    if r in ("SD", "PD"):
        return "NON_RESPONDER"
    return None


def analysis_pop(age, start, visit):
    age = parse_int(age)
    if age is None or age < 18:
        return "N"
    d = days_between(visit, start)
    if d is None or d < 1:
        return "N"
    return "Y"


def risk_group(age, lab):
    age = parse_int(age)
    lab = parse_float(lab)
    if age is None or lab is None:
        return None
    score = 0
    if lab > 7.0:
        score += 2
    elif lab > 5.0:
        score += 1
    if age >= 65:
        score += 2
    elif age >= 50:
        score += 1
    if score >= 3:
        return "HIGH"
    if score >= 1:
        return "MEDIUM"
    return "LOW"


golden = []
for r in ROWS:
    age = parse_int(r["age"])
    golden.append({
        "patient_id": r["patient_id"],
        "AGE_GROUP": age_group(age),
        "TREATMENT_DURATION": treatment_duration(r["treatment_start_date"], r["visit_date"]),
        "RESPONSE_FLAG": response_flag(r["response"]),
        "ANALYSIS_POP_FLAG": analysis_pop(r["age"], r["treatment_start_date"], r["visit_date"]),
        "RISK_GROUP": risk_group(r["age"], r["lab_value"]),
    })

write_csv(GOLDEN_DIR / "expected.csv", golden)
print(f"Wrote {DATA_DIR / 'clinical_sample.csv'} and {GOLDEN_DIR / 'expected.csv'}")

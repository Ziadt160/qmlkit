"""The credit-risk dataset, real if you have it and a stand-in if you do not.

The example this supports is about *method*, not about this particular table, so it
has to run for someone who has not downloaded anything. But a synthetic stand-in that
quietly pretends to be the real thing is exactly the sort of dishonesty the evaluation
layer exists to catch, so:

* if ``credit_risk_dataset.csv`` is next to this file (or at ``QMLKIT_CREDIT_CSV``),
  it is used, and everything downstream is a real result about real data;
* otherwise a synthetic table with the same twelve columns, the same ~21.8% default
  rate and plausible dependencies is generated, and **every report says so**.

Get the real file from
<https://www.kaggle.com/datasets/laotse/credit-risk-dataset> (Kaggle requires a login,
so it cannot be downloaded automatically) and drop it beside this script.

The synthetic generator is not a model of credit risk. It is a table with the right
*shape* — skewed target, mixed categorical and numeric columns, a couple of strongly
predictive features, several useless ones, and missing values in the two columns that
have them for real — so that the pipeline steps in the example have something to bite
on.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["load_credit_risk", "REAL_COLUMNS"]

#: The dataset's own schema, in its own order.
REAL_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

_HOME = ["RENT", "OWN", "MORTGAGE", "OTHER"]
_HOME_P = [0.505, 0.079, 0.413, 0.003]
_INTENT = [
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "PERSONAL",
    "DEBTCONSOLIDATION",
    "HOMEIMPROVEMENT",
]
_INTENT_P = [0.199, 0.187, 0.175, 0.169, 0.159, 0.111]
_GRADE = ["A", "B", "C", "D", "E", "F", "G"]
_GRADE_P = [0.333, 0.320, 0.199, 0.111, 0.030, 0.006, 0.001]
#: Interest rate rises with grade; so does default risk. This is the main signal.
_GRADE_RATE = {"A": 7.3, "B": 10.9, "C": 13.5, "D": 15.4, "E": 17.0, "F": 18.6, "G": 20.3}
_GRADE_RISK = {"A": -1.5, "B": -0.6, "C": 0.2, "D": 1.4, "E": 1.9, "F": 2.3, "G": 2.6}


def _synthesise(n_rows: int, seed: int) -> dict[str, npt.NDArray[Any]]:
    """A table shaped like the real one, with a target that is genuinely learnable."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.gamma(9.0, 3.1, n_rows) + 19, 20, 84).astype(int)
    income = np.clip(rng.lognormal(11.0, 0.62, n_rows), 4000, 2_000_000).astype(int)
    home = rng.choice(_HOME, n_rows, p=_HOME_P)
    emp_length = np.clip(rng.gamma(2.0, 2.4, n_rows), 0, 41)
    intent = rng.choice(_INTENT, n_rows, p=_INTENT_P)
    grade = rng.choice(_GRADE, n_rows, p=_GRADE_P)
    amount = np.clip(rng.lognormal(9.1, 0.62, n_rows), 500, 35_000).astype(int)
    rate = np.clip(
        np.array([_GRADE_RATE[g] for g in grade]) + rng.normal(0, 1.0, n_rows), 5.4, 23.2
    )
    percent_income = np.clip(amount / np.maximum(income, 1.0), 0.0, 0.83)
    prior_default = rng.random(n_rows) < 0.176

    # the target: driven by burden, grade and prior default, plus real noise. The
    # useless columns (intent, age, employment length) stay uncorrelated on purpose --
    # a pipeline that cannot drop them is a pipeline worth exposing.
    logit = (
        -2.55
        + 6.1 * percent_income
        + np.array([_GRADE_RISK[g] for g in grade])
        + 0.85 * prior_default
        + 0.55 * (home == "RENT")
        - 0.45 * (home == "MORTGAGE")
        - 0.35 * np.log1p(income) / 3.0
        + rng.normal(0, 0.85, n_rows)
    )
    status = (rng.random(n_rows) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    # the two columns that really do have gaps, at roughly the real rates
    emp_length[rng.random(n_rows) < 0.027] = np.nan
    rate[rng.random(n_rows) < 0.096] = np.nan

    return {
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home,
        "person_emp_length": emp_length,
        "loan_intent": intent,
        "loan_grade": grade,
        "loan_amnt": amount,
        "loan_int_rate": rate,
        "loan_status": status,
        "loan_percent_income": percent_income,
        "cb_person_default_on_file": np.where(prior_default, "Y", "N"),
        "cb_person_cred_hist_length": np.clip(
            (age - 20) * rng.uniform(0.1, 0.45, n_rows) + 2, 2, 30
        ).astype(int),
    }


def load_credit_risk(
    n_rows: int | None = None, seed: int = 0
) -> tuple[Any, bool]:
    """``(frame, is_real)``. Never silently substitutes one for the other.

    ``n_rows`` subsamples — stratified on the target, so the class balance the whole
    example is about is preserved. A quantum model refitted across folds on 32,000
    rows is an overnight job; the point here is the method, not the row count.
    """
    import pandas as pd

    here = Path(__file__).resolve().parent
    candidates = [
        Path(os.environ["QMLKIT_CREDIT_CSV"]) if os.environ.get("QMLKIT_CREDIT_CSV") else None,
        here / "credit_risk_dataset.csv",
        here.parent / "credit_risk_dataset.csv",
    ]
    path = next((p for p in candidates if p is not None and p.is_file()), None)

    if path is not None:
        frame = pd.read_csv(path)
        missing = [c for c in REAL_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing expected column(s): {missing}")
        is_real = True
    else:
        frame = pd.DataFrame(_synthesise(32_581, seed))
        is_real = False

    if n_rows is not None and n_rows < len(frame):
        from qmlkit.imbalance import stratified_split

        keep_fraction = n_rows / len(frame)
        _, keep = stratified_split(
            frame["loan_status"].to_numpy(), test_size=keep_fraction, seed=seed
        )
        frame = frame.iloc[keep].reset_index(drop=True)
    return frame, is_real


if __name__ == "__main__":  # a quick look at what the stand-in produces
    frame, is_real = load_credit_risk()
    print(f"source: {'REAL Kaggle CSV' if is_real else 'SYNTHETIC stand-in'}")
    print(f"rows: {len(frame):,}   default rate: {frame['loan_status'].mean():.3f}")
    print(frame.head(4).to_string())
    print("\nmissing values:")
    print(frame.isna().sum()[lambda s: s > 0].to_string())

"""Fit sigmoid calibration on early validation; assess on later validation."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from train_initial import FEATURES


ROOT = Path(__file__).resolve().parents[2]


def score_logit(probabilities):
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def metrics(name, target, scores):
    return {
        "model": name,
        "records": len(target),
        "positive_rate": float(target.mean()),
        "average_precision": float(average_precision_score(target, scores)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "brier_score": float(brier_score_loss(target, scores)),
        "log_loss": float(log_loss(target, scores)),
    }


def main():
    df = pd.read_csv(
        ROOT / "data/processed/bookings_validated.csv",
        parse_dates=["arrival_date", "reservation_status_date"],
    )
    assignments = pd.read_csv(
        ROOT / "data/processed/split_assignments.csv"
    )
    df = df.merge(
        assignments, on="booking_id",
        how="left", validate="one_to_one",
    )

    if df["split"].isna().any():
        raise ValueError("Missing split assignments")

    validation = df.loc[df["split"].eq("validation")].copy()
    cutoff = pd.Timestamp("2017-03-01")

    early = validation["arrival_date"].lt(cutoff)
    labels_available = validation["reservation_status_date"].lt(cutoff)

    calibration = validation.loc[early & labels_available]
    assessment = validation.loc[~early]

    if calibration.empty or assessment.empty:
        raise ValueError("Calibration or assessment period is empty")

    for subset in (calibration, assessment):
        if subset["IsCanceled"].nunique() != 2:
            raise ValueError("Both outcome classes are required")

    base_model = joblib.load(ROOT / "models/random_forest.joblib")
    calibration_scores = base_model.predict_proba(
        calibration[FEATURES]
    )[:, 1]
    assessment_scores = base_model.predict_proba(
        assessment[FEATURES]
    )[:, 1]

    # One sigmoid mapping; the underlying forest is unchanged.
    calibrator = LogisticRegression(C=1e6, max_iter=1000)
    calibrator.fit(
        score_logit(calibration_scores),
        calibration["IsCanceled"],
    )

    slope = float(calibrator.coef_[0, 0])
    if slope <= 0:
        raise ValueError(
            "Calibration reverses score ordering; investigate before saving"
        )

    adjusted_scores = calibrator.predict_proba(
        score_logit(assessment_scores)
    )[:, 1]

    results = [
        metrics(
            "random_forest_unadjusted",
            assessment["IsCanceled"],
            assessment_scores,
        ),
        metrics(
            "random_forest_sigmoid",
            assessment["IsCanceled"],
            adjusted_scores,
        ),
    ]

    report = {
        "calibration_period": "January-February 2017",
        "assessment_period": "March-April 2017",
        "calibration_records": len(calibration),
        "early_records_held_aside_for_late_labels": int(
            (early & ~labels_available).sum()
        ),
        "assessment_records": len(assessment),
        "calibrator_slope": slope,
        "calibrator_intercept": float(calibrator.intercept_[0]),
        "input_transform": "logit of probability clipped to [1e-6, 1-1e-6]",
        "results": results,
        "notes": [
            "Base forest unchanged and trained on original training split.",
            "Calibration fit only on early-validation available labels.",
            "Both variants assessed on identical later-validation records.",
            "Validation already informed development; this is not an independent final test.",
            "Final test remains unscored.",
            "Calibration is not automatically selected based on this experiment."
        ],
    }

    report_path = ROOT / "reports/metrics/calibration_experiment.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        calibrator,
        ROOT / "models/random_forest_sigmoid_calibrator.joblib",
    )

    print(f"Calibration records: {len(calibration):,}")
    print(
        "Early records held aside for late labels:",
        report["early_records_held_aside_for_late_labels"],
    )
    print(f"Assessment records: {len(assessment):,}")
    print(f"Calibration slope: {slope:.4f}")
    print("\nMARCH-APRIL ASSESSMENT")
    print(pd.DataFrame(results).round(4).to_string(index=False))
    print(f"\nSaved: {report_path}")
    print("Saved: models/random_forest_sigmoid_calibrator.joblib")
    print("Final test remains unscored.")


if __name__ == "__main__":
    main()

"""Evaluate frozen models on the final chronological test cohort."""

from pathlib import Path
import hashlib
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]


def main():
    report_path = ROOT / "reports/metrics/final_test.json"
    if report_path.exists():
        raise FileExistsError(
            "Final-test report already exists. Do not overwrite it."
        )

    selection = json.loads(
        (ROOT / "docs/model_selection.json").read_text(encoding="utf-8")
    )

    for relative_path, expected_hash in selection["artifact_sha256"].items():
        actual_hash = hashlib.sha256(
            (ROOT / relative_path).read_bytes()
        ).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen artifact changed: {relative_path}")

    features = (
        selection["numeric_features"]
        + selection["categorical_features"]
    )

    df = pd.read_csv(ROOT / "data/processed/bookings_validated.csv")
    assignments = pd.read_csv(
        ROOT / "data/processed/split_assignments.csv"
    )
    df = df.merge(
        assignments, on="booking_id",
        how="left", validate="one_to_one",
    )

    if df["split"].isna().any():
        raise ValueError("Missing split assignments")

    train = df.loc[df["split"].eq("train")]
    test = df.loc[df["split"].eq("test")].copy()

    if len(test) != selection["final_test_expected_records"]:
        raise ValueError("Unexpected final-test record count")

    target = test["IsCanceled"].to_numpy()
    prevalence = float(target.mean())
    positive_count = int(target.sum())

    logistic = joblib.load(ROOT / "models/logistic_regression.joblib")
    forest = joblib.load(ROOT / "models/random_forest.joblib")
    calibrator = joblib.load(
        ROOT / "models/random_forest_sigmoid_calibrator.joblib"
    )

    forest_scores = forest.predict_proba(test[features])[:, 1]
    clipped = np.clip(forest_scores, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)

    scores_by_model = {
        "training_prevalence_baseline": np.full(
            len(test), float(train["IsCanceled"].mean())
        ),
        "logistic_regression": logistic.predict_proba(
            test[features]
        )[:, 1],
        "random_forest_unadjusted": forest_scores,
        "random_forest_sigmoid": calibrator.predict_proba(logits)[:, 1],
    }

    metric_rows = []
    capacity_rows = []

    predictions = test[[
        "booking_id", "hotel_type", "arrival_date", "IsCanceled"
    ]].copy()

    threshold = selection["evaluation_plan"][
        "reference_probability_threshold"
    ]

    for name, scores in scores_by_model.items():
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite scores: {name}")
        if not ((scores >= 0) & (scores <= 1)).all():
            raise ValueError(f"Scores outside [0, 1]: {name}")

        predicted = (scores >= threshold).astype(int)

        metric_rows.append({
            "model": name,
            "records": len(test),
            "positive_rate": prevalence,
            "average_precision": float(
                average_precision_score(target, scores)
            ),
            "roc_auc": float(roc_auc_score(target, scores)),
            "brier_score": float(brier_score_loss(target, scores)),
            "log_loss": float(log_loss(target, scores)),
            "precision_at_0_5": float(
                precision_score(target, predicted, zero_division=0)
            ),
            "recall_at_0_5": float(
                recall_score(target, predicted, zero_division=0)
            ),
        })

        predictions[name] = scores

        # A constant-score baseline has no ranking ability.
        if name == "training_prevalence_baseline":
            continue

        ranked = pd.DataFrame({
            "booking_id": test["booking_id"].to_numpy(),
            "outcome": target,
            "score": scores,
        }).sort_values(
            ["score", "booking_id"],
            ascending=[False, True],
            kind="stable",
        )

        for fraction in selection["evaluation_plan"][
            "review_capacity_fractions"
        ]:
            count = math.ceil(len(test) * fraction)
            found = int(ranked.head(count)["outcome"].sum())
            precision = found / count

            capacity_rows.append({
                "model": name,
                "requested_review_fraction": fraction,
                "reviewed_bookings": count,
                "outcomes_identified": found,
                "precision_at_capacity": precision,
                "recall_at_capacity": found / positive_count,
                "lift_over_random": precision / prevalence,
            })

    metrics = pd.DataFrame(metric_rows)
    capacity = pd.DataFrame(capacity_rows)

    report = {
        "period": "May-August 2017",
        "records": len(test),
        "positive_outcomes": positive_count,
        "selected_model": selection["selected_model"],
        "results": metric_rows,
        "notes": [
            "Frozen artifact fingerprints verified before evaluation.",
            "No model fitting or tuning performed.",
            "Results must not be used to retune the reported models.",
            "Retrospective benchmark; not a live booking-time backtest.",
            "Review capacity applies to the entire test cohort.",
            "Identified outcomes are not prevented cancellations."
        ],
    }

    metrics.to_csv(
        ROOT / "reports/tables/model_comparison_test.csv", index=False
    )
    capacity.to_csv(
        ROOT / "reports/tables/review_capacity_test.csv", index=False
    )
    predictions.to_csv(
        ROOT / "data/processed/test_predictions.csv", index=False
    )
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("Frozen artifact checks: PASS")
    print(f"Final-test bookings: {len(test):,}")
    print(f"Cancellations/no-shows: {positive_count:,}")
    print(f"Outcome rate: {prevalence:.2%}")

    print("\nFINAL-TEST METRICS")
    print(metrics.round(4).to_string(index=False))

    print("\n10% REVIEW CAPACITY")
    print(capacity.loc[
        capacity["requested_review_fraction"].eq(0.10)
    ].round(4).to_string(index=False))

    print(f"\nSaved: {report_path}")
    print("Saved: reports/tables/model_comparison_test.csv")
    print("Saved: reports/tables/review_capacity_test.csv")
    print("Saved: data/processed/test_predictions.csv")


if __name__ == "__main__":
    main()

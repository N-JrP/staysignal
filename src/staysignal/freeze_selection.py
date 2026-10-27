"""Record model selection and artifact fingerprints before final testing."""

from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

from train_initial import NUMERIC, CATEGORICAL


ROOT = Path(__file__).resolve().parents[2]


def main():
    report = json.loads(
        (ROOT / "reports/metrics/calibration_experiment.json")
        .read_text(encoding="utf-8")
    )
    results = {item["model"]: item for item in report["results"]}
    original = results["random_forest_unadjusted"]
    adjusted = results["random_forest_sigmoid"]

    if not (
        adjusted["brier_score"] < original["brier_score"]
        and adjusted["log_loss"] < original["log_loss"]
    ):
        raise ValueError("Calibration selection evidence does not match decision")

    artifact_paths = [
        "models/random_forest.joblib",
        "models/random_forest_sigmoid_calibrator.joblib",
        "models/logistic_regression.joblib",
        "data/processed/split_assignments.csv",
        "docs/split_config.json",
        "requirements.txt",
    ]

    fingerprints = {}
    for relative_path in artifact_paths:
        path = ROOT / relative_path
        fingerprints[relative_path] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    decision = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": "random_forest_with_sigmoid_calibration",
        "target": "Cancellation or no-show: IsCanceled",
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "base_model_training": "Original train split only; no refitting",
        "calibration_training": (
            "January-February validation records with final status before March"
        ),
        "selection_evidence": report["results"],
        "final_test_period": "2017-05-01 to 2017-08-31",
        "final_test_expected_records": 22198,
        "evaluation_plan": {
            "comparators": [
                "training_prevalence_baseline",
                "logistic_regression",
                "random_forest_unadjusted",
                "random_forest_sigmoid",
            ],
            "metrics": [
                "average_precision", "roc_auc",
                "brier_score", "log_loss",
            ],
            "reference_probability_threshold": 0.5,
            "review_capacity_fractions": [0.05, 0.10, 0.20, 0.30],
            "illustrative_review_capacity": 0.10,
            "capacity_scope": "Entire test cohort, not daily workload",
        },
        "artifact_sha256": fingerprints,
        "rules": [
            "Do not tune models or calibration using final-test results.",
            "Report final-test results even if performance deteriorates.",
            "The 0.5 threshold is a reference, not an optimised policy.",
            "Review capacity is illustrative, not a measured hotel requirement.",
            "Historical benchmark; not proof of booking-time deployment.",
            "Cross-hotel robustness work remains a separate experiment."
        ],
    }

    destination = ROOT / "docs/model_selection.json"

    # Avoid silently overwriting an existing frozen decision.
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2)
        handle.write("\n")

    print(f"Saved: {destination}")
    print("Selected: random forest with sigmoid calibration")
    print(f"Artifact fingerprints recorded: {len(fingerprints)}")
    print("Final test remains unscored.")


if __name__ == "__main__":
    main()

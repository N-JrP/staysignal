"""Compare a fixed random forest against the initial validation results."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from train_initial import FEATURES, evaluate


ROOT = Path(__file__).resolve().parents[2]


def main():
    df = pd.read_csv(
        ROOT / "data/processed/bookings_validated.csv"
    )
    assignments = pd.read_csv(
        ROOT / "data/processed/split_assignments.csv"
    )
    df = df.merge(
        assignments,
        on="booking_id",
        how="left",
        validate="one_to_one",
    )

    if df["split"].isna().any():
        raise ValueError("Missing split assignments")

    train = df.loc[df["split"].eq("train")]
    validation = df.loc[df["split"].eq("validation")]

    logistic = joblib.load(
        ROOT / "models/logistic_regression.joblib"
    )

    # Clone resets fitted state; preprocessing is fitted again on training only.
    preprocessing = clone(logistic.named_steps["preprocessing"])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=42,
            n_jobs=2,
        )),
    ])

    print("Training random forest with 200 trees...")
    model.fit(train[FEATURES], train["IsCanceled"])

    result = evaluate(
        "random_forest",
        model,
        validation[FEATURES],
        validation["IsCanceled"],
    )

    initial = json.loads(
        (ROOT / "reports/metrics/initial_validation.json")
        .read_text(encoding="utf-8")
    )

    results = initial["results"] + [result]

    report = {
        "target": "Cancellation or no-show",
        "training_records": len(train),
        "validation_records": len(validation),
        "random_forest_parameters": (
            model.named_steps["classifier"].get_params()
        ),
        "results": results,
        "notes": [
            "One fixed random-forest configuration; no tuning search.",
            "Same feature list and splits as initial models.",
            "All learned preprocessing fitted on training data only.",
            "Final test remains unscored.",
            "Threshold 0.5 is a reference, not a selected operating policy."
        ],
    }

    metrics_path = ROOT / "reports/metrics/model_comparison_validation.json"
    metrics_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    table = pd.DataFrame(results).drop(columns=["confusion_matrix"])
    table.to_csv(
        ROOT / "reports/tables/model_comparison_validation.csv",
        index=False,
    )

    joblib.dump(model, ROOT / "models/random_forest.joblib")

    print("\nVALIDATION COMPARISON")
    print(table[[
        "model", "average_precision", "roc_auc",
        "brier_score", "precision", "recall",
    ]].round(4).to_string(index=False))

    print(
        "\nRandom forest confusion matrix [[TN, FP], [FN, TP]]:",
        result["confusion_matrix"],
    )
    print(f"\nSaved: {metrics_path}")
    print("Saved: models/random_forest.joblib")
    print("Final test remains unscored.")


if __name__ == "__main__":
    main()

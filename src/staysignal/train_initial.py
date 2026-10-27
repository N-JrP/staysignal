"""Train initial models and evaluate on validation data only."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]

NUMERIC = [
    "LeadTime",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
]

CATEGORICAL = [
    "hotel_type",
    "ArrivalDateMonth",
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "CustomerType",
]

FEATURES = NUMERIC + CATEGORICAL


def evaluate(name, model, features, target):
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "model": name,
        "evaluation_split": "validation",
        "records": len(target),
        "positive_rate": float(target.mean()),
        "average_precision": float(
            average_precision_score(target, probabilities)
        ),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "threshold": 0.5,
        "precision": float(
            precision_score(target, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(target, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            target, predictions, labels=[0, 1]
        ).tolist(),
    }


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
        raise ValueError("Some records have no split assignment")

    train = df.loc[df["split"].eq("train")]
    validation = df.loc[df["split"].eq("validation")]

    X_train = train[FEATURES]
    y_train = train["IsCanceled"]
    X_validation = validation[FEATURES]
    y_validation = validation["IsCanceled"]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])

    logistic = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42,
        )),
    ])

    # The baseline uses only the training-set outcome frequency.
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)

    print("Training logistic regression...")
    logistic.fit(X_train, y_train)

    results = [
        evaluate(
            "training_prevalence_baseline",
            baseline,
            X_validation,
            y_validation,
        ),
        evaluate(
            "logistic_regression",
            logistic,
            X_validation,
            y_validation,
        ),
    ]

    metrics_path = ROOT / "reports/metrics/initial_validation.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({
            "target": "Cancellation or no-show",
            "training_records": len(train),
            "numeric_features": NUMERIC,
            "categorical_features": CATEGORICAL,
            "results": results,
            "notes": [
                "Preprocessing fitted on training data only.",
                "Validation used for development evaluation.",
                "Final test data not scored.",
                "Threshold 0.5 is an initial reference, not an operational recommendation.",
                "Retrospective benchmark; booking-time availability not established."
            ],
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    model_path = ROOT / "models/logistic_regression.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic, model_path)

    print(f"\nTraining records: {len(train):,}")
    print(f"Validation records: {len(validation):,}")

    for result in results:
        print(f"\nModel: {result['model']}")
        for metric in (
            "average_precision", "roc_auc",
            "brier_score", "precision", "recall",
        ):
            print(f"  {metric}: {result[metric]:.4f}")
        print(
            "  Confusion matrix [[TN, FP], [FN, TP]]:",
            result["confusion_matrix"],
        )

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {model_path}")
    print("Final test remains unscored.")


if __name__ == "__main__":
    main()

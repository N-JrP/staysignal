"""Diagnose final-test performance using existing frozen predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]

MODELS = [
    "logistic_regression",
    "random_forest_unadjusted",
    "random_forest_sigmoid",
]


def summarise(df, group_columns):
    rows = []

    for group_key, group in df.groupby(
        group_columns, observed=True, dropna=False
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        identifiers = dict(zip(group_columns, group_key))
        target = group["IsCanceled"]
        both_classes = target.nunique() == 2

        for model in MODELS:
            scores = group[model]

            rows.append({
                **identifiers,
                "model": model,
                "bookings": len(group),
                "outcomes": int(target.sum()),
                "outcome_rate": float(target.mean()),
                "mean_predicted_risk": float(scores.mean()),
                "mean_risk_minus_outcome_rate": float(
                    scores.mean() - target.mean()
                ),
                "average_precision": (
                    float(average_precision_score(target, scores))
                    if both_classes else np.nan
                ),
                "roc_auc": (
                    float(roc_auc_score(target, scores))
                    if both_classes else np.nan
                ),
                "brier_score": float(
                    brier_score_loss(target, scores)
                ),
            })

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(
        ROOT / "data/processed/test_predictions.csv",
        parse_dates=["arrival_date"],
    )

    if len(df) != 22198 or not df["booking_id"].is_unique:
        raise ValueError("Unexpected test prediction records")

    if not df["IsCanceled"].isin([0, 1]).all():
        raise ValueError("Invalid outcome labels")

    for model in MODELS:
        scores = df[model].to_numpy()
        if not (
            np.isfinite(scores).all()
            and ((scores >= 0) & (scores <= 1)).all()
        ):
            raise ValueError(f"Invalid scores: {model}")

    df["arrival_month"] = (
        df["arrival_date"].dt.to_period("M").astype(str)
    )

    summaries = {
        "test_performance_by_hotel": ["hotel_type"],
        "test_performance_by_month": ["arrival_month"],
        "test_performance_by_hotel_month": [
            "hotel_type", "arrival_month"
        ],
    }

    for filename, groups in summaries.items():
        table = summarise(df, groups)
        destination = ROOT / f"reports/tables/{filename}.csv"
        table.to_csv(destination, index=False)
        print(f"Saved: {destination}")

        if len(groups) == 1:
            selected = table.loc[
                table["model"].eq("random_forest_sigmoid")
            ]
            print("\nSELECTED MODEL")
            print(selected.drop(
                columns=["model"]
            ).round(4).to_string(index=False))
            print()

    print("Positive mean-risk gap indicates average overprediction.")
    print("Mean-risk agreement alone does not establish calibration.")
    print("Subgroup metrics are descriptive point estimates.")
    print("Models unchanged; final test is not used for tuning.")


if __name__ == "__main__":
    main()

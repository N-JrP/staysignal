"""Compare model rankings at fixed validation review capacities."""

from pathlib import Path
import math

import joblib
import pandas as pd

from train_initial import FEATURES


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

    validation = df.loc[df["split"].eq("validation")].copy()
    total = len(validation)
    positives = int(validation["IsCanceled"].sum())
    prevalence = positives / total

    results = []

    for name in ("logistic_regression", "random_forest"):
        model = joblib.load(ROOT / f"models/{name}.joblib")

        ranked = validation[["booking_id", "IsCanceled"]].copy()
        ranked["risk_score"] = model.predict_proba(
            validation[FEATURES]
        )[:, 1]

        # Reproducible tie-breaking; IDs are not model predictors.
        ranked = ranked.sort_values(
            ["risk_score", "booking_id"],
            ascending=[False, True],
            kind="stable",
        )

        for fraction in (0.05, 0.10, 0.20, 0.30):
            review_count = math.ceil(total * fraction)
            selected = ranked.head(review_count)
            found = int(selected["IsCanceled"].sum())
            precision = found / review_count

            results.append({
                "model": name,
                "split": "validation",
                "requested_review_fraction": fraction,
                "reviewed_bookings": review_count,
                "actual_review_fraction": review_count / total,
                "outcomes_identified": found,
                "reviewed_without_outcome": review_count - found,
                "precision_at_capacity": precision,
                "recall_at_capacity": found / positives,
                "lift_over_random": precision / prevalence,
                "boundary_risk_score": float(
                    selected["risk_score"].iloc[-1]
                ),
            })

    table = pd.DataFrame(results)
    output = ROOT / "reports/tables/review_capacity_validation.csv"
    table.to_csv(output, index=False)

    print(f"Validation bookings: {total:,}")
    print(f"Actual cancellations/no-shows: {positives:,}")
    print(f"Validation outcome rate: {prevalence:.2%}\n")

    print(table[[
        "model",
        "requested_review_fraction",
        "reviewed_bookings",
        "outcomes_identified",
        "precision_at_capacity",
        "recall_at_capacity",
        "lift_over_random",
    ]].round(4).to_string(index=False))

    print(f"\nSaved: {output}")
    print("Capacity applies to the entire validation cohort, not daily workload.")
    print("Outcomes identified are not cancellations prevented.")
    print("Risk scores have not yet been established as calibrated probabilities.")
    print("Final test remains unscored.")


if __name__ == "__main__":
    main()

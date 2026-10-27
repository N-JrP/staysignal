"""Audit repeated predictor patterns without removing records or changing models."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main():
    selection = json.loads(
        (ROOT / "docs/model_selection.json").read_text(encoding="utf-8")
    )
    features = (
        selection["numeric_features"]
        + selection["categorical_features"]
    )

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

    # Stable within this run; includes missing values consistently.
    df["feature_signature"] = pd.util.hash_pandas_object(
        df[features], index=False
    )

    train = df.loc[df["split"].eq("train")]
    train_signatures = set(train["feature_signature"])

    # Stricter comparison adds scheduled arrival date to predictor patterns.
    df["dated_signature"] = pd.util.hash_pandas_object(
        df[features + ["arrival_date"]], index=False
    )
    train_dated_signatures = set(
        df.loc[df["split"].eq("train"), "dated_signature"]
    )

    results = []

    for split_name in ("validation", "test"):
        subset = df.loc[df["split"].eq(split_name)].copy()
        subset["pattern_seen_in_training"] = (
            subset["feature_signature"].isin(train_signatures)
        )
        subset["dated_pattern_seen_in_training"] = (
            subset["dated_signature"].isin(train_dated_signatures)
        )

        results.append({
            "split": split_name,
            "records": len(subset),
            "records_matching_training_predictors": int(
                subset["pattern_seen_in_training"].sum()
            ),
            "fraction_matching_training_predictors": float(
                subset["pattern_seen_in_training"].mean()
            ),
            "records_matching_training_predictors_and_arrival": int(
                subset["dated_pattern_seen_in_training"].sum()
            ),
        })

        if split_name == "test":
            subset[[
                "booking_id",
                "pattern_seen_in_training",
                "dated_pattern_seen_in_training",
            ]].to_csv(
                ROOT / "data/processed/test_pattern_overlap.csv",
                index=False,
            )

    table = pd.DataFrame(results)
    table.to_csv(
        ROOT / "reports/tables/split_pattern_overlap.csv",
        index=False,
    )

    report = {
        "features_compared": features,
        "training_records": len(train),
        "training_unique_predictor_patterns": int(
            train["feature_signature"].nunique()
        ),
        "results": results,
        "limitations": [
            "Matching predictor patterns do not prove duplicate bookings.",
            "Different patterns do not prove different customers.",
            "Anonymous data cannot establish customer-level independence.",
            "Hashes are comparison fingerprints, not customer identifiers.",
            "No records removed; models and frozen selection unchanged."
        ],
    }

    (ROOT / "reports/split_pattern_overlap.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Training records: {len(train):,}")
    print(
        "Unique training predictor patterns:",
        f"{train['feature_signature'].nunique():,}",
    )
    print("\nPATTERN OVERLAP")
    print(table.round(4).to_string(index=False))
    print("\nSaved: reports/tables/split_pattern_overlap.csv")
    print("Saved: reports/split_pattern_overlap.json")
    print("Saved: data/processed/test_pattern_overlap.csv")
    print("No records removed; models unchanged.")


if __name__ == "__main__":
    main()

"""Prepare separate, reconciled CSV sources for Tableau."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main():
    destination = ROOT / "data/processed/tableau"
    destination.mkdir(parents=True, exist_ok=True)

    bookings = pd.read_csv(
        ROOT / "data/processed/bookings_validated.csv",
        dtype={"Agent": "string", "Company": "string"},
        parse_dates=["arrival_date"],
    )
    predictions = pd.read_csv(
        ROOT / "data/processed/test_predictions.csv",
        parse_dates=["arrival_date"],
    )

    if not bookings["booking_id"].is_unique:
        raise ValueError("Booking IDs are not unique")
    if not predictions["booking_id"].is_unique:
        raise ValueError("Prediction IDs are not unique")

    overview_columns = [
        "booking_id", "hotel_type", "arrival_date",
        "MarketSegment", "DistributionChannel", "CustomerType",
        "LeadTime", "total_nights", "total_guests",
        "IsCanceled", "ReservationStatus",
    ] + [
        column for column in bookings.columns
        if column.startswith("flag_")
    ]

    overview = bookings[overview_columns].copy()
    overview["booking_count"] = 1
    overview["cancelled_count"] = (
        overview["ReservationStatus"].eq("Canceled").astype(int)
    )
    overview["no_show_count"] = (
        overview["ReservationStatus"].eq("No-Show").astype(int)
    )
    overview["combined_outcome_count"] = overview["IsCanceled"]
    overview["arrival_month"] = (
        overview["arrival_date"].dt.to_period("M").astype(str)
    )

    # Model data: exactly one row per final-test record.
    context_columns = [
        "booking_id", "MarketSegment",
        "DistributionChannel", "CustomerType", "LeadTime",
    ]
    model_data = predictions.merge(
        bookings[context_columns],
        on="booking_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not model_data["_merge"].eq("both").all():
        raise ValueError("Some test predictions lack booking context")
    model_data = model_data.drop(columns="_merge")

    model_data["selected_risk_score"] = (
        model_data["random_forest_sigmoid"]
    )
    model_data["booking_count"] = 1
    model_data["combined_outcome_count"] = model_data["IsCanceled"]
    model_data["evaluation_split"] = "Final test: May-August 2017"

    if len(overview) != 119390 or len(model_data) != 22198:
        raise ValueError("Unexpected source row counts")
    if int(overview["combined_outcome_count"].sum()) != 44224:
        raise ValueError("Overview outcomes do not reconcile")
    if int(model_data["combined_outcome_count"].sum()) != 9001:
        raise ValueError("Test outcomes do not reconcile")

    overview.to_csv(
        destination / "booking_overview.csv",
        index=False, date_format="%Y-%m-%d",
    )
    model_data.to_csv(
        destination / "model_test_bookings.csv",
        index=False, date_format="%Y-%m-%d",
    )

    supporting_tables = [
        "model_comparison_test",
        "review_capacity_test",
        "test_performance_by_hotel",
        "test_performance_by_month",
        "test_performance_by_hotel_month",
    ]
    for name in supporting_tables:
        table = pd.read_csv(ROOT / f"reports/tables/{name}.csv")
        table.to_csv(destination / f"{name}.csv", index=False)

    manifest = {
        "booking_overview": {
            "grain": "One row per source booking record",
            "rows": len(overview),
            "combined_outcomes": 44224,
        },
        "model_test_bookings": {
            "grain": "One row per final-test booking record",
            "rows": len(model_data),
            "combined_outcomes": 9001,
        },
        "rules": [
            "Use descriptive and model sources separately.",
            "Do not physically join aggregate metric tables to booking rows.",
            "Rates must use SUM(outcome_count) / SUM(booking_count).",
            "Selected risk scores are not established as reliably calibrated probabilities.",
            "Capacity results describe the complete test cohort.",
            "Filtering bookings does not recompute precomputed capacity results.",
            "No occupancy or realised revenue claims.",
            "Files are local; nothing published."
        ],
    }

    (ROOT / "docs/tableau_data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Overview: {len(overview):,} rows; 44,224 combined outcomes")
    print(f"Model test: {len(model_data):,} rows; 9,001 combined outcomes")
    print(f"Supporting tables: {len(supporting_tables)}")
    print("Reconciliation checks: PASS")
    print(f"Tableau CSV folder: {destination}")
    print("Saved: docs/tableau_data_manifest.json")
    print("Nothing published.")


if __name__ == "__main__":
    main()

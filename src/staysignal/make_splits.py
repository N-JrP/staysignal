"""Create chronological benchmark splits with label-availability safeguards."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main():
    df = pd.read_csv(
        ROOT / "data/processed/bookings_validated.csv",
        dtype={"Agent": "string", "Company": "string"},
        parse_dates=["arrival_date", "reservation_status_date"],
    )

    validation_start = pd.Timestamp("2017-01-01")
    test_start = pd.Timestamp("2017-05-01")

    df["split"] = "train"
    df.loc[df["arrival_date"].ge(validation_start), "split"] = "validation"
    df.loc[df["arrival_date"].ge(test_start), "split"] = "test"

    train_pending = (
        df["split"].eq("train")
        & df["reservation_status_date"].ge(validation_start)
    )
    validation_pending = (
        df["split"].eq("validation")
        & df["reservation_status_date"].ge(test_start)
    )

    df.loc[train_pending, "split"] = "held_aside_train_label"
    df.loc[validation_pending, "split"] = "held_aside_validation_label"

    if not df["booking_id"].is_unique:
        raise ValueError("Record IDs must be unique")

    train = df.loc[df["split"].eq("train")]
    validation = df.loc[df["split"].eq("validation")]

    if not train["reservation_status_date"].lt(validation_start).all():
        raise ValueError("Training contains outcomes beyond its cutoff")

    if not validation["reservation_status_date"].lt(test_start).all():
        raise ValueError("Validation contains outcomes beyond its cutoff")

    # Store IDs and assignments only; model code must use explicit feature lists.
    assignments = df[["booking_id", "split"]]
    destination = ROOT / "data/processed/split_assignments.csv"
    assignments.to_csv(destination, index=False)

    counts = {
        str(name): int(count)
        for name, count in df["split"].value_counts().items()
    }

    config = {
        "validation_start": "2017-01-01",
        "test_start": "2017-05-01",
        "split_basis": "Scheduled arrival date",
        "label_availability_proxy": "Final reservation status date",
        "counts": counts,
        "notes": [
            "All records receive exactly one assignment.",
            "Late-finalised development labels are held aside, not deleted.",
            "Final test is reserved for evaluation after model selection.",
            "No anomaly-based exclusions applied.",
            "Feature availability remains limited by historical snapshots.",
            "Arrival-date splitting is a retrospective benchmark, not a live backtest."
        ],
    }

    config_path = ROOT / "docs/split_config.json"
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SPLIT COUNTS")
    for name, count in sorted(counts.items()):
        print(f"{name}: {count:,}")

    print(f"\nTotal assigned: {len(assignments):,}")
    print("Unique IDs:", assignments["booking_id"].is_unique)
    print("Label-cutoff checks: PASS")
    print(f"Saved: {destination}")
    print(f"Saved: {config_path}")


if __name__ == "__main__":
    main()

"""Validate dates and flag anomalies without deleting records."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MONTHS = {
    "January": 1, "February": 2, "March": 3,
    "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9,
    "October": 10, "November": 11, "December": 12,
}


def main():
    source = ROOT / "data/processed/bookings_normalised.csv"
    df = pd.read_csv(
        source,
        dtype={"Agent": "string", "Company": "string"},
    )

    df["arrival_date"] = pd.to_datetime(
        pd.DataFrame({
            "year": df["ArrivalDateYear"],
            "month": df["ArrivalDateMonth"].map(MONTHS),
            "day": df["ArrivalDateDayOfMonth"],
        }),
        errors="coerce",
    )

    df["reservation_status_date"] = pd.to_datetime(
        df["ReservationStatusDate"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    df["total_nights"] = (
        df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    )

    # Preserve unknown guest totals instead of silently treating missing as zero.
    df["total_guests"] = df[
        ["Adults", "Children", "Babies"]
    ].sum(axis=1, min_count=3)

    nonnegative_fields = [
        "LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
        "Adults", "Children", "Babies", "PreviousCancellations",
        "PreviousBookingsNotCanceled", "BookingChanges",
        "DaysInWaitingList", "RequiredCarParkingSpaces",
        "TotalOfSpecialRequests",
    ]

    flags = {
        "invalid_arrival_date": df["arrival_date"].isna(),
        "invalid_status_date": df["reservation_status_date"].isna(),
        "zero_night_stay": df["total_nights"].eq(0),
        "zero_total_guests": df["total_guests"].eq(0),
        "unknown_total_guests": df["total_guests"].isna(),
        "negative_adr": df["ADR"].lt(0),
        "zero_adr": df["ADR"].eq(0),
        "negative_count_or_duration": df[nonnegative_fields].lt(0).any(axis=1),
        "noncancelled_status_before_arrival": (
            df["IsCanceled"].eq(0)
            & df["reservation_status_date"].lt(df["arrival_date"])
        ),
    }

    flag_columns = []
    for name, values in flags.items():
        column = f"flag_{name}"
        df[column] = values.fillna(False)
        flag_columns.append(column)

    summary = {
        "rows": len(df),
        "arrival_min": str(df["arrival_date"].min().date()),
        "arrival_max": str(df["arrival_date"].max().date()),
        "reservation_status_counts": {
            str(key): int(value)
            for key, value in df["ReservationStatus"].value_counts(
                dropna=False
            ).items()
        },
        "flag_counts": {
            name: int(df[f"flag_{name}"].sum())
            for name in flags
        },
        "adr_quantiles": {
            str(key): float(value)
            for key, value in df["ADR"].quantile(
                [0, 0.01, 0.5, 0.99, 1]
            ).items()
        },
        "notes": [
            "Flags identify records for investigation, not automatic deletion.",
            "Zero rates and zero-night stays may have legitimate explanations.",
            "Status dates are outcome information, not model predictors.",
            "No records removed and no missing values imputed."
        ],
    }

    df.to_csv(
        ROOT / "data/processed/bookings_validated.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    flagged = df.loc[df[flag_columns].any(axis=1)]
    flagged.head(100).to_csv(
        ROOT / "reports/anomaly_examples.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    (ROOT / "reports/validation_report.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("\nSaved: data/processed/bookings_validated.csv")
    print("Saved: reports/validation_report.json")
    print("Saved: reports/anomaly_examples.csv (first 100 flagged records)")


if __name__ == "__main__":
    main()

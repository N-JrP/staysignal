"""Export descriptive booking summaries with explicit outcome definitions."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def summarise(df, group_columns):
    result = df.groupby(
        group_columns,
        dropna=False,
        observed=True,
    ).agg(
        bookings=("booking_id", "size"),
        cancellations=("cancelled", "sum"),
        no_shows=("no_show", "sum"),
        cancellations_or_no_shows=("IsCanceled", "sum"),
        median_lead_time=("LeadTime", "median"),
        median_total_nights=("total_nights", "median"),
    ).reset_index()

    for numerator, rate_name in (
        ("cancellations", "cancellation_rate"),
        ("no_shows", "no_show_rate"),
        ("cancellations_or_no_shows", "combined_outcome_rate"),
    ):
        result[rate_name] = result[numerator] / result["bookings"]

    return result


def main():
    df = pd.read_csv(
        ROOT / "data/processed/bookings_validated.csv",
        dtype={"Agent": "string", "Company": "string"},
        parse_dates=["arrival_date"],
    )

    df["cancelled"] = df["ReservationStatus"].eq("Canceled")
    df["no_show"] = df["ReservationStatus"].eq("No-Show")
    df["arrival_month"] = (
        df["arrival_date"].dt.to_period("M").astype(str)
    )

    if not df["IsCanceled"].eq(
        (df["cancelled"] | df["no_show"]).astype(int)
    ).all():
        raise ValueError("Outcome labels do not match final status")

    df["lead_time_band"] = pd.cut(
        df["LeadTime"],
        bins=[-1, 7, 30, 90, 180, float("inf")],
        labels=["0–7", "8–30", "31–90", "91–180", "181+"],
    )

    exports = {
        "hotel_summary": ["hotel_type"],
        "monthly_summary": ["arrival_month", "hotel_type"],
        "channel_summary": ["hotel_type", "DistributionChannel"],
        "segment_summary": ["hotel_type", "MarketSegment"],
        "lead_time_summary": ["hotel_type", "lead_time_band"],
    }

    destination = ROOT / "reports" / "tables"
    destination.mkdir(parents=True, exist_ok=True)

    for name, groups in exports.items():
        result = summarise(df, groups)

        if int(result["bookings"].sum()) != len(df):
            raise ValueError(f"Booking totals do not reconcile: {name}")

        if int(result["cancellations_or_no_shows"].sum()) != int(
            df["IsCanceled"].sum()
        ):
            raise ValueError(f"Outcome totals do not reconcile: {name}")

        result.to_csv(destination / f"{name}.csv", index=False)
        print(f"Saved {name}.csv: {len(result)} groups")

        if name == "hotel_summary":
            print(result.to_string(index=False))

    print("\nRates are proportions from 0 to 1.")
    print("All records retained; monthly groups follow scheduled arrival dates.")
    print("July 2015 and August 2017 are dataset boundaries.")
    print("Summaries describe bookings, not occupancy or realised revenue.")


if __name__ == "__main__":
    main()

"""Create descriptive figures from reconciled summary tables."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main():
    tables = ROOT / "reports" / "tables"
    output = ROOT / "reports" / "figures"
    output.mkdir(parents=True, exist_ok=True)

    hotels = pd.read_csv(tables / "hotel_summary.csv")
    monthly = pd.read_csv(tables / "monthly_summary.csv")
    monthly["arrival_month"] = pd.to_datetime(
        monthly["arrival_month"], format="%Y-%m"
    )

    colors = {
        "City Hotel": "#2563EB",
        "Resort Hotel": "#0F766E",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "StaySignal | Historical Hotel Booking Patterns",
        fontsize=17,
        fontweight="bold",
    )

    for hotel, group in monthly.groupby("hotel_type"):
        group = group.sort_values("arrival_month")
        axes[0].plot(
            group["arrival_month"],
            group["bookings"],
            label=hotel,
            color=colors[hotel],
            linewidth=2,
        )

    axes[0].set_title("Bookings by scheduled arrival month")
    axes[0].set_ylabel("Number of bookings")
    axes[0].set_xlabel("Scheduled arrival month")
    axes[0].legend(frameon=False)
    axes[0].tick_params(axis="x", rotation=30)

    positions = list(range(len(hotels)))
    axes[1].bar(
        positions,
        hotels["cancellation_rate"],
        label="Cancelled",
        color="#2563EB",
        width=0.55,
    )
    axes[1].bar(
        positions,
        hotels["no_show_rate"],
        bottom=hotels["cancellation_rate"],
        label="No-show",
        color="#F59E0B",
        width=0.55,
    )

    axes[1].set_xticks(positions, hotels["hotel_type"])
    axes[1].set_title("Cancellation and no-show rates")
    axes[1].set_ylabel("Share of all bookings within each hotel")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylim(0, 0.55)
    axes[1].legend(frameon=False)

    for position, rate in zip(
        positions, hotels["combined_outcome_rate"]
    ):
        axes[1].text(
            position,
            rate + 0.012,
            f"{rate:.1%} combined",
            ha="center",
            fontsize=10,
        )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)

    fig.text(
        0.02,
        0.025,
        "Source: Antonio et al., Hotel Booking Demand Datasets. "
        "Portugal, July 2015–August 2017.\n"
        "Historical booking records; volumes are not occupancy. "
        "Differences are descriptive, not causal.",
        fontsize=9,
        color="#475569",
    )

    fig.tight_layout(rect=[0, 0.12, 1, 0.91])

    destination = output / "booking_overview.png"
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)

    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()

"""Inspect validation calibration without fitting or changing models."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {
        "logistic_regression": "#2563EB",
        "random_forest": "#0F766E",
    }
    tables = []

    axes[0].plot(
        [0, 1], [0, 1],
        linestyle="--", color="#64748B",
        label="Perfect calibration",
    )

    for name, color in colors.items():
        model = joblib.load(ROOT / f"models/{name}.joblib")
        scores = model.predict_proba(validation[FEATURES])[:, 1]

        records = pd.DataFrame({
            "score": scores,
            "outcome": validation["IsCanceled"].to_numpy(),
        })

        # Ten fixed-width bins; empty bins are omitted from the curve.
        records["risk_bin"] = pd.cut(
            records["score"],
            bins=[i / 10 for i in range(11)],
            include_lowest=True,
        )

        summary = records.groupby(
            "risk_bin", observed=True
        ).agg(
            bookings=("outcome", "size"),
            mean_predicted_risk=("score", "mean"),
            observed_outcome_rate=("outcome", "mean"),
        ).reset_index()

        summary["calibration_gap"] = (
            summary["observed_outcome_rate"]
            - summary["mean_predicted_risk"]
        )
        summary.insert(0, "model", name)
        summary["risk_bin"] = summary["risk_bin"].astype(str)
        tables.append(summary)

        label = name.replace("_", " ").title()
        axes[0].plot(
            summary["mean_predicted_risk"],
            summary["observed_outcome_rate"],
            marker="o", color=color, label=label,
        )
        axes[1].hist(
            scores,
            bins=[i / 10 for i in range(11)],
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
        )

        print(f"\n{label}")
        print(summary[[
            "risk_bin", "bookings",
            "mean_predicted_risk",
            "observed_outcome_rate",
            "calibration_gap",
        ]].round(4).to_string(index=False))

    axes[0].set(
        title="Validation calibration",
        xlabel="Mean predicted risk",
        ylabel="Observed cancellation-or-no-show rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))

    axes[1].set(
        title="Validation risk-score distribution",
        xlabel="Predicted risk",
        ylabel="Number of bookings",
        xlim=(0, 1),
    )
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))

    for axis in axes:
        axis.legend(frameon=False, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)

    fig.suptitle("StaySignal | Do Risk Scores Match Outcomes?")
    fig.text(
        0.02, 0.02,
        "Historical validation cohort only. Bin estimates have sampling "
        "uncertainty; small bins require caution.\n"
        "Models unchanged. Final test remains unscored.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])

    figure_path = ROOT / "reports/figures/calibration_validation.png"
    fig.savefig(figure_path, dpi=180, facecolor="white")
    plt.close(fig)

    table_path = ROOT / "reports/tables/calibration_validation.csv"
    pd.concat(tables, ignore_index=True).to_csv(table_path, index=False)

    print(f"\nSaved: {figure_path}")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()

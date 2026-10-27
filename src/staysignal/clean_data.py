"""Normalise original hotel data while preserving all source records."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main():
    frames = []
    file_reports = []

    for filename, hotel_type, expected_rows in (
        ("H1.csv", "Resort Hotel", 40060),
        ("H2.csv", "City Hotel", 79330),
    ):
        source = ROOT / "data" / "raw" / filename

        # Preserve identifiers as text, including any leading zeros.
        df = pd.read_csv(
            source,
            dtype={"Agent": "string", "Company": "string"},
        )

        if len(df) != expected_rows:
            raise ValueError(f"Unexpected row count in {filename}")

        text_changes = {}

        for column in df.select_dtypes(include=["object", "string"]).columns:
            original = df[column].astype("string")
            cleaned = original.str.strip()

            padded_count = int(
                original.ne(cleaned).fillna(False).sum()
            )
            missing_marker = (
                cleaned.eq("")
                | cleaned.str.upper().eq("NULL")
            ).fillna(False)

            df[column] = cleaned.mask(missing_marker, pd.NA)

            if padded_count or missing_marker.any():
                text_changes[column] = {
                    "trimmed_values": padded_count,
                    "missing_markers_normalised": int(missing_marker.sum()),
                }

        if not df["IsCanceled"].isin([0, 1]).all():
            raise ValueError(f"Invalid cancellation labels in {filename}")

        source_rows = pd.Series(range(1, len(df) + 1), index=df.index)

        df.insert(
            0,
            "booking_id",
            filename.removesuffix(".csv") + "-" +
            source_rows.astype(str).str.zfill(6),
        )
        df.insert(1, "hotel_type", hotel_type)
        df.insert(2, "source_file", filename)
        df.insert(3, "source_row", source_rows)

        frames.append(df)
        file_reports.append({
            "source_file": filename,
            "rows_retained": len(df),
            "text_changes": text_changes,
        })

    combined = pd.concat(frames, ignore_index=True)

    if len(combined) != 119390:
        raise ValueError("Combined row count does not match source data")
    if not combined["booking_id"].is_unique:
        raise ValueError("Generated booking IDs are not unique")

    destination = ROOT / "data" / "processed" / "bookings_normalised.csv"
    combined.to_csv(destination, index=False, encoding="utf-8")

    report = {
        "rows": len(combined),
        "columns": len(combined.columns),
        "files": file_reports,
        "missing_values": {
            column: int(count)
            for column, count in combined.isna().sum().items()
            if count
        },
        "decisions": [
            "All source records retained, including duplicate-looking rows.",
            "Whitespace trimmed and blank/NULL markers normalised.",
            "Agent and Company identifiers preserved as text.",
            "Missing values not imputed.",
            "Dates and numerical anomalies not yet validated.",
            "Generated booking IDs identify source rows, not real-world bookings.",
            "Outcome fields retained for analysis; not approved as model features."
        ],
    }

    report_path = ROOT / "reports" / "normalisation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Rows retained: {len(combined):,}")
    print(f"Columns: {len(combined.columns)}")
    print(f"Unique record IDs: {combined['booking_id'].is_unique}")
    print("Missing values:", report["missing_values"])
    print(f"Saved data: {destination}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()

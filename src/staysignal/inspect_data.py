"""Inspect original hotel CSV files without modifying them."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def inspect_file(filename):
    path = ROOT / "data" / "raw" / filename
    df = pd.read_csv(path)

    text_checks = {}

    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].astype("string")
        stripped = values.str.strip()

        text_checks[column] = {
            "padded_values": int(
                values.ne(stripped).fillna(False).sum()
            ),
            "literal_null_values": int(
                stripped.str.upper().eq("NULL").fillna(False).sum()
            ),
            "blank_values": int(
                stripped.eq("").fillna(False).sum()
            ),
        }

    return {
        "filename": filename,
        "rows": len(df),
        "columns": len(df.columns),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "pandas_missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "text_checks": text_checks,
        "duplicate_looking_rows_after_first": int(
            df.duplicated().sum()
        ),
        "cancellation_label_counts": {
            str(label): int(count)
            for label, count in
            df["IsCanceled"].value_counts(dropna=False).items()
        },
        "notes": [
            "No records modified or removed.",
            "Literal NULL and blank counts are separate from pandas missing values.",
            "Duplicate-looking rows are not proven duplicate bookings."
        ],
    }


def main():
    reports = [
        inspect_file(filename)
        for filename in ("H1.csv", "H2.csv")
    ]

    output = ROOT / "reports" / "data_inspection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(reports, indent=2) + "\n",
        encoding="utf-8",
    )

    for report in reports:
        print(f"\n{report['filename']}: {report['rows']:,} rows")
        print("Cancellation labels:", report["cancellation_label_counts"])
        print(
            "Duplicate-looking rows:",
            report["duplicate_looking_rows_after_first"],
        )
        print("Columns with pandas missing values:")
        print({
            column: count
            for column, count in report["pandas_missing_values"].items()
            if count
        })
        print("Text columns needing attention:")
        print({
            column: checks
            for column, checks in report["text_checks"].items()
            if any(checks.values())
        })

    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()

# StaySignal
### Hotel booking analytics and cancellation-risk modelling

StaySignal explores historical hotel booking patterns and evaluates whether machine learning can prioritise bookings associated with cancellation or no-show outcomes.

Built with Python, pandas, scikit-learn and Tableau, the project covers data preparation, quality checks, model evaluation and visual communication using 119,390 real booking records from two Portuguese hotels.

## Project at a glance

- **Data:** 119,390 bookings, July 2015–August 2017.
- **Analysis:** Booking trends, hotel differences, cancellations and no-shows.
- **Models:** Training-prevalence baseline, logistic regression and random forest.
- **Evaluation:** Chronological splits, label-cutoff checks, calibration assessment and capacity-based ranking.
- **Deliverables:** Python scripts, documented decisions, evaluation reports and two Tableau dashboards.
- **Scope:** Historical retrospective benchmark; not a live hotel deployment.

## Hospitality use case

A hotel analytics team may have limited capacity to review bookings. StaySignal evaluates whether model scores can concentrate cancellation and no-show outcomes within a smaller review queue.

The review example uses the entire historical test cohort. It does not simulate daily operations or establish the effectiveness of contacting guests.

## Dashboards

### Hotel Booking Overview

Booking volumes, cancellation and no-show rates, monthly trends and hotel comparisons. Cancellation and no-show rates are reported separately.

![Hotel Booking Overview](docs/images/hotel_booking_overview.png)

### Model Performance

Comparison of ranking performance and probability prediction error on the final temporal test.

![Model Performance](docs/images/model_performance.png)

### Review Capacity

Trade-off between the proportion of bookings reviewed and the share of cancellation/no-show outcomes identified.

![Review Capacity](docs/images/review_capacity.png)

Open [the packaged Tableau workbook](dashboards/StaySignal.twbx) in Tableau Public or a compatible Tableau Desktop version. It contains the dashboard data extracts.

## Final-test results

The final test contains **22,198 bookings** scheduled to arrive during **May–August 2017**, including **9,001 cancellations or no-shows**.

| Model | Average precision ↑ | ROC-AUC ↑ | Brier score ↓ | Log loss ↓ |
|---|---:|---:|---:|---:|
| Training-prevalence baseline | 0.4055 | 0.5000 | 0.2428 | 0.6789 |
| Logistic regression | 0.6281 | 0.6984 | 0.2128 | 0.6129 |
| Random forest | 0.7082 | 0.7564 | 0.1909 | 0.5590 |
| Random forest + sigmoid calibration | 0.7082 | 0.7564 | 0.1930 | 0.5617 |

The calibrated random forest was selected before final-test scoring. Calibration improved development assessment results but slightly worsened final-test Brier score and log loss. The frozen selection was retained, and both forest variants are reported transparently.

### Illustrative 10% review capacity

For the selected model:

| Measure | Result |
|---|---:|
| Bookings reviewed | 2,220 |
| Cancellation/no-show outcomes identified | 1,986 |
| Precision among reviewed bookings | 89.46% |
| Share of all outcomes identified | 22.06% |
| Lift over random selection | 2.21× |

These are outcomes identified retrospectively, not cancellations prevented or revenue saved.

## Data preparation and evaluation

- Recorded source-file fingerprints and dataset provenance.
- Normalised padded text and missing-value representations.
- Retained duplicate-looking records because identical anonymous rows do not establish duplicate bookings.
- Flagged unusual records rather than automatically deleting them.
- Documented feature decisions and excluded direct outcome fields from predictors.
- Used chronological arrival-date splits and held aside development records with late final-status dates.
- Compared models against a training-prevalence baseline.
- Assessed calibration on a later development period.
- Froze model selection and artifact fingerprints before final-test scoring.
- Examined final-test performance by hotel and month.
- Audited predictor-pattern overlap across splits.

Arrival-date splitting and label-cutoff checks do not establish booking-time feature availability.

## Technology

| Area | Tools |
|---|---|
| Data preparation and analysis | Python 3.11, pandas, NumPy |
| Machine learning | scikit-learn |
| Model persistence | joblib |
| Python visualisation | Matplotlib, seaborn |
| Business intelligence | Tableau Public |
| Development and version control | VS Code, Git, GitHub |

The workflow does not require paid APIs or cloud services.

## Repository guide

| Path | Contents |
|---|---|
| `src/staysignal/` | Data preparation, analysis, modelling and evaluation scripts |
| `docs/` | Data provenance, feature decisions, split configuration and findings |
| `docs/images/` | Dashboard screenshots |
| `reports/` | Generated analysis and evaluation outputs |
| `dashboards/StaySignal.twbx` | Packaged Tableau workbook |
| `requirements.txt` | Recorded Python environment dependencies |

Key scripts include:

- `inspect_data.py`, `clean_data.py`, `validate_data.py`: data preparation and quality checks.
- `analyse_data.py`, `plot_overview.py`: descriptive analysis.
- `make_splits.py`: historical split assignments.
- `train_initial.py`, `train_forest.py`: baseline and model training.
- `evaluate_capacity.py`, `check_calibration.py`, `calibrate_forest.py`: development evaluation.
- `freeze_selection.py`, `evaluate_final_test.py`: frozen selection and final evaluation.
- `evaluate_segments.py`, `audit_split_overlap.py`: diagnostic analysis.
- `export_tableau.py`: dashboard data exports.

## Local environment setup

Run from the repository root using Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The recorded environment was used on Windows with Python 3.11.7. Cross-platform installation has not been verified.

The Tableau workbook can be reviewed independently of Python model training.

## Data source

António, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49.

- DOI: https://doi.org/10.1016/j.dib.2018.11.126
- Article and supplementary files: https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/
- Original CSV files: `H1.csv` and `H2.csv`, supplied in supplementary archive `mmc2.zip`.
- Source fingerprints: [data source manifest](docs/data_source_manifest.json).

## Limitations

- Historical data from two Portuguese hotels; results do not establish performance for current German hotels.
- A complete booking-time snapshot is unavailable, so candidate features are not approved for live prediction.
- The target combines cancellations and no-shows.
- Anonymous records prevent reliable customer-level grouping.
- The selected scores are not established as reliably calibrated probabilities across future periods.
- Metrics are point estimates without confidence intervals.
- No live PMS integration, intervention effectiveness, revenue savings or production performance has been demonstrated.

See the [feature audit](docs/feature_audit.md) and [final-test findings](docs/final_test_findings.md) for further methodological detail.

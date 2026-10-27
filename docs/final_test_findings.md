# StaySignal — Final Temporal Benchmark Findings

## Evaluation setup

- Final-test period: May–August 2017.
- Records: 22,198.
- Cancellations or no-shows: 9,001.
- Observed combined outcome rate: 40.55%.
- Model artifacts and evaluation decisions frozen before test scoring.
- No model retraining or tuning performed on the test cohort.

## Model comparison

| Model | Average precision | ROC-AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| Training-prevalence baseline | 0.4055 | 0.5000 | 0.2428 | 0.6789 |
| Logistic regression | 0.6281 | 0.6984 | 0.2128 | 0.6129 |
| Random forest, unadjusted | 0.7082 | 0.7564 | 0.1909 | 0.5590 |
| Random forest, sigmoid calibration | 0.7082 | 0.7564 | 0.1930 | 0.5617 |

Average precision and ROC-AUC: higher is better.
Brier score and log loss: lower is better.

## Review-capacity example

For the frozen selected model, reviewing the highest-risk 10% means:

- Bookings reviewed: 2,220.
- Actual cancellations/no-shows identified: 1,986.
- Reviewed bookings without that outcome: 234.
- Precision at capacity: 89.46%.
- Recall at capacity: 22.06%.
- Lift over random selection: 2.21 times.

This is retrospective ranking across the entire test cohort.
It is not a daily workload simulation.
Identified outcomes are not prevented cancellations.

## Calibration result

Sigmoid calibration improved Brier score and log loss on the
March–April development assessment, but slightly worsened both
metrics on the May–August final test.

The frozen selection remains the calibrated random forest.
The unadjusted model is reported as a predeclared comparator.
No post-test model switching or recalibration is performed.

The increasing sigmoid adjustment preserves ranking, explaining
identical average precision, ROC-AUC and top-capacity results.

Do not describe the selected scores as reliably calibrated
probabilities across future periods.

## Generalisation

Random-forest average precision declined from 0.7396 on the full
validation cohort to 0.7082 on the final test. ROC-AUC declined
from 0.7999 to 0.7564.

These are different historical cohorts, so the change is descriptive.
The cause has not been established.
Average precision also depends on outcome prevalence.

## Business interpretation

The benchmark demonstrates useful prioritisation relative to random
selection, while leaving most outcomes unidentified at 10% capacity.

The 10% review capacity is an illustrative assumption, not a measured
hotel staffing requirement.

No revenue savings, successful interventions, or live-system impact
have been demonstrated.

## Scope limitations

- Historical records from two Portuguese hotels.
- Booking-time feature availability is not established.
- Arrival-date splits constitute a retrospective benchmark.
- Anonymous records limit customer-level leakage checks.
- Duplicate-looking records retained; split overlap requires further assessment.
- Reported metrics are point estimates without confidence intervals.
- Cross-hotel transfer and segment-level reliability remain to be assessed.

## Next analysis

Inspect performance by hotel and arrival month using existing frozen
predictions. This is diagnostic analysis, not permission to tune models
on the final test.

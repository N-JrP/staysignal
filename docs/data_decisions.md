# StaySignal — Data Decisions

## Dataset and record retention

- Source: Hotel Booking Demand Datasets, DOI: 10.1016/j.dib.2018.11.126.
- Records: 119,390 across two Portuguese hotels.
- Arrival period: July 2015–August 2017.
- Raw files remain unchanged.
- All records remain in the normalised and validated datasets.
- Generated booking_id values identify source rows, not verified unique real-world bookings.
- Duplicate-looking records are retained because identical attributes do not prove duplicate bookings.

## Target definition

The primary classification target is IsCanceled:

- 0: Check-Out — 75,166 records.
- 1: Canceled or No-Show — 44,224 records.
  - Canceled: 43,017.
  - No-Show: 1,207.

Dashboard wording must distinguish:
- Cancellation rate: ReservationStatus equals Canceled.
- No-show rate: ReservationStatus equals No-Show.
- Combined cancellation-or-no-show rate: IsCanceled equals 1.

Use the combined outcome for the primary ML benchmark.
These records do not establish why a booking failed to materialise.

## Normalisation already performed

- Trimmed surrounding whitespace from text fields.
- Converted literal NULL and blank text markers to missing values.
- Preserved Agent and Company as text identifiers.
- Retained missing values without imputation.
- Added source-file and source-row identifiers.
- Parsed arrival and final-status dates.
- Calculated total_nights and total_guests.
- Kept total_guests missing when a component is missing.

## Anomalies and treatment

- Zero-night bookings: 715.
- Zero-total-guest bookings: 180.
- Unknown-total-guest bookings: 4.
- Negative ADR: 1 record, booking H1-014970, ADR -6.38.
- Zero ADR: 1,959.
- Maximum ADR: 5,400, booking H2-008456.
- Zero-night and zero-guest overlap: 70 records.

Do not silently delete or overwrite these records.
Zero prices and zero-night records may have valid operational explanations.
The negative and extreme ADR values require separate visibility.

For descriptive booking counts:
- Include all records and provide quality flags.
- Explain any filtered views and their denominators.

For rate-related analysis:
- Report negative, zero and extreme values separately.
- Do not replace the maximum ADR merely because it looks unusual.
- Use robust summaries such as median and percentiles.
- Any later exclusion or sensitivity analysis must be documented.

Before modelling:
- Specify the eligible population and compare excluded versus retained records.
- Decide exclusions using domain validity, not improved model scores.
- Fit imputation or other learned transformations on training data only.

## Financial and occupancy claims

- ADR multiplied by nights is an estimated booking-value proxy, not measured revenue loss.
- Do not claim prevented cancellations or realised savings.
- Do not calculate occupancy without room-capacity information.
- Do not infer agency commissions from booking-channel categories.

## Prediction-time limitations

The data do not provide a complete booking-time snapshot history.
Treat the initial model as a retrospective benchmark until feature availability is justified.

Never use these as predictors:
- IsCanceled: target.
- ReservationStatus: outcome.
- ReservationStatusDate: outcome timing.
- reservation_status_date: parsed outcome timing.
- booking_id, source_file, source_row: lineage identifiers.
- Flags derived from final status or outcome dates.

Audit other potentially updated fields before approving them.
Chronological splitting does not itself eliminate feature leakage.

## Generalisation

Results concern historical records from two hotels.
They do not establish current German hotel behaviour or production readiness.

# StaySignal — Feature Availability Audit

## Modelling scope

Initial scope: retrospective cancellation-or-no-show classification benchmark.

The dataset does not establish a complete booking-time snapshot for every field.
No feature is approved for live booking-time deployment.
"Candidate" below means eligible for further investigation in the historical benchmark.

## Field decisions

| Field | Decision | Reason |
|---|---|---|
| IsCanceled | Target only | Includes cancellations and no-shows |
| ReservationStatus | Exclude | Directly reveals the outcome |
| ReservationStatusDate | Exclude | Final-status timing |
| reservation_status_date | Exclude | Parsed final-status timing |
| booking_id | Exclude | Generated row identifier |
| source_file | Exclude | Lineage identifier |
| source_row | Exclude | Row order could encode collection or outcome patterns |
| hotel_type | Candidate | Known hotel context; also use for evaluation |
| LeadTime | Candidate | Booking-to-arrival interval; changes to arrival date may affect it |
| ArrivalDateYear | Split/reporting only initially | Defines historical period; avoid relying on year trend |
| ArrivalDateMonth | Candidate | Seasonal context |
| ArrivalDateWeekNumber | Exclude initially | Redundant with other calendar features |
| ArrivalDateDayOfMonth | Candidate | Calendar context |
| arrival_date | Split/reporting only | Derive approved calendar features explicitly |
| StaysInWeekendNights | Candidate | Stay details; potentially amended |
| StaysInWeekNights | Candidate | Stay details; potentially amended |
| total_nights | Exclude initially | Redundant with component night counts |
| Adults | Candidate | Guest composition; potentially amended |
| Children | Candidate | Guest composition; preserve unknown values |
| Babies | Candidate | Guest composition |
| total_guests | Exclude initially | Redundant with component guest counts |
| Meal | Candidate | Package information; potentially amended |
| Country | Exclude initially | Geographic proxy; unnecessary for initial model |
| MarketSegment | Candidate | Booking segment; historical snapshot caveat |
| DistributionChannel | Candidate | Booking channel; historical snapshot caveat |
| IsRepeatedGuest | Candidate | Prior relationship indicator |
| PreviousCancellations | Exclude initially | Audit whether previous outcomes were known at prediction time |
| PreviousBookingsNotCanceled | Exclude initially | Audit timing of previous outcomes |
| ReservedRoomType | Candidate | Reserved product; potentially amended |
| AssignedRoomType | Exclude initially | Operational assignment may occur later |
| BookingChanges | Exclude initially | Accumulates after booking creation |
| DepositType | Exclude initially | Derived from payments before arrival/cancellation |
| Agent | Exclude initially | High-cardinality identifier; not needed for initial model |
| Company | Exclude initially | High missingness and identifier |
| DaysInWaitingList | Exclude initially | Accumulated waiting time |
| CustomerType | Candidate | Booking category; historical snapshot caveat |
| ADR | Exclude initially | Transaction-derived rate; availability and anomalies require investigation |
| RequiredCarParkingSpaces | Exclude initially | Requests may be added later |
| TotalOfSpecialRequests | Exclude initially | Requests accumulate over time |
| flag_* | Exclude | Quality/reporting fields; some depend on outcomes |

## Rules

- Candidates are not a claim of booking-time availability.
- Use explicit feature allowlists; never automatically train on every column.
- Missing-value handling and categorical encoding must be fitted on training data.
- Keep the final temporal test period untouched during model selection.
- Investigate label availability at training cutoffs.
- Anonymous records prevent reliable customer-level grouping; document this limitation.
- Cross-hotel evaluation must exclude hotel_type as a predictive feature.
- Feature importance indicates association, not causation.
- Any change to the allowlist needs a documented reason.

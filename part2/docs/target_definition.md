# Target Definition

Prepared for Part 1 run `82927c54-85b2-4dc1-aee9-dba8a32815ca`.

## Three Clocks

Three independent time windows govern eligibility, labeling, and service continuity:

| Clock | Definition | Value in this study |
|---|---|---|
| **Prediction horizon** | Contracts with `term_end` within this window of `as_of_date` are eligible candidates. | 90 days: `days_to_renewal ≤ 90` at each scoring snapshot |
| **Label maturity** | How long after expiry before a label is declared. An outcome is unknown (not assumed retained) until this window closes. | 90 days: `term_end + 90d ≤ obs_end (2026-10-01)` → `label_matured = True` |
| **Entitlement grace** | How long service may continue after contract expiry. Applied to support ticket contract attribution, not to label generation. | 30 days: `renewal_grace_period_days` in `config/simulation.yaml` |

These clocks are independent. A contract can be within the prediction horizon but have an
immature label (excluded from supervised evaluation). A cancellation event within the
entitlement grace period is still recorded as a churn event; the grace period only affects
support ticket attribution, not the churn label.

**Maturity lag note:** a contract expiring 90 days after scoring has its label available
approximately 180 days after scoring (90-day horizon + 90-day maturity window).

## Label: `churned`

A renewal unit (one contract row at a given snapshot) is labeled **churned = 1** if:
- `label_matured = True` (term_end + 90 days ≤ observation_end 2026-10-01)
- A cancellation event is found in `contract_events.csv` joined on `contract_id`

A renewal unit is labeled **churned = 0** if:
- `label_matured = True`
- No cancellation event is found for that `contract_id`

**A mature zero means no recorded cancellation.** It does not prove a successful commercial
renewal or healthy ongoing use. The zero is filled via `fillna(0)` after the join.

## Label Source

Labels derive from the **events table** (`contract_events.csv`), not from `order_item_fact`.

## Eligibility

Only rows where `eligibility_reason == 'eligible_subscription'` are used for supervised
training and evaluation:
- `offering_type == 'subscription'`
- `label_matured == True`
- `incomplete_history == False`

## Excluded Rows

| eligibility_reason | Reason | Disposition |
|---|---|---|
| `not_matured` | Label not yet observable (includes 2026-10-01 forward scoring rows) | Excluded from supervised cohort; 1,620 subscription rows retained as scoring queue |
| `eligible_all` | Non-subscription products (support/services/hardware) that are mature + complete | Excluded from subscription churn model; retained for business-metric analysis |
| `incomplete_history` | Predecessor contract quarantined in cleaning; renewal chain incomplete | Excluded from primary supervised cohort; preserved in review queue `incomplete_history_review_queue.csv` |
| `support_or_nonrecurring` | Support/hardware/services rows that are incomplete or immature | Excluded entirely from modeling cohort |

## Forward-Looking Cohort

The 2026-10-01 scoring cohort has **1,620 subscription rows** with no observed cancellation
label — these are forward-looking predictions only.

## Prevalence by Split

Counts below apply the strict `label_available_at < stage_cutoff` rule enforced by
`make_splits.py`. The `additional_train` split has 1,375 rows dropped (labels not yet
mature before the calibration stage cutoff of 2025-10-01); only 33 rows are usable for
post-selection refit before calibration begins.

| Split | As-of dates | Rows | Positives | Prevalence |
|---|---|---:|---:|---:|
| train | 2023-10-01, 2024-10-01 | 2,049 | 141 | 6.88% |
| model_selection | 2025-04-01 | 1,312 | 97 | 7.39% |
| additional_train | 2025-07-01 | 33 | 2 | 6.06% |
| calibration | 2025-10-01 | 1,332 | 107 | 8.03% |
| final_test | 2026-04-01 | 1,502 | 98 | 6.52% |
| scoring | 2026-10-01 | 1,620 | — | forward-looking |

> **Note on `additional_train`:** the 2025-07-01 snapshot has 1,408 subscription rows in
> the raw cohort. Labels mature at `term_end + 90d`. Most of those term_end dates are in
> Q4 2025, so their labels do not become available until after the calibration stage starts.
> Only 33 rows have labels available before the 2025-10-01 calibration cutoff. Include
> them in a post-selection refit only; do not include all 1,408 rows.

## Entity-Overlap Audit

Company recurrence across splits is legitimate for existing-customer renewal prediction.
Key figures from `part2/outputs/entity_overlap_audit.json`:

- **37.6%** of companies appear in multiple labeled splits (expected: customers renew repeatedly)
- **100%** of final-test companies were never in training (no company-level train→test leakage)
- **41%** of parent organizations appear in multiple labeled splits

The final-test never-seen-company subset (all 1,183 final-test companies) serves as the
sensitivity check for performance on companies with no prior training history.

## Temporal Enforcement

`make_splits.py` enforces `label_available_at < stage_cutoff` as a hard gate and exits
with a non-zero status code if any training row violates this constraint. The gate is
re-run every time splits are rebuilt. See `part2/outputs/split_manifest.json` for the
per-split row and positive counts from this run.

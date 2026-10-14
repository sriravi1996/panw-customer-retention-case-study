> Current revision results: `output/model_demo_revision/RESULTS.md`. Scenario SA-011 supersedes additive-only generation descriptions. Current predictive feature lists and cleaning/label timing remain unchanged.

> Current revision results: `output/model_demo_revision/RESULTS.md`. Scenario SA-011 supersedes additive-only generation descriptions. Current predictive feature lists and cleaning/label timing remain unchanged.

> Current revision results: `output/model_demo_revision/RESULTS.md`. Scenario SA-011 supersedes additive-only generation descriptions. Current predictive feature lists and cleaning/label timing remain unchanged.

> Current revision results: `output/model_demo_revision/RESULTS.md`. Scenario SA-011 supersedes additive-only generation descriptions. Current predictive feature lists and cleaning/label timing remain unchanged.

> **Experimental copy, 2026-10-14:** The active data/model scenario is
> `interaction_demo_v1`. Previous numerical findings and copied presentations
> predate this regeneration and must not be used as current evidence. See
> `output/model_demo_revision/RESULTS.md` for the regenerated results and
> `output/model_demo_revision/frozen_design.md` for the declared assumptions.
> This is a synthetic mechanism demonstration, not validated PANW performance.

# Analysis contract

## Business question

Which subscription renewals deserve customer-success review when capacity is
limited to an assumed 30 purchasing accounts per cycle? Compare cancellation
identification with annual-value exposure. All accounts are directly managed
under the case-study assumption. See `customer_management_scope.md`.

## Observation and prediction

Operating observation period: 2021-10-01 to 2026-10-01, with opening histories
back to 2018-10-01. One row per eligible subscription contract at
`pred_date = term_end - 90 days`; contract_id is unique in the per-expiry table.
The target is a qualifying recorded cancellation after prediction through
expiry plus 90 days. A mature zero does not confirm commercial renewal.

## Temporal evaluation

Train through the first fitting boundary, choose candidates in the next period,
refit on train plus selection, and inspect later periods. Label maturity requires
180 days from prediction, so 6,815 transition observations
are excluded. Exact date boundaries are in `make_splits.py` and `timing_audit.json`.
The repeatedly inspected later evaluation remains exploratory.

## Business review

On 2026-04-01, gather expiries from 2026-04-02 through 2026-06-30.
The active pool contains 1,505 contracts across
1,185 accounts. All policies use the
same pool. Keep one highest-ranked lead contract per account under each policy.
Original expiry-minus-90 scores can be 0 to 89 days old at review. Four manual
checks concern prior contact, measurement quality, escalation and urgency.

## Measures

Selection: average precision. Business review: cancelled-lead precision,
account recall and share of subsequently observed cancelled ACV. Report
parent-bootstrap uncertainty with shared draws across policies. Model score
multiplied by ACV is an uncalibrated priority index, not expected dollar loss.
Financial snapshots and full-history expansion answer a separate growth question.
No treatment effect is simulated. Evaluate interventions separately.

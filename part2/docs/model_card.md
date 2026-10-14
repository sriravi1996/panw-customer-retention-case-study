# Model card: synthetic experience demonstration

**Scope:** generated customer histories under declared assumptions. These results do not validate real Palo Alto Networks behavior. The original project and its validated presentation remain unchanged.

## What changed

Part 1 now uses a shared customer-experience state more consistently for coverage, successful workflows and cancellation. Cancellation risk includes declared interactions rather than only an additive weighted sum. A separate reference population anchors annual segment probabilities. Master-seed isolation prevents shared renewal draws across replications. No outcomes were manually changed, no evaluation accounts were removed to improve scores, and the rule formula was not weakened.

Part 2 keeps the existing feature sets, LR/HGB grids, AP-based historical model selection, refit and temporal splits. The review scorer now accepts whichever feature experiment actually wins and includes capacity 100. Run-ID and refit metadata are derived from current artifacts rather than stale constants. Regression checks validate the declared policy-selection rule rather than forcing a particular historical winner.

## Selected model and untouched temporal evaluation

- Selected: **full_context_hgb**, using 41 numeric and 4 categorical inputs.
- Selection AP: **0.328**. Rule selection AP: **0.211**.
- Refit: 7,004 rows, 402 cancellations. No final-test labels in fitting.
- Primary final-test AP: **0.246**. ROC-AUC: **0.792**. Rule AP: **0.132**.

AP describes the ranking across thresholds. It does not guarantee superiority at each account-review capacity. Scores are rankings, not validated probabilities or estimated revenue saved.

## Main business comparison: 30 account reviews

| Simulation seed | Approach | Cancelled leads found | Precision | Cancelled ACV captured |
|---|---|---|---|---|
| 137 | Health rules | 3/30 | 10.0% | 8.0% |
| 137 | ACV only | 0/30 | 0.0% | 0.0% |
| 137 | Model score | 13/30 | 43.3% | 7.3% |
| 137 | Model score × ACV | 11/30 | 36.7% | 34.1% |
| 991 | Health rules | 8/30 | 26.7% | 8.7% |
| 991 | ACV only | 0/30 | 0.0% | 0.0% |
| 991 | Model score | 16/30 | 53.3% | 13.7% |
| 991 | Model score × ACV | 3/30 | 10.0% | 18.2% |
| 2027 | Health rules | 7/30 | 23.3% | 76.1% |
| 2027 | ACV only | 2/30 | 6.7% | 75.1% |
| 2027 | Model score | 9/30 | 30.0% | 2.9% |
| 2027 | Model score × ACV | 4/30 | 13.3% | 76.7% |

The same eligible accounts and label definition are used across policies within each seed. One policy-specific lead contract represents each account. The primary seed is 137. Seeds 991 and 2027 are generated separately and scored with the frozen primary model and its preprocessing, without refitting. All predeclared seeds are shown.

- Seed 137: model minus rules at 30 reviews = **+33.3 percentage points** (+10 cancelled leads).
- Seed 991: model minus rules at 30 reviews = **+26.7 percentage points** (+8 cancelled leads).
- Seed 2027: model minus rules at 30 reviews = **+6.7 percentage points** (+2 cancelled leads).

The model has higher observed top-30 precision in **3/3** predeclared populations. Three replications are a sensitivity check, not a guarantee. Do not turn a favorable pooled result into a claim that every individual cohort wins.

## Temporal test metrics on each population

| Seed | Approach | Test contracts | Cancellations | Prevalence | AP | ROC-AUC |
|---|---|---|---|---|---|---|
| 137 | Model score | 3139 | 141 | 0.045 | 0.246 | 0.792 |
| 137 | Health rules | 3139 | 141 | 0.045 | 0.132 | 0.763 |
| 991 | Model score | 3013 | 136 | 0.045 | 0.217 | 0.757 |
| 991 | Health rules | 3013 | 136 | 0.045 | 0.126 | 0.709 |
| 2027 | Model score | 3185 | 125 | 0.039 | 0.228 | 0.826 |
| 2027 | Health rules | 3185 | 125 | 0.039 | 0.142 | 0.757 |

## Top five predictive inputs

Importance is the fall in selection-set average precision after shuffling one input. Correlated inputs can share importance. These are predictive associations within our assumed simulation, not measured causal effects.

- **Successful use: the share of product workflow attempts that succeeded in the preceding 30 days** (`workflow_success_30d`): AP importance +0.1816, repeat standard deviation 0.0116.
- **Protection coverage: average protected share of intended scope over the preceding 90 days** (`avg_coverage_90d`): AP importance +0.1724, repeat standard deviation 0.0166.
- **Business size: context for the explicitly assumed segment risk anchors, not a causal reason to cancel** (`employee_count`): AP importance +0.0127, repeat standard deviation 0.0211.
- **Change in successful use: recent workflow success rate minus the preceding 30-day rate** (`productive_use_delta_30d`): AP importance +0.0121, repeat standard deviation 0.0058.
- **Deployment reach: deployed targets divided by intended targets in recent observations** (`deployment_reach_30d`): AP importance +0.0061, repeat standard deviation 0.0072.

Coverage and deployment reach overlap, so their individual importance is not independent. Small importance relative to repeat variability should not be described as a robust top driver.

Primary paired parent-bootstrap 90% interval for model-minus-rules precision: **+16.7 to +53.3 percentage points**. This includes ranking resampling and parent coordination. The interval applies to this synthetic review population and does not measure uncertainty about real PANW customers.

## Assumptions to explain in the presentation

1. Annual segment risk anchors are invented: SMB 8%, midmarket 5%, enterprise 2.5%, large enterprise 1%. Reference annual risk differs from observed contract cancellation frequency.
2. Multiple negative experiences can reinforce each other. The interaction coefficients and measurement strengths were deliberately chosen for this demonstration and frozen before the new evaluations.
3. Telemetry is largely reliable. Low protected scope and poor successful use are customer-experience signals; missing measurement is not itself treated as a real business cause of cancellation.
4. Stable company experience creates persistence over time, which makes historical learning possible. This is simpler than real changing product health.
5. A recorded product-contract cancellation is the target. No recorded cancellation does not establish that a renewal succeeded. Support products are not in the subscription churn cohort.
6. Predictions use information available 90 days before expiry, with outcomes observed through expiry plus 90 days. Accounts can have multiple contracts but review capacity counts accounts.
7. All accounts are directly managed. No reseller features or route exclusions are used.
8. Score × ACV prioritizes financial exposure. It is distinct from maximizing precision and does not estimate intervention benefits.

## What can be claimed

“Under a declared synthetic customer-experience scenario, we tested whether a learned model could rank cancellation risk more effectively than a fixed health score. We preserved the benchmark and evaluated the frozen model on all predeclared populations. The tables above state where it helped and where it did not.”

Any claim of a successful top-30 ranking must cite the actual seed-specific results, capacity sensitivity and uncertainty. The scenario was developed after seeing the original experiment, so this is an intentionally constructed simulation demonstration, not independent confirmation that the assumed mechanisms are true.

## Remaining work

Use real customer data to test the assumed mechanisms, validate probability calibration before financial-loss interpretation, and run an intervention pilot to estimate retention impact. The current editable presentation is `output/presentations/Case_Study_Retention_and_Revenue_Growth_2026-10-14.pptx`. Its charts, model features, counts and assumptions match the current results, including all three reported populations. The submission contains one final presentation.

## Reproduction

Run `bash run_part1.sh` and `bash part2/run_part2.sh` for the primary pipeline. Run `python output/model_demo_revision/run_replications.py` to generate both fixed replication populations, then `python output/model_demo_revision/evaluate_frozen_replications.py` with the same pinned Python environment. The checked-in RESULTS.md contains the report.

The parent bootstrap uses an equivalent row-gather implementation to avoid thousands of dataframe copies per iteration. Exact resampled rows and occurrence identities matched the old implementation on 50 tests. No statistical method or model result was changed by this optimization. A later audit added deterministic contract-ID tie-breaking to every policy, correcting row-order sensitivity at equal scores. Rankings and intervals were regenerated with that common rule. Eleven additional mechanism, reproducibility and current-review checks accompany the existing pipeline checks. Final tests: 44 passed and 6 skipped. The six skips concern archived outputs from retired snapshot stages; they are not checks on the active review.

Original-project source/model/result hashes verified unchanged. The old health-rule file also matches its original hash. Detailed evidence lives in the adjacent CSVs, frozen-source receipt, pipeline logs and replication receipt.

# Simulation Assumptions Register

**SA-001 through SA-009.** Stable IDs — do not renumber when assumptions change; add a
note and date of revision instead.

Active revision: `interaction_demo_v1`, primary seed 137. Current run identifiers are recorded in `outputs/generation_manifest.json`.

---

## SA-001 Annual-to-Term Risk Conversion

**Claim:** Cancellation probability is specified per year and converted to a per-term risk
before sampling.

**Formula:** `q = 1 − (1 − r)^T`  
where `r` is the segment annual cancellation rate and `T` is the contract term in years.

**Implementation:** `generate_data.py`, lines 1111–1112:
```python
per_renewal_prob = 1.0 - (1.0 - annual_cancel_prob) ** current_term_years
```

**Implication:** A 3-year contract for an SMB company (r = 8%) has a per-renewal
cancellation probability of `1 − 0.92³ ≈ 22.5%`, not 8%. This is the chosen conversion from an annual probability to a contract-term outcome.
It models accumulated risk observed at expiry, not a claim that a locked multi-year
contract can cancel freely in each intervening year.

**Status:** Correctly implemented. This is a documentation note, not a code bug.

---

## SA-002 Segment Annual Cancellation Rates

**Values (from `config/simulation.yaml`):**

| Segment | Annual cancellation rate |
|---|---|
| SMB | 8% |
| Midmarket | 5% |
| Enterprise | 2.5% |
| Large Enterprise | 1% |

**Interpretation:** These are hazard rates for a customer deciding not to renew in any
given 12-month period, independent of contract term. The actual per-contract cancellation
probability scales with term (SA-001).

**Note:** These are synthetic reference-population annual probabilities, not estimates
of actual Palo Alto Networks cancellation rates. The active scenario uses an independent
reference population to anchor them; actual contract cancellation frequency also depends
on term mix, survival, budgets and switching costs. See SA-011 and current results.

---

## SA-003 Shared Forward Experience State

A noisy company-level experience state is assigned before cancellation is sampled.
It is a common cause of cancellation and observable operating measurements.
The generator first creates commercial records and sampled outcomes, then materializes
telemetry, support and quotes from that same pre-outcome state. The physical CSV-writing
order does not mean cancellation determines historical coverage or workflow success.
Deployment end dates determine how long observations continue, while historical features
are restricted to source availability at prediction time. The existing future-change
invariance regression remains in force. Hidden states and probabilities stay in eval-only
files, never in predictor tables. See SA-011 for active interactions and measurement noise.

---

## SA-004 Reliable Telemetry Scenario

**Claim:** The main simulation scenario generates near-zero missing telemetry.

**Value:** `missing_telemetry_pct_90d ≈ 0%` in every row of the main dataset.

**Consequence:** `missing_telemetry_pct_90d` is excluded from all feature sets and the
rule baseline (see `_features.py::PROHIBITED_PREDICTORS`). Including it would add a
constant-near-zero predictor that adds noise and no signal. A real deployment would need
to re-evaluate this feature.

**Caveat:** `obs_completeness_90d` measures the fraction of expected weekly telemetry
windows that are non-null. In the main scenario this is high, but it is retained as a
feature because late-arriving data can still cause partial completeness even when there is
no systematic reporting failure.

---

## SA-005 Product-Level Feature Applicability

**Claim:** Some features are defined only for specific product types.

| Feature | Applies to | Non-applicable fill |
|---|---|---|
| `scan_success_rate_90d` | Prisma Cloud only | 1.0 (no penalty for non-Prisma) |
| `deployment_reach_30d` | Products with measurable deployment scope | 0.0 + `_missing` indicator |
| `workflow_success_30d` | Products with logged workflow events | 0.5 (neutral) + `_missing` indicator |
| `onboarding_delay_days` | Products with a defined first-value milestone | 0.0 (no delay recorded) |

**Implication:** Models trained on mixed product-family data will see these features with
their non-applicable fill values for contracts where the measurement does not exist.
This is intended — the indicator columns (`deployment_reach_30d_missing`, etc.) allow the
model to distinguish "not applicable" from "genuinely zero."

---

## SA-006 Observation Window and Temporal Scope

**Values:**
- Observation start: 2021-10-01
- Observation end (data freeze): 2026-10-01
- First labeled prediction window: contracts with `pred_date ≥ 2021-10-01`
- Label maturity: `label_available_at = term_end + 90 days ≤ 2026-10-01`

**Per-contract prediction date:** `pred_date = term_end − 90 days`. Features are computed
from data available at `pred_date` (i.e., source timestamps and `available_at` ≤ `pred_date`).

**Forward scoring queue:** Contracts with `pred_date ≥ 2025-10-01` and
`label_available_at > 2026-10-01` are in the scoring queue. These have no observed outcome
and are used only for forward-looking prioritization.

---

## SA-007 Company-Level vs. Contract-Level Attribution

**Claim:** Support tickets are attributed to a deployment (and thus a contract) by FK, but
company-level support context can affect a contract's service features.

**Details:**
- `contract_id` FK in `support_tickets_clean.csv` links tickets to a specific contract
- Service features are aggregated at the contract level via this FK
- A single account-level support incident affecting multiple products may appear under
  only one contract's FK, creating uneven attribution across contracts from the same company

**Implication:** `unresolved_customer_impact_hours_90d` and `repeat_sla_breaches_90d` should
be interpreted as *contract-level ticket context*, not as a complete picture of the account's
support burden.

---

## SA-008 Relationship Signal in Synthetic Data

**Claim:** The simulator generates CRM interaction frequency correlated with company size
segment, not specifically with cancellation events.

**Consequence:** The relationship feature block (`executive_interactions_180d`,
`marketing_engagements_180d`, `days_since_last_interaction`) may hurt average precision
on synthetic data because the signal is not causally linked to the churn mechanism.

**Action:** Do not drop the relationship block based solely on synthetic data results.
Re-evaluate on real CRM data before excluding. See `feature_contract.md` Known Limitations.

---

## SA-009 Term Mix by Segment

**Claim:** The simulator uses realistic term distributions by segment.

**Values (from `config/simulation.yaml`):**

| Segment | 1-yr | 2-yr | 3-yr |
|---|---|---|---|
| SMB | 70% | 20% | 10% |
| Midmarket | 50% | 30% | 20% |
| Enterprise | 30% | 35% | 35% |
| Large Enterprise | 20% | 30% | 50% |

**Implication:** Enterprise and large-enterprise customers are more likely to have 3-year
terms, increasing per-contract risk at a fixed annual probability (SA-001). Their lower annual
segment anchor can offset that increase; longer terms alone do not reduce per-contract risk. Interpreting model outputs for these segments requires understanding that
the model scores per-contract risk, not annual attrition rate.

---

*Last updated: 2026-10-13 for per-contract modeling rebuild.*


## SA-011: Separate customer-experience demonstration scenario (2026-10-14)

This folder is an experimental copy. The original project remains unchanged at
`a separately archived development workspace`. This revision was
designed after inspecting prior model results. It demonstrates assumed synthetic
mechanisms and is not evidence of real PANW customer behavior.

The active `experience_scenario` in `config/simulation.yaml` supersedes older
additive-only renewal descriptions in this document. The existing rule baseline
remains unchanged and is not fitted to the new outcomes.

**Latent-state interactions in log-odds units:**

- Deployment gap >45% and value deterioration >35%: +3.2.
- Value deterioration >55% and deployment gap >15%: +2.2.
- Commercial pressure >60% and value deterioration >20%: +1.2.
- Service problem >20% and deployment gap >30%: +0.8.
- Smooth main effects: 1.2×deployment gap + 0.8×value deterioration +
  0.6×service problem + 0.3×commercial pressure.

These are invented scenario coefficients, not causal estimates or learned business
thresholds. More than one interaction can apply. A separate fixed reference sample
of 20,000 latent customers per segment anchors mean annual probabilities to the
existing 8%/5%/2.5%/1% assumptions. No observed cancellation labels or holdout data
are used to set intercepts. Budget pressure, sibling effects, term mix, probability
floors/caps and survival selection mean realized contract rates need not equal
those annual reference rates. Multi-year terms still use `1-(1-annual_risk)^years`.

**Observation coherence:** coverage combines 75% shared experience (one minus
deployment gap) and 25% the existing deployment trajectory, with weekly noise.
Workflow success has mean `0.90 - 0.80×value deterioration`, with weekly noise and
binomial event counts. Neither depends on the sampled cancellation. The hidden
state and computed probabilities stay in eval-only files and are excluded from
model features. Raw-file write order is not a claim that outcomes caused behavior: a
shared pre-outcome state generates both independently, conditional on that state.

**Frozen evaluation:** seed 137 is the primary 10,000-account run. Seeds 991 and
2027 are independent 10,000-account replications scored with the frozen primary
model and preprocessing. All seeds are reported. Historical AP selects the model;
precision at 30 account reviews is the main operational comparison, with 25/50/100
sensitivity and cancelled-ACV capture reported separately. No top-30 advantage is
guaranteed. Unchanged temporal splits, labels, cleaning and model grids apply.

See `output/model_demo_revision/frozen_design.md`, `frozen_source_hashes.json`
and the generated results report for the current revision's evidence.

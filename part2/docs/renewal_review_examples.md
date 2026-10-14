# Renewal Review — Three Illustrative Action Examples

**Portfolio review date D:** 2026-04-01  
**Review cohort:** 1,540 contracts, term_end in (2026-04-01, 2026-06-30]  
**Model:** full_context_hgb_refitted.pkl (full_context/hgb; fitting complete by 2024-10-01)  

> Scores shown are historical 90-day-before-expiry scores generated at > each contract's original_prediction_date. They were not refreshed at D. > Review-date eligibility checks (ownership, duplicate outreach, escalation > gates) apply separately and are not shown here.

> This cohort is drawn from an inspected test period. Examples are > retrospective illustrations, not prospective predictions.

## Example A — High-Risk, Churned (True Positive)

| Field | Value |
|---|---|
| contract_id | `CTR0000056312` |
| product_family | SASE |
| size_segment | smb |
| ACV | $13,941 |
| original_prediction_date | 2026-03-26 |
| term_end | 2026-06-24 |
| score_age_days | 6 |
| **Actual outcome** | **CANCELLED** |
| model_score | 0.7572 |
| rule_score | 0.4968 |

**Original feature values at prediction_date:**

| Feature | Value | Plain-language meaning |
|---|---|---|
| deployment_reach_30d | 0.05 | Low deployment reach (5%) — significant undeployed scope |
| workflow_success_30d | 0.45 | Moderate workflow success (45%) |
| avg_coverage_90d | 0.08 | Low active coverage (8%) |
| p1_tickets_90d | 0 | No critical tickets in 90 days |
| unresolved_impact_hrs | 0 | No unresolved customer impact |
| repeat_sla_breaches_90d | 0 | No open SLA breaches at prediction |
| dated_budget_concern | 0 | No budget concern recorded |
| quoted_unit_price_change_pct | 2.88% | Price increase |
| days_since_last_interaction | 587 | 587 days since last interaction (engagement gap) |
| overdue_agreed_actions | 0 | No overdue agreed actions |
| onboarding_delay_days | 0 | Onboarding on-time or no milestone defined |

**Proposed action:** adoption investigation

**Review status:** Included in review cohort. Score is historical (from original_prediction_date); review-date operational checks apply separately.

**Illustrative owner/deadline:** Customer Success Manager; contact within 10 business days of 2026-04-01.

> *Note: Action packages are illustrative. No measured treatment effect. Existing escalation processes are a review gate, not proof the model caused an improvement.*

## Example B — High-Risk, Retained (False Positive)

| Field | Value |
|---|---|
| contract_id | `CTR0000038239` |
| product_family | Prisma |
| size_segment | smb |
| ACV | $37,284 |
| original_prediction_date | 2026-03-09 |
| term_end | 2026-06-07 |
| score_age_days | 23 |
| **Actual outcome** | **RENEWED** |
| model_score | 0.7381 |
| rule_score | 0.6336 |

**Original feature values at prediction_date:**

| Feature | Value | Plain-language meaning |
|---|---|---|
| deployment_reach_30d | 0.18 | Low deployment reach (18%) — significant undeployed scope |
| workflow_success_30d | 0.46 | Moderate workflow success (46%) |
| avg_coverage_90d | 0.22 | Low active coverage (22%) |
| p1_tickets_90d | 0 | No critical tickets in 90 days |
| unresolved_impact_hrs | 1248 | 1248h unresolved customer impact — escalation warranted |
| repeat_sla_breaches_90d | 1 | 1 open SLA breach(es) at prediction time |
| dated_budget_concern | 0 | No budget concern recorded |
| quoted_unit_price_change_pct | 5.60% | Price increase |
| days_since_last_interaction | 704 | 704 days since last interaction (engagement gap) |
| overdue_agreed_actions | 0 | No overdue agreed actions |
| onboarding_delay_days | 0 | Onboarding on-time or no milestone defined |

**Proposed action:** adoption investigation; technical escalation; value/renewal review

**Review status:** Included in review cohort. Score is historical (from original_prediction_date); review-date operational checks apply separately.

**Illustrative owner/deadline:** Customer Success Manager; contact within 10 business days of 2026-04-01.

> *Note: Action packages are illustrative. No measured treatment effect. Existing escalation processes are a review gate, not proof the model caused an improvement.*

## Example C — Low-Risk, Retained (True Negative)

| Field | Value |
|---|---|
| contract_id | `CTR0000042730` |
| product_family | Prisma |
| size_segment | large_enterprise |
| ACV | $1,366,620 |
| original_prediction_date | 2026-02-14 |
| term_end | 2026-05-15 |
| score_age_days | 46 |
| **Actual outcome** | **RENEWED** |
| model_score | 0.0015 |
| rule_score | 0.0485 |

**Original feature values at prediction_date:**

| Feature | Value | Plain-language meaning |
|---|---|---|
| deployment_reach_30d | 0.93 | Strong deployment reach (93%) |
| workflow_success_30d | 0.92 | High workflow success (92%) |
| avg_coverage_90d | 0.94 | Good coverage (94%) |
| p1_tickets_90d | 0 | No critical tickets in 90 days |
| unresolved_impact_hrs | 0 | No unresolved customer impact |
| repeat_sla_breaches_90d | 0 | No open SLA breaches at prediction |
| dated_budget_concern | 0 | No budget concern recorded |
| quoted_unit_price_change_pct | 1.05% | Price increase |
| days_since_last_interaction | 24 | 24 days since last interaction |
| overdue_agreed_actions | 0 | No overdue agreed actions |
| onboarding_delay_days | 0 | Onboarding on-time or no milestone defined |

**Proposed action:** routine engagement

**Review status:** Included in review cohort. Score is historical (from original_prediction_date); review-date operational checks apply separately.

**Illustrative owner/deadline:** Customer Success Manager; contact within 10 business days of 2026-04-01.

> *Note: Action packages are illustrative. No measured treatment effect. Existing escalation processes are a review gate, not proof the model caused an improvement.*

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

# Feature contract: active cancellation model

The selected artifact reads only the feature lists below. Sales-route fields are
excluded. Every measurement must be known by each contract's prediction date,
90 days before expiry. Count/binary fields and missing indicators are numeric.
Raw simulator states, true cancellation probabilities, future outcomes and
successor terms are prohibited.

## Selected inputs

### Numeric

- `deployment_reach_30d`
- `onboarding_delay_days`
- `workflow_success_30d`
- `productive_use_delta_30d`
- `unresolved_customer_impact_hours_90d`
- `repeat_sla_breaches_90d`
- `p1_tickets_90d`
- `open_tickets_90d`
- `avg_open_age_hrs_90d`
- `avg_resolution_hrs_90d`
- `avg_coverage_90d`
- `coverage_trend_90d`
- `obs_completeness_90d`
- `quoted_unit_price_change_pct`
- `dated_budget_concern`
- `days_without_confirmed_owner`
- `overdue_agreed_actions`
- `days_to_renewal`
- `tenure_days`
- `avg_resolution_hrs_90d_missing`
- `deployment_reach_30d_missing`
- `workflow_success_30d_missing`
- `days_since_last_interaction_missing`
- `median_prior_renewal_delay_missing`
- `employee_count`
- `acv`
- `total_acv_in_window`
- `n_products_in_window`
- `n_families_in_window`
- `n_active_products_in_family`
- `family_sibling_cancellations_1yr`
- `family_adoption_completeness`
- `executive_interactions_180d`
- `marketing_engagements_180d`
- `days_since_last_interaction`
- `median_prior_renewal_delay_days`
- `tickets_90d`
- `policy_events_per_unit_90d`
- `remediation_velocity_90d`
- `automation_adoption_90d`
- `scan_success_rate_90d`

### Categorical

- `product_family`
- `size_segment`
- `industry`
- `region`

## Interpretation and missingness

Deployment reach is protected/deployed scope divided by eligible intended scope.
Workflow success uses successful attempts divided by attempted operations.
Coverage is the protected share, not raw administrator login activity.
Observed completeness is received / (received + missing + overdue) observations.
`median_prior_renewal_delay_days` currently measures the immediate prior
subscription transition gap, not a full-history median.
`days_without_confirmed_owner` is recency since an owner-bearing CRM interaction,
not proof of absent ownership. `family_adoption_completeness` measures eligible
catalog ownership, not deployment success. `repeat_sla_breaches_90d` counts open
breached cases at prediction. `days_to_renewal` is fixed at 90 in this design.

Deployment/workflow measures use recent 30 days; other operating/support measures
generally use 90 days. Executive and marketing counts use 180 days. Tenure and
previous-renewal context can extend earlier. Use `_features.py` and
`imputation_values.json` for exact preprocessing, including train-fitted medians.

Current rule and ML preprocessing fill missing deployment reach with zero before
scoring, so the health rule then assigns a full deployment-gap penalty. A missing
flag preserves this case. This is a declared assumption, not proof of poor adoption.
Missing workflow success receives neutral 0.5. A scan-success measure can be
inapplicable for products without a scan workflow.

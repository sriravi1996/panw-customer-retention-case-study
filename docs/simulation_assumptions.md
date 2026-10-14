# Simulation Assumptions

All parameters are explicit synthetic assumptions unless marked [PUBLIC SOURCE].

## Scale Parameters

| Parameter | Value | Source / Rationale | Uncertainty | Sensitivity range |
|---|---|---|---|---|
| company_count | 10,000 | Mid-scale simulation run for this case study. Real PANW serves ~70,000 customers [PUBLIC: FY2026 results]; this study uses 10,000 synthetic accounts whose size distribution and churn rates are declared assumptions, not a verified sample of the PANW customer base. | Scale only affects statistical power, not logic | 2,000 (dev) – 10,000 (full) |
| avg_users_per_account | 3.0 | Synthetic: buyer + admin + champion | Medium | 2–5 |
| avg_teams_per_account | 1.5 | Synthetic: SMB 1, enterprise 3-5 | Medium | 1–4 |
| observation_window | 2021-10-01 → 2026-10-01 (60 months) | 5-year window supports multiple historical scoring snapshots and opening installed base | Fixed | — |
| opening_base_start | 2018-10-01 | Up to 3 years of pre-window first purchases (~40% of accounts) | Synthetic | — |
| history_months | 60 | 5-year window chosen to support multiple training cohorts and mature labels | Fixed | — |
| label_window_days | 90 | Outcome window after contract expiry: label_matured=True iff term_end + 90d ≤ obs_end. This is NOT 90 days after the scoring date — contracts that expire after the scoring date use the 90-day window from their own term_end. | Fixed | — |
| seed | 42 | Reproducibility | Fixed | — |

**Note on multi-account parent realized distribution**: The configured `parent_org_share_multi_account` is a parent-level probability; the realized share of multi-account parents depends on the drawn parent count. At 10,000 accounts, 625/8,749 parents (7.1%) have multiple accounts — lower than the 15% configured probability because most parents draw only one account.

## Company Size Distribution

| Segment | % of companies | Employee range | ARR range (synthetic) |
|---|---|---|---|
| SMB | 45% | 50–499 | $5K–$50K |
| Mid-market | 35% | 500–4,999 | $50K–$500K |
| Enterprise | 18% | 5,000–49,999 | $500K–$5M |
| Large Enterprise | 2% | 50,000+ | $5M–$30M |

## Product Penetration (initial purchase probability per company)

| Product Family | Penetration | Rationale |
|---|---|---|
| Network Security (Strata) | 55% | Largest installed base, firewall incumbency |
| SASE / Prisma Access | 22% | Growing but newer motion |
| Cloud Security (Prisma Cloud) | 18% | Cloud-native buyers, higher skew to enterprise |
| SOC / XDR (Cortex XDR) | 20% | Endpoint-first entry; broad mid-market |
| XSOAR | 8% | Larger SOC teams; enterprise skew |
| XSIAM | 5% | New; enterprise early adopters |
| Xpanse | 7% | Attack surface management niche |
| Support (Premium/Enhanced) | 40% | Attached to hardware/subscription contracts |

Multi-product: ~30% of companies own 2+ families; ~8% own 3+ families. These are synthetic assumptions, not published PANW attach rates.

## Billing Units (per product, synthetic list prices)

| Product | Billing Unit | Synthetic list price |
|---|---|---|
| PA-Series HW (SMB) | device | $8,000 one-time |
| PA-Series HW (Enterprise) | device | $45,000 one-time |
| VM-Series | vCPU | $1,200/vCPU/yr |
| CN-Series | credit | $500/credit/yr |
| Prisma Access – Mobile Users | user/yr | $120/user/yr |
| Prisma Access – Remote Networks | Mbps | $800/Mbps/yr |
| GlobalProtect | user/yr | $60/user/yr |
| Prisma Cloud CSPM | cloud_account/yr | $2,400/account/yr |
| Prisma Cloud CWP | workload/yr | $1,800/workload/yr |
| Prisma Cloud DSPM | TB/yr | $3,000/TB/yr |
| Prisma Cloud AppSec | app/yr | $5,000/app/yr |
| Cortex Cloud | credit | $600/credit/yr |
| Cortex XDR | endpoint/yr | $50/endpoint/yr |
| Cortex XSOAR | user/yr | $8,000/user/yr |
| Cortex XSIAM | TB_ingest/yr | $15,000/TB/yr |
| Cortex Xpanse | asset/yr | $25/asset/yr |
| Premium Support | % of contract | 20% of contract ACV |
| Enhanced Support | % of contract | 12% of contract ACV |
| Professional Services | engagement | $25,000–$250,000 one-time |

All prices are synthetic approximations for simulation purposes only. They are not published PANW prices.

## Term Lengths

| Segment | 1-yr % | 2-yr % | 3-yr % |
|---|---|---|---|
| SMB | 70% | 20% | 10% |
| Mid-market | 45% | 35% | 20% |
| Enterprise | 25% | 35% | 40% |
| Large Enterprise | 15% | 30% | 55% |

## Churn / Retention Assumptions (synthetic)

| Metric | Value | Notes |
|---|---|---|
| Annual logo churn rate | 1–8% | Varies by segment: SMB ~8%, Mid-market ~5%, Enterprise ~2.5%, Large Enterprise ~1% (config/simulation.yaml) |
| Annual contraction rate | 5% | Among retained accounts |
| Annual expansion rate | 12% | Among retained accounts |
| Product non-renewal (single product) | ~18% effective | Higher without multi-product stickiness |
| Abandonment (operational) | 15% | Not all become financial churn |

Active scenario: cancellation is sampled from shared experience states and the nonlinear interactions in SA-011, using reference-population annual segment anchors. The weighted annual anchor across the 45/35/18/2% size mix is approximately 5.8%. Realized contract cancellation frequency differs because of term mix, survival, budget pressure and switching costs.

## Behavioral Scenarios (deployment-level)

Each deployment is assigned one of these behavioral archetypes at generation time (proportions from `config/simulation.yaml`):

| Archetype | % of deployments | Description |
|---|---|---|
| healthy_steady | 38% | Stable usage, grows with workload |
| delayed_onboarding | 12% | Slow ramp — full plateau reached ~15 months in (normalized to a 5-year horizon) |
| gradual_abandonment | 10% | Usage declines monotonically over the deployment lifetime |
| workload_growth | 8% | Expanding coverage over time |
| workload_shrinkage | 6% | Contracted scope |
| seasonal | 5% | Sinusoidal pattern with ~30-month cycle (2 complete cycles over 5 years) |
| failed_deployment | 4% | Never reaches healthy state |
| sensor_outage | 5% | Gaps in telemetry |
| platform_migration | 4% | Legacy down, replacement up |
| champion_turnover | 4% | Usage dip spanning ~10–15 months mid-tenure, followed by recovery |
| budget_pressure | 4% | Steady decline, risk of downgrade |

**Behavioral trajectory note**: Usage ratios are computed using `t = elapsed_days / (5 × 365)`, a fixed 5-year normalization. This means all temporal durations (ramp periods, dip durations, cycle lengths) are proportional to the 5-year observation window, not to calendar benchmarks like "90 days" or "quarterly". The implemented trajectories are deterministic and reproducible (seed 42).

These archetypes are simulator-only labels stored in `eval/deployment_archetypes.csv`, never in deployable feature tables.

## Dirty Data Injection Rates

| Anomaly type | Rate | Purpose |
|---|---|---|
| Duplicate order records | 1.5% of orders | Test dedup logic |
| Inconsistent product_id casing | 2% of order_items | Test normalization |
| Missing team mappings | 3% of deployments | Test FK handling |
| Invalid/unparseable dates | 1% of records | Test date cleaning |
| Late-arriving telemetry events | 5% of weekly rows | Test freshness logic |
| Missing telemetry (not zero) | 8% of deployment-weeks | Test missing vs zero |

Corruption is injected after creating the canonical source; ground-truth mapping preserved in `eval/anomaly_ground_truth.csv`.

## Account Interactions

CRM-style relationship touchpoints generated per account from first_purchase_date onward.

### Volume assumptions (synthetic, per customer-year)

| Segment | Interactions / yr | Source / Rationale |
|---|---|---|
| SMB | 1.5 | Low-touch motion; primarily marketing + CS |
| Mid-market | 3.0 | Moderate CS cadence with occasional exec engagement |
| Enterprise | 6.0 | Named-account CS model; regular QBRs and exec briefings |
| Large Enterprise | 10.0 | Dedicated CSM; frequent exec engagement |

Volume is computed as a Poisson process over the account's **eligible duration** (from `max(first_purchase_date, observation_start)` to `observation_end`), not the full 5-year window. This prevents inflated counts for mid-window new customers.

### Interaction type weights (synthetic, by segment)

| Type | SMB | Mid-market | Enterprise | Large Enterprise | Lifecycle constraint |
|---|---|---|---|---|---|
| executive_briefing | 5% | 10% | 20% | 30% | Post-sale only (≥ first_purchase_date) |
| marketing_event | 35% | 30% | 25% | 20% | first_purchase_date (customer-only; all types are post-sale) |
| cs_meeting | 45% | 40% | 35% | 30% | Post-sale only |
| sales_meeting | 15% | 20% | 20% | 20% | Post-sale only |

### Timestamp assumptions

- `event_at`: the date the interaction occurred; drawn uniformly within the eligible window.
- `available_at`: CRM logging delay of 0–3 days after `event_at` (uniform integer). Always ≥ event_at; never null.
- Downstream features apply point-in-time filters: `available_at <= as_of_date` AND `event_at <= as_of_date`.

### Cleaning rules

Rows are quarantined (not silently corrected) for: null `interaction_id`, unparseable `event_at`, null `available_at`, `available_at < event_at`, unknown `company_id`, and unknown `interaction_type`. Deduplication on `interaction_id`.

## Support Ticket Volume

Tickets are generated per-account as a Poisson process scaled by actual customer exposure (from `first_purchase_date` to `observation_end`), not a fixed 5-year window. This prevents inflated counts for recently-acquired accounts.

| Parameter | Value | Rationale |
|---|---|---|
| Onboarding window | First 90 days | Higher-touch period as customer learns product |
| Onboarding multiplier | 2.5× annual rate | Tickets arrive faster at launch |
| Mature annual rate (SMB) | 3 tickets/yr | Low-touch segment |
| Mature annual rate (Mid-market) | 5 tickets/yr | |
| Mature annual rate (Enterprise) | 12 tickets/yr | Named-account, more complex deployments |
| Mature annual rate (Large Enterprise) | 30 tickets/yr | |

Expected tickets = `annual_rate × (onboard_yrs × 2.5 + mature_yrs)` where `onboard_yrs = min(90/365.25, eligible_yrs)`.

## Telemetry — scan_success_rate

`scan_success_rate` in telemetry is only meaningful for **Prisma Cloud** products, where it measures the reliability of the cloud-scan process (degrades with low coverage or misconfiguration). For all other product families (Strata, SASE, Cortex), the field is `null` — it should not be aggregated or used as a feature for non-Prisma renewal units.

## Revenue Scale Note

PANW reported $9.10B NGS ARR in Q4 FY2026 [PUBLIC: FY2026 results]. NGS ARR covers next-generation security products (not hardware). This study's synthetic ACV ranges are declared simulation parameters chosen to produce plausible-looking dollar figures; they are not calibrated to PANW's actual revenue, pricing, or mix. The simulated total ARR is not a validated subset of $9.10B. Specific dollar figures in this study are synthetic assumptions and should not be interpreted as financial estimates of PANW's actual business.


## SA-010: Customer-management scope for the analysis

For the current analytical scenario, assume the customer-success team manages
every purchasing account. Original sales-route labels remain only in source
metadata for provenance. They do not enter active model inputs, review eligibility
or cohort groupings. This simplifies the case study without claiming that PANW
operates this way. Source transactions and simulated cancellation outcomes are
unchanged. See `part2/docs/customer_management_scope.md`.


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

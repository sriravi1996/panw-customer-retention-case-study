# Business Context

## Product Scope

This study covers a PANW-inspired B2B cybersecurity portfolio organized into three platform families:

| Platform | Product Family | SKUs in study |
|---|---|---|
| Strata (Network Security) | Next-Gen Firewall, SASE/Prisma Access | PA-Series HW, VM-Series, CN-Series, Prisma Access Mobile Users, Prisma Access Remote Networks, GlobalProtect |
| Prisma (Cloud Security) | Prisma Cloud, Cortex Cloud | Prisma Cloud CSPM, Prisma Cloud CWP, Prisma Cloud DSPM, Prisma Cloud AppSec, Cortex Cloud |
| Cortex (AI-Ops / SOC) | XDR, XSOAR, XSIAM, Xpanse | Cortex XDR, Cortex XSOAR, Cortex XSIAM, Cortex Xpanse |
| Support & Services | Support, PS | Premium Support, Enhanced Support, Professional Services |

Hardware (PA-Series) is a one-time purchase. All other families are recurring subscriptions. Support entitlements attach to hardware or software contracts.

## Buying Roles

| Role | Description |
|---|---|
| Economic Buyer | VP/CISO level; approves budget, signs contracts |
| Technical Buyer | Security Architect / Eng Manager; evaluates and recommends |
| Champion | Security practitioner who runs the day-to-day deployment |
| Administrator | Manages tenant, licenses, policies |
| End User | Security analyst consuming alerts, dashboards |

A single **user** record maps to one of these roles. Multiple users from the same company may place orders (e.g., procurement contact places the order; the champion is different).

## Deployment Ownership

- Each **deployment** is a tenant, environment, fleet, or protected workload scope.
- Deployments are owned by a **team** (e.g., Cloud Security, DevSecOps, Network Ops, SOC).
- A single order / contract may fund deployments used by **multiple teams** (e.g., an enterprise Prisma Access deal covers both the Network Ops and Remote Work teams).
- The **entitlement_deployment_map** table holds the allocation; recurring value is counted once at the contract level.

## Sales Channel

- **Direct**: Field AE + SE motion, enterprise accounts.
- **Channel / MSSP**: Reseller or MSSP places the invoice; end-customer is the **company**. The `channel` field records the category (direct / channel / digital) as account metadata; individual reseller identities are not modeled.
- **Digital / Self-serve**: SMB and mid-market; lower ACV, shorter terms.

## Commercial Renewal Unit

The unit of renewal analysis is a **contract_id** — a recurring entitlement with:
- One **product_id**
- One **company_id** (the end-customer)
- A defined **term_start / term_end**
- A **renewal_lineage_id** linking successive renewals of the same commitment

A contract may cover multiple teams and deployments. ARR is computed at contract level and never double-counted across teams.

## Observation Window

| Parameter | Value |
|---|---|
| observation_start | 2021-10-01 |
| observation_end | 2026-10-01 |
| scoring_date | 2026-10-01 |
| opening_base_start | 2018-10-01 (up to 3 years of pre-window purchase history) |
| lookback_90d | 2026-07-03 |
| lookback_180d | 2026-04-04 |
| outcome_window_end | 2026-12-30 (90 days post scoring) |
| renewal_grace_period | 30 days |

## Retention Definitions

| Term | Operational definition |
|---|---|
| Abandonment | Deployment activity or protection coverage falls below threshold for ≥ 30 consecutive days. Early warning; does not automatically trigger financial churn. |
| Product churn | A paid recurring entitlement reaches term_end and is not renewed (cancellation event). Measured via `contract_events` with `event_type = cancellation`. |
| Contraction | Renewal ACV < prior-period ACV (delta < 0); partial loss of recurring value. Classified separately from churn in the simulator. |
| Expansion | Renewal ACV > prior-period ACV, or a net-new entitlement is added. |
| Company churn | All paid recurring entitlements for a company are in churned state after the grace period and after migration checks (e.g., Prisma Cloud → Cortex Cloud upgrade). |

## Deployment State Taxonomy

`trial` → `active_paid` → `migrating` → `decommissioned` | `abandoned` | `outage`

- **trial**: Evaluation or POC; not counted in recurring revenue.
- **active_paid**: Funded entitlement with active telemetry.
- **migrating**: In-flight platform migration (e.g., Prisma Cloud to Cortex Cloud); legacy usage decline is expected.
- **decommissioned**: Planned end-of-life; contract not renewed by design.
- **abandoned**: Paid but no meaningful activity for ≥ 30 days; financial risk.
- **outage**: Telemetry gap; status unknown.

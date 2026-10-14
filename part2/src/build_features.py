"""
build_features.py — Per-contract renewal feature table builder (v2).

One row per eligible subscription contract at pred_date = term_end − 90 days.

All features are computed from raw source files at each contract's actual
pred_date — no re-labelled snapshot features.  Point-in-time integrity is
enforced for every feature block.

Feature sources
---------------
  Service (tickets)        : support_tickets_clean.csv     (contract_id FK)
  Relationship (CRM)       : account_interactions_clean.csv (company_id FK)
  Commercial (quotes)      : renewal_quotes_clean.csv       (contract_id FK)
  Telemetry / deployment   : telemetry_weekly_clean.csv     (deployment_id FK)
                             deployments_clean.csv          (renewal_lineage_id bridge)
  Structural               : company_summary.csv, order_item_fact.csv

Future contracts (pred_date > OBS_END = 2026-10-01) are separated into
part2/outputs/future_opportunities.csv and excluded from the main cohort.

decision_known_at_pred_date: True if a cancellation/renewal event with
event_date < pred_date already exists for this expiring_contract_id.
These rows are retained in labeled splits but excluded from the scoring queue.

Outputs
-------
  data/analysis/renewal_cohort_subscriptions.csv   — main cohort (≤ OBS_END)
  part2/outputs/future_opportunities.csv           — pred_date > OBS_END
  part2/outputs/cohort_counts.csv
  part2/outputs/exclusion_ledger.csv
  part2/outputs/build_features_summary.json
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent.parent
CLEANED   = ROOT / "data" / "cleaned"
ANALYSIS  = ROOT / "data" / "analysis"
PART2_OUT = ROOT / "part2" / "outputs"

ANALYSIS.mkdir(parents=True, exist_ok=True)
PART2_OUT.mkdir(parents=True, exist_ok=True)

OBS_START = pd.Timestamp("2021-10-01")
OBS_END   = pd.Timestamp("2026-10-01")

EXPECTED_RUN_ID = "82927c54-85b2-4dc1-aee9-dba8a32815ca"

print("=" * 70)
print("build_features.py v2 — raw-file feature computation at pred_date")
print("=" * 70)

# ---------------------------------------------------------------------------
# Step 0: Run provenance
# ---------------------------------------------------------------------------
qr_path = ROOT / "outputs" / "quality_report.json"
actual_run_id = None
if qr_path.exists():
    with open(qr_path) as _f:
        actual_run_id = json.load(_f).get("run_id", "")

print(f"\n[0] Provenance")
print(f"    Expected run_id : {EXPECTED_RUN_ID}")
print(f"    Actual run_id   : {actual_run_id}")
if actual_run_id != EXPECTED_RUN_ID:
    print(
        "    WARNING: run_id mismatch. Continuing — update EXPECTED_RUN_ID "
        "if this is a deliberate re-run.",
        file=sys.stderr,
    )
else:
    print("    ✓ run_id matches")

# ---------------------------------------------------------------------------
# Step 1: Load source files
# ---------------------------------------------------------------------------
print("\n[1] Loading source files ...")

oif = pd.read_csv(CLEANED / "order_item_fact.csv",           low_memory=False)
ce  = pd.read_csv(CLEANED / "contract_events_clean.csv",     low_memory=False)
cs  = pd.read_csv(CLEANED / "company_summary.csv",           low_memory=False)
dep = pd.read_csv(CLEANED / "deployments_clean.csv",         low_memory=False)
twc = pd.read_csv(CLEANED / "telemetry_weekly_clean.csv",    low_memory=False)
tix = pd.read_csv(CLEANED / "support_tickets_clean.csv",     low_memory=False)
ixn = pd.read_csv(CLEANED / "account_interactions_clean.csv",low_memory=False)
rq  = pd.read_csv(CLEANED / "renewal_quotes_clean.csv",      low_memory=False)

print(f"    order_item_fact              : {len(oif):>9,} rows")
print(f"    contract_events_clean        : {len(ce):>9,} rows")
print(f"    company_summary              : {len(cs):>9,} rows")
print(f"    deployments_clean            : {len(dep):>9,} rows")
print(f"    telemetry_weekly_clean       : {len(twc):>9,} rows")
print(f"    support_tickets_clean        : {len(tix):>9,} rows")
print(f"    account_interactions_clean   : {len(ixn):>9,} rows")
print(f"    renewal_quotes_clean         : {len(rq):>9,} rows")

# ---------------------------------------------------------------------------
# Step 1.5: Timestamp null audit (after coerce)
# ---------------------------------------------------------------------------
print("\n[1.5] Timestamp null audit ...")

# Parse timestamps now for audit (files will be re-parsed later as needed)
# Use format='mixed' to handle both naive and UTC-offset timestamps without coercion failures.
_tix_created  = pd.to_datetime(tix["created_at"],  format='mixed', errors="coerce")
_tix_orig_null_resolved = tix["resolved_at"].isna().sum()
_tix_resolved_parsed    = pd.to_datetime(tix["resolved_at"], format='mixed', errors="coerce")
_tix_parse_failures     = _tix_resolved_parsed.isna().sum() - _tix_orig_null_resolved
_tix_resolved           = _tix_resolved_parsed
_ixn_avail    = pd.to_datetime(ixn["available_at"],  errors="coerce")
_twc_week     = pd.to_datetime(twc["week_start"],    errors="coerce")
_twc_avail    = pd.to_datetime(twc["available_at"],  errors="coerce")
_dep_start    = pd.to_datetime(dep["deployment_start"], errors="coerce")

print(f"    Timestamp null audit:")
for name, _col, _ser in [
    ("support_tickets",         "created_at",    _tix_created),
    ("account_interactions",    "available_at",  _ixn_avail),
    ("telemetry_weekly",        "week_start",    _twc_week),
    ("telemetry_weekly",        "available_at",  _twc_avail),
    ("deployments",             "deployment_start", _dep_start),
]:
    print(f"      {name}.{_col}: {_ser.isna().sum()} nulls")
# resolved_at breakdown: originally null vs parse failure
_tix_resolved_ok = len(_tix_resolved) - _tix_resolved.isna().sum()
print(f"      support_tickets.resolved_at: {_tix_orig_null_resolved} originally null, "
      f"{_tix_parse_failures} parse failures, {_tix_resolved_ok} successfully parsed")

# ---------------------------------------------------------------------------
# Step 2: Build base contract table
# ---------------------------------------------------------------------------
print("\n[2] Building base contract table ...")

subs = oif[
    (oif["offering_type"] == "subscription") &
    (oif["term_end"].notna()) &
    (oif["acv"].fillna(0) > 0)
].copy()

subs["term_start"]  = pd.to_datetime(subs["term_start"],  errors="coerce")
subs["term_end"]    = pd.to_datetime(subs["term_end"],    errors="coerce")
subs["pred_date"]          = subs["term_end"] - timedelta(days=90)
subs["label_available_at"] = subs["term_end"] + timedelta(days=90)

# Eligibility: contract active at pred_date; pred_date within OBS window
mask = (subs["term_start"] < subs["pred_date"]) & (subs["pred_date"] >= OBS_START)
contracts = subs[mask].copy().reset_index(drop=True)

print(f"    Subscription contracts total          : {len(subs):>8,}")
print(f"    Eligible (pred_date >= OBS_START)     : {len(contracts):>8,}")

OIF_KEEP = [
    "contract_id", "renewal_lineage_id", "company_id", "product_id",
    "product_family", "offering_type", "term_start", "term_end",
    "pred_date", "label_available_at",
    "acv", "incomplete_history", "predecessor_contract_id",
]
contracts = contracts[[c for c in OIF_KEEP if c in contracts.columns]].copy()

# Separate future contracts
future = contracts[contracts["pred_date"] > OBS_END].copy()
contracts = contracts[contracts["pred_date"] <= OBS_END].copy().reset_index(drop=True)

print(f"    Within OBS window (pred_date ≤ {OBS_END.date()}) : {len(contracts):>8,}")
print(f"    Future (pred_date > {OBS_END.date()})          : {len(future):>8,}")

# ---------------------------------------------------------------------------
# Step 3: Labels
# ---------------------------------------------------------------------------
print("\n[3] Attaching labels ...")

ce["event_date"] = pd.to_datetime(ce["event_date"], errors="coerce")

cancel_ids = set(
    ce.loc[ce["event_type"] == "cancellation", "expiring_contract_id"].dropna()
)
print(f"    Cancellation events found             : {len(cancel_ids):>8,}")

contracts["label_matured"] = contracts["label_available_at"] <= OBS_END
contracts["churned"] = np.where(
    contracts["label_matured"],
    contracts["contract_id"].isin(cancel_ids).astype(float),
    np.nan,
)

labeled = contracts[contracts["label_matured"]]
print(f"    Labeled contracts                     : {len(labeled):>8,}")
print(f"    Positives (churned=1)                 : {int(labeled['churned'].sum()):>8,}")
print(f"    Cancellation rate                     : {labeled['churned'].mean():>8.3%}")

# ---------------------------------------------------------------------------
# Step 4: decision_known_at_pred_date
# ---------------------------------------------------------------------------
print("\n[4] Computing decision_known_at_pred_date ...")

# A decision is known if any renewal/cancellation event for this expiring
# contract was recorded before pred_date.
decided_events = ce[
    ce["event_type"].isin(["cancellation", "renewal", "renewal_expansion", "renewal_contraction"]) &
    ce["expiring_contract_id"].notna()
][["expiring_contract_id", "event_date"]].copy()

decided_events = decided_events.rename(columns={"expiring_contract_id": "contract_id"})

anchor_dk = contracts[["contract_id", "pred_date"]].merge(
    decided_events, on="contract_id", how="left"
)
anchor_dk["decided_before_pred"] = (
    anchor_dk["event_date"].notna() &
    (anchor_dk["event_date"] < anchor_dk["pred_date"])
)
dk_flag = (
    anchor_dk.groupby("contract_id")["decided_before_pred"]
    .any()
    .reset_index()
    .rename(columns={"decided_before_pred": "decision_known_at_pred_date"})
)
contracts = contracts.merge(dk_flag, on="contract_id", how="left")
contracts["decision_known_at_pred_date"] = (
    contracts["decision_known_at_pred_date"].fillna(False).astype(bool)
)
n_decided = int(contracts["decision_known_at_pred_date"].sum())
print(f"    Contracts with decision already known : {n_decided:>8,}")

# ---------------------------------------------------------------------------
# Step 5: Structural features from company_summary
# ---------------------------------------------------------------------------
print("\n[5] Attaching structural features ...")

cs_cols = ["company_id", "parent_org_id", "industry", "region",
           "size_segment", "employee_count", "first_purchase_date"]
cs_sub  = cs[[c for c in cs_cols if c in cs.columns]].drop_duplicates(subset=["company_id"])
cs_sub["first_purchase_date"] = pd.to_datetime(cs_sub["first_purchase_date"], errors="coerce")

contracts = contracts.merge(cs_sub, on="company_id", how="left")
contracts["tenure_days"] = (
    contracts["pred_date"] - contracts["first_purchase_date"]
).dt.days.clip(lower=0)
contracts["days_to_renewal"] = 90  # by construction: pred_date = term_end - 90d

# ---------------------------------------------------------------------------
# Step 6: Portfolio window features
#         n_products_in_window, n_families_in_window, total_acv_in_window
#         Window: co-expiring contracts from same company within 90 days of term_end
# ---------------------------------------------------------------------------
print("\n[6] Computing portfolio window features ...")

anchor = contracts[["contract_id", "company_id", "pred_date", "term_end"]].rename(columns={
    "contract_id": "anchor_id",
    "pred_date":   "anchor_pred",
    "term_end":    "anchor_term_end",
})
portfolio = contracts[["company_id", "product_id", "product_family", "acv", "term_end", "term_start"]].copy()

joined = anchor.merge(portfolio, on="company_id", how="left", suffixes=("", "_p"))
in_window = (
    (joined["term_end"]   >= joined["anchor_pred"]) &
    (joined["term_end"]   <= joined["anchor_term_end"]) &
    (joined["term_start"] <= joined["anchor_pred"])
)
win_agg = (
    joined[in_window]
    .groupby("anchor_id")
    .agg(
        n_products_in_window = ("product_id",     "nunique"),
        n_families_in_window = ("product_family", "nunique"),
        total_acv_in_window  = ("acv",            "sum"),
    )
    .reset_index()
    .rename(columns={"anchor_id": "contract_id"})
)
contracts = contracts.merge(win_agg, on="contract_id", how="left")
contracts["n_products_in_window"] = contracts["n_products_in_window"].fillna(1).astype(int)
contracts["n_families_in_window"] = contracts["n_families_in_window"].fillna(1).astype(int)
contracts["total_acv_in_window"]  = contracts["total_acv_in_window"].fillna(contracts["acv"])

# ---------------------------------------------------------------------------
# Step 7: Breadth features
#         n_active_products_in_family, family_adoption_completeness,
#         family_sibling_cancellations_1yr
# ---------------------------------------------------------------------------
print("\n[7] Computing breadth context features ...")

# n_active_products_in_family: count of active subscription products in same
# product_family at pred_date (for same company, term_start <= pred_date <= term_end)
sib = contracts[["company_id", "product_family", "product_id",
                  "term_start", "term_end"]].copy()
sib = sib.rename(columns={
    "product_family": "s_family",
    "product_id":     "s_product_id",
    "term_start":     "s_term_start",
    "term_end":       "s_term_end",
})

all_active = contracts[["contract_id", "company_id", "product_family",
                         "pred_date"]].copy()
all_active = all_active.rename(columns={
    "contract_id":   "a_id",
    "pred_date":     "a_pred",
    "product_family":"a_family",
})

joined_b = all_active.merge(sib, on="company_id", how="left")
active_same_fam = (
    (joined_b["s_family"]     == joined_b["a_family"]) &
    (joined_b["s_term_start"] <= joined_b["a_pred"]) &
    (joined_b["s_term_end"]   >= joined_b["a_pred"])
)
n_active = (
    joined_b[active_same_fam]
    .groupby("a_id")["s_product_id"]
    .nunique()
    .reset_index()
    .rename(columns={"a_id": "contract_id", "s_product_id": "n_active_products_in_family"})
)
contracts = contracts.merge(n_active, on="contract_id", how="left")
contracts["n_active_products_in_family"] = (
    contracts["n_active_products_in_family"].fillna(0).astype(int)
)

# family_adoption_completeness: how many distinct products in family does this
# company have (active at pred_date) / total catalog products in this family
catalog_per_family = (
    oif[oif["offering_type"] == "subscription"]
    .groupby("product_family")["product_id"]
    .nunique()
    .reset_index()
    .rename(columns={"product_id": "catalog_size"})
)
fac_numer = (
    joined_b[active_same_fam]
    .groupby("a_id")["s_product_id"]
    .nunique()
    .reset_index()
    .rename(columns={"a_id": "contract_id", "s_product_id": "active_prods_in_fam"})
)
contracts = contracts.merge(fac_numer, on="contract_id", how="left")
contracts["active_prods_in_fam"] = contracts["active_prods_in_fam"].fillna(0)
contracts = contracts.merge(catalog_per_family, on="product_family", how="left")
contracts["family_adoption_completeness"] = np.where(
    contracts["catalog_size"].fillna(0) > 0,
    (contracts["active_prods_in_fam"] / contracts["catalog_size"]).clip(0, 1),
    0.0,
)
contracts = contracts.drop(columns=["active_prods_in_fam", "catalog_size"])

# family_sibling_cancellations_1yr: cancellations of same-family products for
# same company in 12 months prior to pred_date
cancel_ev = ce[ce["event_type"] == "cancellation"][
    ["expiring_contract_id", "event_date", "company_id", "product_id"]
].copy()
cancel_ev["event_date"] = pd.to_datetime(cancel_ev["event_date"], errors="coerce")

sib_base = contracts[["contract_id", "company_id", "product_family", "pred_date"]].merge(
    cancel_ev, on="company_id", how="left"
)
sib_base = sib_base.merge(
    oif[["contract_id", "product_family"]].rename(columns={
        "contract_id": "expiring_contract_id",
        "product_family": "exp_family",
    }),
    on="expiring_contract_id", how="left"
)
sib_cancel_in_1yr = (
    (sib_base["exp_family"] == sib_base["product_family"]) &
    (sib_base["event_date"] >= sib_base["pred_date"] - timedelta(days=365)) &
    (sib_base["event_date"] <  sib_base["pred_date"])
)
fsc = (
    sib_base[sib_cancel_in_1yr]
    .groupby("contract_id")["expiring_contract_id"]
    .count()
    .reset_index()
    .rename(columns={"expiring_contract_id": "family_sibling_cancellations_1yr"})
)
contracts = contracts.merge(fsc, on="contract_id", how="left")
contracts["family_sibling_cancellations_1yr"] = (
    contracts["family_sibling_cancellations_1yr"].fillna(0).astype(int)
)

# ---------------------------------------------------------------------------
# Step 8: Service features from support_tickets_clean
#         Point-in-time reconstruction at pred_date
# ---------------------------------------------------------------------------
print("\n[8] Computing service features from support_tickets ...")

tix["created_at"]  = pd.to_datetime(tix["created_at"],  format='mixed', errors="coerce")
# Track original nulls to distinguish missing values from parse failures.
_s8_orig_null = tix["resolved_at"].isna().sum()
tix["resolved_at"] = pd.to_datetime(tix["resolved_at"], format='mixed', errors="coerce")
_s8_parse_fail = tix["resolved_at"].isna().sum() - _s8_orig_null
_s8_parsed_ok  = tix["resolved_at"].notna().sum()
print(f"    resolved_at: {_s8_orig_null} originally null, "
      f"{_s8_parse_fail} parse failures (should be 0), {_s8_parsed_ok} parsed OK")

anchor_s = contracts[["contract_id", "pred_date"]].copy()
st = anchor_s.merge(tix, on="contract_id", how="left")

# Point-in-time: only tickets created before pred_date
st_pit = st[st["created_at"].isna() | (st["created_at"] < st["pred_date"])].copy()

# 90-day window tickets (created in [pred_date-90d, pred_date))
in_90d = (
    st_pit["created_at"].notna() &
    (st_pit["created_at"] >= (st_pit["pred_date"] - timedelta(days=90)))
)
st_90 = st_pit[in_90d].copy()

# tickets_90d and p1_tickets_90d
svc_agg = (
    st_90.groupby("contract_id")
    .agg(
        tickets_90d    = ("ticket_id", "count"),
        p1_tickets_90d = ("priority",  lambda x: (x == "P1").sum()),
    )
    .reset_index()
)

# repeat_sla_breaches_90d (point-in-time): SLA breach at pred_date means
# (created_at + sla_target_hours) <= pred_date AND ticket still open at pred_date
def _sla_breach_at_pred(df: pd.DataFrame) -> pd.Series:
    """True if SLA target has elapsed AND ticket is still open at pred_date."""
    sla_hrs = pd.to_numeric(df["sla_target_hours"], errors="coerce").fillna(0)
    sla_deadline = df["created_at"] + sla_hrs.apply(lambda h: timedelta(hours=h))
    breach_elapsed = sla_deadline <= df["pred_date"]
    still_open = df["resolved_at"].isna() | (df["resolved_at"] > df["pred_date"])
    return (breach_elapsed & still_open).astype(int)

st_90["_sla_pit"] = _sla_breach_at_pred(st_90)
sla_agg = (
    st_90.groupby("contract_id")["_sla_pit"]
    .sum()
    .reset_index()
    .rename(columns={"_sla_pit": "repeat_sla_breaches_90d"})
)
svc_agg = svc_agg.merge(sla_agg, on="contract_id", how="left")
svc_agg["repeat_sla_breaches_90d"] = svc_agg["repeat_sla_breaches_90d"].fillna(0).astype(int)

# avg_resolution_hrs_90d: use only tickets resolved BEFORE pred_date
# (resolved_at < pred_date), clip resolution_hours to 90*24
res_90 = st_90[
    st_90["resolved_at"].notna() &
    (st_90["resolved_at"] < st_90["pred_date"])
].copy()
res_90["resolution_hours_clipped"] = (
    pd.to_numeric(res_90["resolution_hours"], errors="coerce")
    .clip(upper=90 * 24)
)
res_agg = (
    res_90.groupby("contract_id")["resolution_hours_clipped"]
    .mean()
    .reset_index()
    .rename(columns={"resolution_hours_clipped": "avg_resolution_hrs_90d"})
)
svc_agg = svc_agg.merge(res_agg, on="contract_id", how="left")
svc_agg["avg_resolution_hrs_90d_missing"] = (
    svc_agg["avg_resolution_hrs_90d"].isna().astype(int)
)
svc_agg["avg_resolution_hrs_90d"] = svc_agg["avg_resolution_hrs_90d"].fillna(0.0)

# open_tickets_90d: ticket is open at pred_date if
#   created_at <= pred_date AND (resolved_at is null OR resolved_at > pred_date)
# (already filtered st_pit to created_at < pred_date, so created_at <= pred_date)
open_mask = (
    in_90d &
    (st_pit["resolved_at"].isna() | (st_pit["resolved_at"] > st_pit["pred_date"]))
)
open_90 = st_pit[open_mask].copy()
open_90["open_age_hrs"] = (
    (open_90["pred_date"] - open_90["created_at"]).dt.total_seconds() / 3600
).clip(lower=0)

open_agg = (
    open_90.groupby("contract_id")
    .agg(
        open_tickets_90d     = ("ticket_id",    "count"),
        avg_open_age_hrs_90d = ("open_age_hrs", "mean"),
    )
    .reset_index()
)
svc_agg = svc_agg.merge(open_agg, on="contract_id", how="left")
svc_agg["open_tickets_90d"]     = svc_agg["open_tickets_90d"].fillna(0).astype(int)
svc_agg["avg_open_age_hrs_90d"] = svc_agg["avg_open_age_hrs_90d"].fillna(0.0)

# unresolved_customer_impact_hours_90d: point-in-time impact hours
# Only customer_impact=True tickets that are open at pred_date in the 90d window
# hours = max(0, (pred_date - impact_start_at)) capped at pred_date
uci = open_90[open_90["customer_impact"].fillna(False).astype(str).str.lower() == "true"].copy()
uci["impact_start_at"] = pd.to_datetime(uci.get("impact_start_at", pd.NaT), errors="coerce")
# Only count impact starting before pred_date
uci_valid = uci[uci["impact_start_at"].notna() & (uci["impact_start_at"] <= uci["pred_date"])].copy()
uci_valid["impact_hrs_pit"] = (
    (uci_valid["pred_date"] - uci_valid["impact_start_at"]).dt.total_seconds() / 3600
).clip(lower=0)
uci_agg = (
    uci_valid.groupby("contract_id")["impact_hrs_pit"]
    .sum()
    .reset_index()
    .rename(columns={"impact_hrs_pit": "unresolved_customer_impact_hours_90d"})
)
svc_agg = svc_agg.merge(uci_agg, on="contract_id", how="left")
svc_agg["unresolved_customer_impact_hours_90d"] = (
    svc_agg["unresolved_customer_impact_hours_90d"].fillna(0.0)
)

contracts = contracts.merge(svc_agg, on="contract_id", how="left")
for col in ["tickets_90d", "p1_tickets_90d", "repeat_sla_breaches_90d",
            "open_tickets_90d"]:
    contracts[col] = contracts[col].fillna(0).astype(int)
for col in ["avg_resolution_hrs_90d", "unresolved_customer_impact_hours_90d",
            "avg_open_age_hrs_90d"]:
    contracts[col] = contracts[col].fillna(0.0)
contracts["avg_resolution_hrs_90d_missing"] = (
    contracts["avg_resolution_hrs_90d_missing"].fillna(1).astype(int)
)

n_svc = (contracts["tickets_90d"] > 0).sum()
print(f"    Contracts with ≥1 ticket in 90d window : {n_svc:>8,}")

# ---------------------------------------------------------------------------
# Step 9: Relationship features from account_interactions_clean
#         Point-in-time: available_at <= pred_date
# ---------------------------------------------------------------------------
print("\n[9] Computing relationship features from account_interactions ...")

ixn["available_at"]             = pd.to_datetime(ixn["available_at"],             errors="coerce")
ixn["event_at"]                 = pd.to_datetime(ixn["event_at"],                 errors="coerce")
ixn["agreed_action_due_date"]   = pd.to_datetime(ixn["agreed_action_due_date"],   errors="coerce")
ixn["agreed_action_completed_at"] = pd.to_datetime(ixn["agreed_action_completed_at"], errors="coerce")

anchor_r = contracts[["contract_id", "company_id", "pred_date"]].copy()
ri = anchor_r.merge(ixn, on="company_id", how="left")

# Point-in-time: available_at <= pred_date
ri_pit = ri[ri["available_at"].isna() | (ri["available_at"] <= ri["pred_date"])]

# 180d window for executive / marketing
in_180d = ri_pit["event_at"] >= (ri_pit["pred_date"] - timedelta(days=180))
ri_180  = ri_pit[in_180d].copy()

exec_agg = (
    ri_180.groupby("contract_id")
    .agg(
        executive_interactions_180d = ("interaction_type",
            lambda x: (x == "executive_briefing").sum()),
        marketing_engagements_180d  = ("interaction_type",
            lambda x: (x == "marketing_event").sum()),
    )
    .reset_index()
)

# days_since_last_interaction (any interaction at available_at <= pred_date)
last_ix = (
    ri_pit[ri_pit["available_at"].notna()]
    .groupby("contract_id")["available_at"]
    .max()
    .reset_index()
    .rename(columns={"available_at": "last_interaction_at"})
)
contracts = contracts.merge(last_ix, on="contract_id", how="left")
contracts["days_since_last_interaction"] = (
    contracts["pred_date"] - contracts["last_interaction_at"]
).dt.days.clip(lower=0)
contracts["days_since_last_interaction_missing"] = (
    contracts["last_interaction_at"].isna().astype(int)
)
contracts = contracts.drop(columns=["last_interaction_at"])

# dated_budget_concern: exact match on interaction_type == "budget_concern"
budget_mask = (ri_pit["interaction_type"] == "budget_concern")
bc_agg = (
    ri_pit[budget_mask]
    .groupby("contract_id")["interaction_id"]
    .count()
    .gt(0)
    .astype(int)
    .reset_index()
    .rename(columns={"interaction_id": "dated_budget_concern"})
)
contracts = contracts.merge(bc_agg, on="contract_id", how="left")
contracts["dated_budget_concern"] = contracts["dated_budget_concern"].fillna(0).astype(int)

# days_without_confirmed_owner: days since last interaction where owner_role is not null
# (a confirmed owner was present), with available_at <= pred_date
owner_events = ri_pit[ri_pit["owner_role"].notna() & ri_pit["available_at"].notna()]
last_owner = (
    owner_events.groupby("contract_id")["available_at"]
    .max()
    .reset_index()
    .rename(columns={"available_at": "last_owner_at"})
)
contracts = contracts.merge(last_owner, on="contract_id", how="left")
contracts["days_without_confirmed_owner"] = (
    contracts["pred_date"] - contracts["last_owner_at"]
).dt.days.clip(lower=0).fillna(0.0)
contracts = contracts.drop(columns=["last_owner_at"])

# overdue_agreed_actions: count of agreed actions past due at pred_date
overdue_mask = (
    ri_pit["agreed_action_due_date"].notna() &
    (ri_pit["agreed_action_due_date"] < ri_pit["pred_date"]) &
    ri_pit["agreed_action_completed_at"].isna()
)
od_agg = (
    ri_pit[overdue_mask]
    .groupby("contract_id")["agreed_action_id"]
    .count()
    .reset_index()
    .rename(columns={"agreed_action_id": "overdue_agreed_actions"})
)
contracts = contracts.merge(od_agg, on="contract_id", how="left")
contracts["overdue_agreed_actions"] = contracts["overdue_agreed_actions"].fillna(0).astype(int)

# median_prior_renewal_delay_days: direct predecessor linkage via predecessor_contract_id.
# Using the lineage-chain approach inflates delay through multi-hop renewals (up to 1,141d).
# predecessor_contract_id gives a single-hop delay (mean ~15d, max ~29d in this dataset).
_pred_lookup = oif[["contract_id", "term_end"]].copy()
_pred_lookup["term_end"] = pd.to_datetime(_pred_lookup["term_end"], errors="coerce")
_pred_lookup = _pred_lookup.rename(columns={
    "contract_id": "predecessor_contract_id",
    "term_end":    "pred_term_end",
})

if "predecessor_contract_id" in oif.columns:
    _anchor_rd = contracts[["contract_id", "predecessor_contract_id", "term_start"]].copy()
    _anchor_rd["term_start"] = pd.to_datetime(_anchor_rd["term_start"], errors="coerce")
    _anchor_rd = _anchor_rd.merge(_pred_lookup, on="predecessor_contract_id", how="left")
    _anchor_rd["renewal_delay"] = (_anchor_rd["term_start"] - _anchor_rd["pred_term_end"]).dt.days
    # Only keep non-negative delays (negative = data error; cap to exclude outliers)
    _valid_delay = _anchor_rd[
        _anchor_rd["renewal_delay"].notna() & (_anchor_rd["renewal_delay"] >= 0)
    ][["contract_id", "renewal_delay"]]
    rd_agg = _valid_delay.rename(columns={"renewal_delay": "median_prior_renewal_delay_days"})
    _delay_max = rd_agg["median_prior_renewal_delay_days"].max()
    _delay_mean = rd_agg["median_prior_renewal_delay_days"].mean()
    print(f"    Renewal delay via predecessor_contract_id: mean={_delay_mean:.1f}d, max={_delay_max:.0f}d")
else:
    # Fallback: no predecessor linkage available — fill with missing indicator
    rd_agg = pd.DataFrame(columns=["contract_id", "median_prior_renewal_delay_days"])
    print("    WARNING: predecessor_contract_id not found in order_item_fact — delay set to missing")

contracts = contracts.merge(rd_agg, on="contract_id", how="left")
contracts["median_prior_renewal_delay_missing"] = (
    contracts["median_prior_renewal_delay_days"].isna().astype(int)
)
contracts["median_prior_renewal_delay_days"] = (
    contracts["median_prior_renewal_delay_days"].fillna(0.0)
)

# Merge exec/marketing after delay computation
contracts = contracts.merge(exec_agg, on="contract_id", how="left")
contracts["executive_interactions_180d"] = (
    contracts["executive_interactions_180d"].fillna(0).astype(int)
)
contracts["marketing_engagements_180d"] = (
    contracts["marketing_engagements_180d"].fillna(0).astype(int)
)

n_rel = (contracts["days_since_last_interaction"].notna() &
         (contracts["days_since_last_interaction_missing"] == 0)).sum()
print(f"    Contracts with CRM interaction record  : {n_rel:>8,}")

# ---------------------------------------------------------------------------
# Step 10: Commercial features from renewal_quotes_clean
#          Point-in-time: available_at <= pred_date; take latest quote
# ---------------------------------------------------------------------------
print("\n[10] Computing commercial features from renewal_quotes ...")

rq["available_at"] = pd.to_datetime(rq["available_at"], errors="coerce")

anchor_c = contracts[["contract_id", "pred_date"]].copy()
rq_merge = anchor_c.merge(rq, on="contract_id", how="left")
rq_pit   = rq_merge[
    rq_merge["available_at"].isna() | (rq_merge["available_at"] <= rq_merge["pred_date"])
]

# Take the most recent quote per contract
latest_quote = (
    rq_pit.sort_values("available_at")
    .groupby("contract_id")
    .last()
    .reset_index()[["contract_id", "unit_price_change_pct"]]
    .rename(columns={"unit_price_change_pct": "quoted_unit_price_change_pct"})
)
contracts = contracts.merge(latest_quote, on="contract_id", how="left")
contracts["quoted_unit_price_change_pct"] = (
    contracts["quoted_unit_price_change_pct"].fillna(0.0)
)

n_qp = contracts["quoted_unit_price_change_pct"].ne(0.0).sum()
print(f"    Contracts with a quote record          : {n_qp:>8,}")

# ---------------------------------------------------------------------------
# Step 11: Telemetry + deployment features
#          Bridge: telemetry_weekly_clean → deployments_clean → renewal_lineage_id
#          → contracts
#          Window: week_start in [pred_date - 90d, pred_date)
#          Point-in-time: available_at <= pred_date
# ---------------------------------------------------------------------------
print("\n[11] Computing telemetry & deployment features ...")

twc["week_start"]   = pd.to_datetime(twc["week_start"],   errors="coerce")
twc["available_at"] = pd.to_datetime(twc["available_at"], errors="coerce")
dep["deployment_start"]             = pd.to_datetime(dep["deployment_start"],             errors="coerce")
dep["planned_first_value_date"]     = pd.to_datetime(dep["planned_first_value_date"],     errors="coerce")
dep["first_value_achieved_date"]    = pd.to_datetime(dep["first_value_achieved_date"],    errors="coerce")

# Merge telemetry with deployment metadata to get renewal_lineage_id
dep_meta = dep[["deployment_id", "renewal_lineage_id", "company_id",
                 "product_id", "product_family", "eligible_units",
                 "deployment_start",
                 "planned_first_value_date", "first_value_achieved_date"]].copy()
dep_meta = dep_meta.rename(columns={"eligible_units": "dep_eligible_units"})

twc_dep = twc.merge(dep_meta, on="deployment_id", how="left")

# Join via renewal_lineage_id to contracts
anchor_t = contracts[["contract_id", "renewal_lineage_id", "pred_date", "product_family"]].copy()
tm = anchor_t.merge(twc_dep, on="renewal_lineage_id", how="left")

# Point-in-time: available_at <= pred_date
tm_pit = tm[tm["available_at"].isna() | (tm["available_at"] <= tm["pred_date"])]

# 90-day window ending at pred_date
tm_90 = tm_pit[
    tm_pit["week_start"].notna() &
    (tm_pit["week_start"] >= tm_pit["pred_date"] - timedelta(days=90)) &
    (tm_pit["week_start"] <  tm_pit["pred_date"])
].copy()

# 30-day window for recent metrics
tm_30 = tm_90[tm_90["week_start"] >= (tm_90["pred_date"] - timedelta(days=30))].copy()
# Prior 60d-to-30d window for delta
tm_60_30 = tm_90[
    (tm_90["week_start"] >= (tm_90["pred_date"] - timedelta(days=60))) &
    (tm_90["week_start"] <  (tm_90["pred_date"] - timedelta(days=30)))
].copy()

# avg_coverage_90d: mean weekly coverage_pct over 90 days
# coverage_trend_90d: linear slope of weekly coverage
def _slope(series: pd.Series) -> float:
    """OLS slope, or 0 if fewer than 2 distinct x-values."""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series), dtype=float)
    y = series.values.astype(float)
    if np.std(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])

# Sort chronologically before applying trend slope
tm_90 = tm_90.sort_values(["contract_id", "week_start"])

cov_agg = (
    tm_90.groupby("contract_id")
    .agg(
        avg_coverage_90d      = ("coverage_pct",      "mean"),
        coverage_trend_90d    = ("coverage_pct",      _slope),
        obs_weeks             = ("week_start",         "count"),
        policy_events_sum     = ("policy_events",      "sum"),
        eligible_sum          = ("dep_eligible_units", "sum"),
        remediation_sum       = ("remediation_events", "sum"),
        automation_sum        = ("automation_events",  "sum"),
        deployment_start_min  = ("deployment_start",   "min"),
    )
    .reset_index()
)
# obs_completeness_90d = observed_weeks / expected_weeks
# expected_weeks = min(13, max(1, floor((pred_date - deployment_start) / 7 days)))
# Join pred_date from anchor_t
cov_agg = cov_agg.merge(
    anchor_t[["contract_id", "pred_date"]].drop_duplicates("contract_id"),
    on="contract_id", how="left"
)
_days_deployed = (cov_agg["pred_date"] - cov_agg["deployment_start_min"]).dt.days.fillna(0)
cov_agg["expected_weeks"] = (_days_deployed / 7).apply(np.floor).clip(1, 13).astype(int)
cov_agg["obs_completeness_90d"] = (cov_agg["obs_weeks"] / cov_agg["expected_weeks"]).clip(0, 1)
cov_agg = cov_agg.drop(columns=["deployment_start_min", "pred_date", "expected_weeks"])

# policy_events_per_unit_90d
cov_agg["policy_events_per_unit_90d"] = np.where(
    cov_agg["eligible_sum"] > 0,
    cov_agg["policy_events_sum"] / cov_agg["eligible_sum"],
    0.0,
)

# remediation_velocity_90d: mean remediations per week (use obs_weeks as denominator)
cov_agg["remediation_velocity_90d"] = np.where(
    cov_agg["obs_weeks"] > 0,
    cov_agg["remediation_sum"] / cov_agg["obs_weeks"],
    0.0,
)

# automation_adoption_90d: fraction of weeks with ≥1 automation event
# Need per-week flag — recompute from tm_90
auto_weeks = (
    tm_90.assign(has_auto=(tm_90["automation_events"] >= 1))
    .groupby("contract_id")["has_auto"]
    .mean()
    .reset_index()
    .rename(columns={"has_auto": "automation_adoption_90d"})
)
cov_agg = cov_agg.merge(auto_weeks, on="contract_id", how="left")
cov_agg["automation_adoption_90d"] = cov_agg["automation_adoption_90d"].fillna(0.0)

# scan_success_rate_90d: Prisma Cloud only; mean of non-null scan_success_rate weeks
scan_agg = (
    tm_90[tm_90["scan_success_rate"].notna()]
    .groupby("contract_id")["scan_success_rate"]
    .mean()
    .reset_index()
    .rename(columns={"scan_success_rate": "scan_success_rate_90d"})
)
cov_agg = cov_agg.merge(scan_agg, on="contract_id", how="left")
# scan_success_rate_90d: keep NaN for Prisma (missing data → ML fills with median)
# For non-Prisma the metric is inapplicable → fill 1.0 (no penalty)
# Join product_family from anchor_t to distinguish
cov_agg = cov_agg.merge(
    anchor_t[["contract_id", "product_family"]].drop_duplicates("contract_id"),
    on="contract_id", how="left"
)
cov_agg["scan_success_rate_90d"] = np.where(
    cov_agg["product_family"] == "Prisma",
    cov_agg["scan_success_rate_90d"],   # keep NaN for Prisma with no data
    np.where(cov_agg["scan_success_rate_90d"].isna(), 1.0, cov_agg["scan_success_rate_90d"])
)
cov_agg = cov_agg.drop(columns=["product_family"])

# workflow_success_30d (30-day window)
wf30_agg = (
    tm_30.groupby("contract_id")
    .agg(wf_attempts=("workflow_attempts", "sum"), wf_successes=("workflow_successes", "sum"))
    .reset_index()
)
wf30_agg["workflow_success_30d"] = np.where(
    wf30_agg["wf_attempts"] > 0,
    wf30_agg["wf_successes"] / wf30_agg["wf_attempts"],
    np.nan,  # undefined ratio → neutral fill 0.5
)
wf30_agg["workflow_success_30d_missing"] = wf30_agg["wf_attempts"].eq(0).astype(int)
wf30_agg = wf30_agg.drop(columns=["wf_attempts", "wf_successes"])

# productive_use_delta_30d: recent_30d rate minus prior_60-to-30d rate
wf_prior_agg = (
    tm_60_30.groupby("contract_id")
    .agg(wf_attempts_p=("workflow_attempts", "sum"), wf_successes_p=("workflow_successes", "sum"))
    .reset_index()
)
wf_prior_agg["wf_rate_prior"] = np.where(
    wf_prior_agg["wf_attempts_p"] > 0,
    wf_prior_agg["wf_successes_p"] / wf_prior_agg["wf_attempts_p"],
    np.nan,
)
wf_recent_agg = (
    tm_30.groupby("contract_id")
    .agg(wf_attempts_r=("workflow_attempts", "sum"), wf_successes_r=("workflow_successes", "sum"))
    .reset_index()
)
wf_recent_agg["wf_rate_recent"] = np.where(
    wf_recent_agg["wf_attempts_r"] > 0,
    wf_recent_agg["wf_successes_r"] / wf_recent_agg["wf_attempts_r"],
    np.nan,
)
wf_delta = wf_recent_agg[["contract_id", "wf_rate_recent", "wf_attempts_r"]].merge(
    wf_prior_agg[["contract_id", "wf_rate_prior"]], on="contract_id", how="outer"
)
wf_delta["productive_use_delta_30d"] = np.where(
    wf_delta["wf_rate_recent"].notna() & wf_delta["wf_rate_prior"].notna(),
    wf_delta["wf_rate_recent"] - wf_delta["wf_rate_prior"],
    0.0,
)
wf_delta = wf_delta[["contract_id", "productive_use_delta_30d"]]

# deployment_reach_30d: fraction of scope with active deployment in 30d
dr30_agg = (
    tm_30.groupby("contract_id")
    .agg(dep_targets=("deployed_targets", "sum"), scope_targets=("scope_targets", "sum"))
    .reset_index()
)
dr30_agg["deployment_reach_30d"] = np.where(
    dr30_agg["scope_targets"] > 0,
    (dr30_agg["dep_targets"] / dr30_agg["scope_targets"]).clip(0, 1),
    np.nan,
)
dr30_agg["deployment_reach_30d_missing"] = dr30_agg["scope_targets"].eq(0).astype(int)
dr30_agg = dr30_agg.drop(columns=["dep_targets", "scope_targets"])

# Merge all telemetry aggregates
telemetry_cols = cov_agg.drop(columns=["obs_weeks", "policy_events_sum", "eligible_sum",
                                        "remediation_sum", "automation_sum"], errors="ignore")
contracts = contracts.merge(telemetry_cols, on="contract_id", how="left")
contracts = contracts.merge(wf30_agg,       on="contract_id", how="left")
contracts = contracts.merge(wf_delta,       on="contract_id", how="left")
contracts = contracts.merge(dr30_agg,       on="contract_id", how="left")

# Fill telemetry NaN per feature_contract.md
contracts["avg_coverage_90d"]   = contracts["avg_coverage_90d"].fillna(0.5)    # unknown → neutral
contracts["coverage_trend_90d"] = contracts["coverage_trend_90d"].fillna(0.0)
contracts["obs_completeness_90d"] = contracts["obs_completeness_90d"].fillna(0.5)
contracts["policy_events_per_unit_90d"] = contracts["policy_events_per_unit_90d"].fillna(0.0)
contracts["remediation_velocity_90d"]   = contracts["remediation_velocity_90d"].fillna(0.0)
contracts["automation_adoption_90d"]    = contracts["automation_adoption_90d"].fillna(0.0)
# scan_success_rate_90d: NaN retained for Prisma (filled by ML median); non-Prisma already set to 1.0
# After merge, any remaining NaN means contract had no telemetry at all: apply same rule
contracts["scan_success_rate_90d"] = np.where(
    contracts["product_family"] == "Prisma",
    contracts["scan_success_rate_90d"],  # keep NaN for Prisma → ML median fill
    contracts["scan_success_rate_90d"].fillna(1.0),
)
contracts["workflow_success_30d"] = contracts["workflow_success_30d"].fillna(0.5)  # neutral
contracts["workflow_success_30d_missing"] = (
    contracts["workflow_success_30d_missing"].fillna(1).astype(int)
)
contracts["productive_use_delta_30d"] = contracts["productive_use_delta_30d"].fillna(0.0)
contracts["deployment_reach_30d"]     = contracts["deployment_reach_30d"].fillna(0.0)
contracts["deployment_reach_30d_missing"] = (
    contracts["deployment_reach_30d_missing"].fillna(1).astype(int)
)

n_tel = (contracts["obs_completeness_90d"] > 0.5).sum()
print(f"    Contracts with telemetry data          : {n_tel:>8,}")

# ---------------------------------------------------------------------------
# Step 12: Onboarding delay from deployments_clean
#          onboarding_delay_days = max(0, pred_date - planned_first_value_date)
#          if milestone not yet achieved at pred_date; 0 if achieved or no milestone
# ---------------------------------------------------------------------------
print("\n[12] Computing onboarding delay ...")

# One deployment per contract: most recent deployment_start that is <= pred_date
dep_sorted = dep.sort_values("deployment_start", ascending=False)

# Join all deployments to contracts on renewal_lineage_id, then filter point-in-time
od_merge_all = contracts[["contract_id", "renewal_lineage_id", "pred_date"]].merge(
    dep_sorted[["renewal_lineage_id", "deployment_start",
                "planned_first_value_date", "first_value_achieved_date"]],
    on="renewal_lineage_id", how="left"
)
# Filter to deployments that started at or before pred_date
od_merge_pit = od_merge_all[
    od_merge_all["deployment_start"].isna() |
    (od_merge_all["deployment_start"] <= od_merge_all["pred_date"])
]
# Take most recent deployment per contract (already sorted descending)
od_merge = (
    od_merge_pit
    .sort_values("deployment_start", ascending=False)
    .drop_duplicates(subset=["contract_id"])
    .reset_index(drop=True)
)

# Delay = days past planned milestone at pred_date, if milestone not yet achieved
# If achieved before pred_date → delay = 0
achieved_before = (
    od_merge["first_value_achieved_date"].notna() &
    (od_merge["first_value_achieved_date"] <= od_merge["pred_date"])
)
planned_past = (
    od_merge["planned_first_value_date"].notna() &
    (od_merge["planned_first_value_date"] < od_merge["pred_date"])
)

od_merge["onboarding_delay_days"] = np.where(
    achieved_before,
    0.0,
    np.where(
        planned_past,
        (od_merge["pred_date"] - od_merge["planned_first_value_date"]).dt.days.clip(lower=0),
        0.0,
    )
)

od_out = od_merge[["contract_id", "onboarding_delay_days"]]
contracts = contracts.merge(od_out, on="contract_id", how="left")
contracts["onboarding_delay_days"] = contracts["onboarding_delay_days"].fillna(0.0)

# ---------------------------------------------------------------------------
# Step 13: Gates
# ---------------------------------------------------------------------------
print("\n[13] Running integrity gates ...")

n_null_id = contracts["contract_id"].isna().sum()
if n_null_id > 0:
    print(f"  GATE FAIL: {n_null_id} null contract_ids.", file=sys.stderr)
    sys.exit(1)

n_dup = contracts["contract_id"].duplicated().sum()
if n_dup > 0:
    print(f"  GATE FAIL: {n_dup} duplicate contract_ids.", file=sys.stderr)
    sys.exit(1)

PROHIBITED = [
    "channel", "sales_channel",  # excluded by the directly managed case-study assumption
    "churn_prob", "true_churn_probability", "deployment_archetype",
    "experience_state", "missing_telemetry_pct_90d",
    "accepted_price", "renewal_accepted", "successor_contract_id",
    "as_of_date",
]
leaked = [c for c in PROHIBITED if c in contracts.columns]
if leaked:
    contracts = contracts.drop(columns=leaked)
    print(f"    Dropped prohibited columns: {leaked}")

print("    ✓ No null contract_ids, no duplicates, no prohibited columns")

# ---------------------------------------------------------------------------
# Step 14: Save outputs
# ---------------------------------------------------------------------------
print("\n[14] Saving outputs ...")

for col in ["pred_date", "term_start", "term_end", "label_available_at"]:
    if col in contracts.columns and pd.api.types.is_datetime64_any_dtype(contracts[col]):
        contracts[col] = contracts[col].dt.strftime("%Y-%m-%d")

contracts.to_csv(ANALYSIS / "renewal_cohort_subscriptions.csv", index=False)
print(f"    data/analysis/renewal_cohort_subscriptions.csv — {len(contracts):,} rows")
# Compatibility alias for downstream reporting. It has the same per-contract
# prediction-date grain; it is not an October operating snapshot.
contracts.to_csv(ANALYSIS / "renewal_cohort.csv", index=False)
print(f"    data/analysis/renewal_cohort.csv               — current per-contract alias")

# Future opportunities
for col in ["pred_date", "term_start", "term_end", "label_available_at"]:
    if col in future.columns and pd.api.types.is_datetime64_any_dtype(future[col]):
        future[col] = future[col].dt.strftime("%Y-%m-%d")
future.to_csv(PART2_OUT / "future_opportunities.csv", index=False)
print(f"    part2/outputs/future_opportunities.csv          — {len(future):,} rows")

# Exclusion ledger (feature coverage)
coverage_cols = [
    "avg_coverage_90d", "workflow_success_30d", "tickets_90d",
    "days_since_last_interaction", "quoted_unit_price_change_pct",
    "deployment_reach_30d",
]
contracts_tmp = contracts.copy()
contracts_tmp["pred_date_dt"] = pd.to_datetime(contracts_tmp["pred_date"], errors="coerce")
train_mask  = contracts_tmp["pred_date_dt"] < pd.Timestamp("2023-10-04")
score_mask  = (
    contracts_tmp["label_matured"].astype(bool) == False
) if "label_matured" in contracts_tmp.columns else pd.Series(False, index=contracts_tmp.index)

ledger_rows = []
for col in coverage_cols:
    if col in contracts.columns:
        n_tot  = int(contracts[col].notna().sum())
        n_tr   = int(contracts.loc[train_mask, col].notna().sum()) if train_mask.sum() > 0 else 0
        n_sc   = int(contracts.loc[score_mask, col].notna().sum()) if score_mask.sum() > 0 else 0
        ledger_rows.append({
            "feature":            col,
            "n_present_total":    n_tot,
            "pct_present_total":  round(n_tot / max(len(contracts), 1), 4),
            "n_present_train":    n_tr,
            "pct_present_train":  round(n_tr  / max(int(train_mask.sum()), 1), 4),
            "n_present_scoring":  n_sc,
            "pct_present_scoring":round(n_sc  / max(int(score_mask.sum()), 1), 4),
        })
pd.DataFrame(ledger_rows).to_csv(PART2_OUT / "exclusion_ledger.csv", index=False)
print(f"    part2/outputs/exclusion_ledger.csv")

# Cohort counts by quarter
contracts_tmp["pred_quarter"] = contracts_tmp["pred_date_dt"].dt.to_period("Q").astype(str)
cohort_counts = (
    contracts_tmp.groupby("pred_quarter")
    .agg(
        n_rows      = ("contract_id",   "count"),
        n_labeled   = ("label_matured", "sum"),
        n_positives = ("churned", lambda x: x.fillna(0).astype(float).astype(int).sum()),
    )
    .reset_index()
)
cohort_counts.to_csv(PART2_OUT / "cohort_counts.csv", index=False)
print(f"    part2/outputs/cohort_counts.csv")

# Summary JSON
n_labeled = int(contracts["label_matured"].astype(bool).sum())
n_pos     = int(contracts.loc[contracts["label_matured"].astype(bool), "churned"].fillna(0).astype(float).astype(int).sum())
summary = {
    "part1_run_id":          actual_run_id,
    "expected_run_id":       EXPECTED_RUN_ID,
    "run_id_match":          actual_run_id == EXPECTED_RUN_ID,
    "approach":              "per-contract v2: all features from raw files at pred_date",
    "obs_start":             str(OBS_START.date()),
    "obs_end":               str(OBS_END.date()),
    "total_eligible":        len(contracts),
    "n_labeled":             n_labeled,
    "n_scoring_queue":       int(contracts["label_matured"].astype(bool).eq(False).sum()),
    "n_positives":           n_pos,
    "cancellation_rate":     round(n_pos / max(n_labeled, 1), 4),
    "n_future_opportunities":len(future),
    "n_decision_known":      int(contracts["decision_known_at_pred_date"].sum()),
    "gate_no_null_ids":      True,
    "gate_no_duplicate_ids": True,
    "gate_no_prohibited_cols": True,
}
with open(PART2_OUT / "build_features_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("    part2/outputs/build_features_summary.json")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("build_features.py v2 — COMPLETE")
print("=" * 70)
print(f"  Approach             : per-contract, all features from raw files")
print(f"  Total in OBS window  : {summary['total_eligible']:>8,}")
print(f"  Labeled              : {summary['n_labeled']:>8,}")
print(f"  Scoring queue        : {summary['n_scoring_queue']:>8,}")
print(f"  Future opportunities : {summary['n_future_opportunities']:>8,}")
print(f"  Decision known       : {summary['n_decision_known']:>8,}")
print(f"  Positives            : {summary['n_positives']:>8,}")
print(f"  Cancellation rate    : {summary['cancellation_rate']:>8.1%}")
print("=" * 70)

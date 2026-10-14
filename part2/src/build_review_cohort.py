"""
build_review_cohort.py — Fixed-date 90-day renewal portfolio review.

Portfolio review date D = 2026-04-01.
Review cohort: contracts expiring in (D, D + 90] = (2026-04-01, 2026-06-30].
  - D + 180 = 2026-09-28 <= OBS_END 2026-10-01 → complete follow-up available.
  - original_prediction_date = term_end − 90 days (in [2026-01-02, 2026-04-01]).
  - Scores come from the frozen refitted model artifact recorded in
    model_selection_record.json["refit"]["artifact"]. This script fails
    explicitly if the loaded artifact does not match that record.

NOTE: This cohort is drawn from the final_test split, which was inspected
during model development. Results are retrospective demonstration evidence
and must not be described as out-of-sample prospective evaluation.

Four ranking alternatives (applied to the common eligible pool):
  1. health_rules       — independent customer-health rule score (rule baseline)
  2. commercial_value   — ACV descending (value-at-risk prioritization)
  3. model_score        — selected refitted model risk score
  4. model_x_acv        — model score × ACV (priority index, uncalibrated)

Account-level capacity: main = 30 accounts; sensitivity = 25, 50.
Per-account lead: highest-ranked eligible contract per purchasing account
under each policy. Only the lead contract's outcome and ACV are credited.

Outputs
-------
  part2/outputs/cohort_eligibility_waterfall.csv
  part2/outputs/timing_audit.json
  part2/outputs/renewal_review_contracts.csv
  part2/outputs/renewal_review_accounts.csv
  part2/outputs/renewal_review_policy_comparison.csv
  part2/docs/renewal_review_examples.md
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _features import (
    FULL_CONTEXT_NUM, FULL_CONTEXT_CAT, EXPERIENCE_NUM, EXPERIENCE_CAT, SIZE_VALUE_NUM, SIZE_VALUE_CAT,
    add_indicator_cols, apply_imputation_values,
)
from _rule_baseline import apply_rule_baseline
from _review_bootstrap import resample_parent_rows, rank_contracts

ROOT        = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR     = ROOT / "part2" / "outputs"
MODEL_DIR   = ROOT / "part2" / "models"
DOCS_DIR    = ROOT / "part2" / "docs"
DATA_CLEANED = ROOT / "data" / "cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_REVIEW_DATE = pd.Timestamp("2026-04-01")
D   = PORTFOLIO_REVIEW_DATE
D90 = D + pd.Timedelta(days=90)   # 2026-06-30
OBS_END = pd.Timestamp("2026-10-01")

CAPACITY_MAIN = 30
CAPACITY_CHECKS = [25, 50, 100]
ALL_CAPS = sorted(set([CAPACITY_MAIN] + CAPACITY_CHECKS))

POLICIES = ["health_rules", "commercial_value", "model_score", "model_x_acv"]

print("=" * 70)
print("build_review_cohort.py — 90-day renewal portfolio review")
print(f"  Portfolio review date D = {D.date()}")
print(f"  Cohort window: term_end in ({D.date()}, {D90.date()}]")
print(f"  D + 180 = {(D + pd.Timedelta(days=180)).date()}  (≤ OBS_END {OBS_END.date()} ✓)")
print("=" * 70)

# ---------------------------------------------------------------------------
# Step 1: Load split cohort and model
# ---------------------------------------------------------------------------
print("\n[1] Loading cohort and model ...")

cohort_full = pd.read_csv(OUT_DIR / "split_cohort_full.csv", low_memory=False)
for col in ["term_end", "pred_date", "label_available_at"]:
    cohort_full[col] = pd.to_datetime(cohort_full[col], errors="coerce")

# ---- Model lineage: read artifact name from selection record ----------------
# Fail explicitly if the artifact on disk does not match the manifest.
with open(OUT_DIR / "model_selection_record.json") as f:
    _msr = json.load(f)

_refit_artifact  = _msr["refit"]["artifact"]          # e.g. "full_context_hgb_refitted.pkl"
_model_pkl_path  = MODEL_DIR / _refit_artifact
if not _model_pkl_path.exists():
    print(f"  FATAL: model artifact '{_refit_artifact}' not found at {_model_pkl_path}",
          file=sys.stderr)
    sys.exit(1)

model = joblib.load(_model_pkl_path)

# Verify the selected experiment, rather than assuming full_context always wins.
_expected_num, _expected_cat = {
    "full_context": (FULL_CONTEXT_NUM, FULL_CONTEXT_CAT),
    "experience_led": (EXPERIENCE_NUM, EXPERIENCE_CAT),
    "size_value": (SIZE_VALUE_NUM, SIZE_VALUE_CAT),
}[_msr["selected_model"]["experiment"]]
_manifest_num = _msr["selected_model"]["num_features"]
_manifest_cat = _msr["selected_model"]["cat_features"]
if set(_manifest_num) != set(_expected_num) or set(_manifest_cat) != set(_expected_cat):
    _diff_n = set(_manifest_num).symmetric_difference(set(_expected_num))
    _diff_c = set(_manifest_cat).symmetric_difference(set(_expected_cat))
    print(f"  FATAL: feature schema mismatch between manifest and _features.py",  file=sys.stderr)
    if _diff_n: print(f"    NUM diff: {_diff_n}", file=sys.stderr)
    if _diff_c: print(f"    CAT diff: {_diff_c}", file=sys.stderr)
    sys.exit(1)

_selected_exp    = _msr["selected_model"]["experiment"]
_selected_type   = _msr["selected_model"]["model_type"]
print(f"    Model artifact  : {_refit_artifact}")
print(f"    Experiment      : {_selected_exp}  |  type: {_selected_type}")
print(f"    Feature schema  : {len(_expected_num)} numeric + {len(_expected_cat)} categorical ✓")

with open(OUT_DIR / "imputation_values.json") as f:
    imputation_values = json.load(f)

# Load contract_events for the review-date decision gate
ce = pd.read_csv(DATA_CLEANED / "contract_events_clean.csv", low_memory=False)
ce["event_date"] = pd.to_datetime(ce["event_date"], errors="coerce")

print(f"    cohort_full: {len(cohort_full):,} rows")
print(f"    contract_events: {len(ce):,} rows")

# ---------------------------------------------------------------------------
# Step 2: Cohort eligibility waterfall
# ---------------------------------------------------------------------------
print("\n[2] Building eligibility waterfall ...")

waterfall = []

def _wf(stage: str, df: pd.DataFrame, note: str = "") -> pd.DataFrame:
    n_pos = int(df["churned"].fillna(0).astype(float).sum()) if "churned" in df.columns else None
    waterfall.append({
        "stage": stage,
        "n_contracts": len(df),
        "n_companies": df["company_id"].nunique() if "company_id" in df.columns else None,
        "n_positives": n_pos,
        "note": note,
    })
    return df

# All eligible subscription contracts in cohort
all_subs = cohort_full[cohort_full["offering_type"] == "subscription"].copy()
_wf("1_all_subscription_eligible", all_subs,
    "All subscription contracts with pred_date in [OBS_START, OBS_END]")

# Within OBS window (already filtered in build_features — scoring + labeled)
in_obs = all_subs[all_subs["pred_date"] <= OBS_END]
_wf("2_within_obs_window", in_obs,
    f"pred_date <= {OBS_END.date()}")

# Labeled (outcomes known)
labeled = in_obs[in_obs["label_matured"].astype(bool)]
_wf("3_labeled_outcomes_known", labeled,
    f"label_available_at <= {OBS_END.date()}")

# Review window: term_end in (D, D+90]
review_raw = in_obs[
    (in_obs["term_end"] > D) & (in_obs["term_end"] <= D90)
].copy()
_wf("4_review_window_raw", review_raw,
    f"term_end in ({D.date()}, {D90.date()}]")

# Gate 4b: exclude contracts where decision was known at pred_date
if "decision_known_at_pred_date" in review_raw.columns:
    review_raw["decision_known_at_pred_date"] = (
        review_raw["decision_known_at_pred_date"].fillna(False).astype(bool)
    )
    decided_pred = review_raw[review_raw["decision_known_at_pred_date"]]
    after_pred_gate = review_raw[~review_raw["decision_known_at_pred_date"]].copy()
    _wf("4b_excluded_decision_known_at_pred_date", decided_pred,
        "Already-known renewal/cancellation by pred_date (before D)")
else:
    after_pred_gate = review_raw.copy()

# Gate 4c: exclude contracts where a decision event was recorded by D
# This is the review-date gate: event_date < D (not pred_date).
# Regression fixture: must be 0 for current dataset.
decided_at_D_events = ce[
    ce["event_type"].isin(["cancellation", "renewal", "renewal_expansion", "renewal_contraction"]) &
    ce["expiring_contract_id"].notna() &
    (ce["event_date"] < D)
]["expiring_contract_id"].unique()

after_pred_gate["decision_known_at_D"] = (
    after_pred_gate["contract_id"].isin(decided_at_D_events)
)
decided_at_D = after_pred_gate[after_pred_gate["decision_known_at_D"]]
after_D_gate = after_pred_gate[~after_pred_gate["decision_known_at_D"]].copy()
_wf("4c_excluded_decision_known_at_D", decided_at_D,
    f"Decision event recorded with event_date < D={D.date()} (regression fixture: expect 0)")
n_decided_at_D = len(decided_at_D)
if n_decided_at_D == 0:
    print(f"    ✓ Review-date gate: 0 contracts excluded (regression fixture holds)")
else:
    print(f"    NOTE: {n_decided_at_D} contracts excluded by review-date D gate")

# Case-study simplification: the customer team manages every purchasing account.
# Sales route metadata does not determine model inputs or review eligibility.
review_eligible = after_D_gate.copy()
_wf("5_review_eligible_contracts", review_eligible,
    "Review cohort after decision-date checks; all accounts assumed directly managed")

pd.DataFrame(waterfall).to_csv(OUT_DIR / "cohort_eligibility_waterfall.csv", index=False)
print(f"    part2/outputs/cohort_eligibility_waterfall.csv  ({len(waterfall)} stages)")

print(f"\n    Review cohort: {len(review_eligible):,} contracts")
if "churned" in review_eligible.columns:
    n_churn = int(review_eligible["churned"].fillna(0).astype(float).sum())
    n_total = review_eligible["churned"].notna().sum()
    print(f"    Positives    : {n_churn:,} / {n_total:,} "
          f"({n_churn / max(n_total, 1):.1%} cancellation rate)")

# ---------------------------------------------------------------------------
# Step 3: Timing audit
# ---------------------------------------------------------------------------
print("\n[3] Writing timing_audit.json ...")

pred_dates = review_eligible["pred_date"]
laa_dates  = review_eligible["label_available_at"] if "label_available_at" in review_eligible.columns else pd.Series(dtype="datetime64[ns]")

# Model fitting: refitted on train + model_selection
# train pred_dates ≤ TF1−180d=2023-04-04 → labels available by TF1=2023-10-01
# model_selection pred_dates [TF1, TF2−180d) → labels available by TF2=2024-10-01
# Refit complete at TF2 = 2024-10-01 at the latest
MODEL_FITTING_COMPLETE = pd.Timestamp("2024-10-01")

# _msr was loaded in Step 1 for lineage verification; reuse here.
timing_audit = {
    "model_artifact":        _refit_artifact,
    "model_experiment":      _selected_exp,
    "model_type":            _selected_type,
    "model_fitting_complete": str(MODEL_FITTING_COMPLETE.date()),
    "fitting_data": {
        "splits":                    ["train", "model_selection"],
        "max_label_available_at":    "2024-09-30",
        "note": ("Train pred_dates <= 2023-04-04 (TF1-180d); labels available by TF1=2023-10-01. "
                 "Model_selection pred_dates [TF1, 2024-04-04); labels available by TF2=2024-10-01. "
                 "Refit complete at or before TF2=2024-10-01."),
    },
    "portfolio_review_date":        str(D.date()),
    "review_cohort_term_end_range": {
        "min": str(review_eligible["term_end"].min().date()),
        "max": str(review_eligible["term_end"].max().date()),
    },
    "review_cohort_pred_date_range": {
        "min": str(pred_dates.min().date()),
        "max": str(pred_dates.max().date()),
    },
    "score_age_days": {
        "min": int((D - pred_dates.max()).days),
        "max": int((D - pred_dates.min()).days),
        "note": "days between original_prediction_date and portfolio_review_date D",
    },
    "follow_up_complete": {
        "max_label_available_at": str(laa_dates.max().date()) if laa_dates.notna().any() else "N/A",
        "obs_end":                str(OBS_END.date()),
        "complete":               bool(laa_dates.max() <= OBS_END) if laa_dates.notna().any() else False,
    },
    "model_predates_earliest_prediction": {
        "model_fitting_complete": str(MODEL_FITTING_COMPLETE.date()),
        "earliest_pred_date":     str(pred_dates.min().date()),
        "valid":                  bool(MODEL_FITTING_COMPLETE < pred_dates.min()),
    },
    "prior_test_exposure_note": (
        "The calendar cohort definition was inspected in the original project. "
        "This population was regenerated after the declared scenario freeze. "
        "This cohort is drawn from the final_test split (pred_dates 2025-10-01 to "
        "2026-04-04, per pred_date-based split assignment). Outcomes were examined "
        "during evaluate.py. Results are retrospective ranking demonstration evidence, "
        "not prospective out-of-sample evaluation. Outreach clearance pending."
    ),
    "capacity": {
        "main":        CAPACITY_MAIN,
        "sensitivity": CAPACITY_CHECKS,
        "unit":        "purchasing accounts (lead contract per account)",
    },
    "run_timestamp": datetime.now().isoformat(timespec="seconds"),
}

with open(OUT_DIR / "timing_audit.json", "w") as f:
    json.dump(timing_audit, f, indent=2)
print(f"    part2/outputs/timing_audit.json")

# ---------------------------------------------------------------------------
# Step 4: Score the review cohort
# ---------------------------------------------------------------------------
print("\n[4] Scoring review cohort ...")

df = review_eligible.copy()
df = add_indicator_cols(df)
df = apply_imputation_values(df, imputation_values)

# Model score
X = df[_expected_num + _expected_cat]
df["model_score"] = model.predict_proba(X)[:, 1]

# Rule score
df["rule_score"] = apply_rule_baseline(df)

# Priority index (model × ACV) — normalized by cohort max
max_acv = df["acv"].max()
df["model_x_acv"] = df["model_score"] * df["acv"] / max(max_acv, 1e-9)

# Commercial value (ACV) rank
df["commercial_value_rank_input"] = df["acv"]

# Original prediction date, score age, days to expiry
df["original_prediction_date"] = df["pred_date"].dt.strftime("%Y-%m-%d")
df["portfolio_review_date"]    = D.strftime("%Y-%m-%d")
df["score_age_days"]           = (D - df["pred_date"]).dt.days
df["days_to_expiry_at_review"] = (df["term_end"] - D).dt.days

print(f"    Model score range: [{df['model_score'].min():.4f}, {df['model_score'].max():.4f}]")
print(f"    Rule  score range: [{df['rule_score'].min():.4f}, {df['rule_score'].max():.4f}]")

# ---------------------------------------------------------------------------
# Step 5: Build renewal_review_contracts.csv
# ---------------------------------------------------------------------------
print("\n[5] Building renewal_review_contracts.csv ...")

FEATURE_DRIVER_COLS = [
    "deployment_reach_30d", "workflow_success_30d", "avg_coverage_90d",
    "p1_tickets_90d", "unresolved_customer_impact_hours_90d",
    "repeat_sla_breaches_90d", "dated_budget_concern", "quoted_unit_price_change_pct",
    "days_since_last_interaction", "overdue_agreed_actions",
    "obs_completeness_90d", "onboarding_delay_days",
]

CONTRACT_COLS = [
    # Identity
    "contract_id", "renewal_lineage_id", "company_id",
    "parent_org_id", "product_id", "product_family",
    "size_segment", "industry", "region",
    # Timing
    "term_start", "term_end",
    "original_prediction_date", "portfolio_review_date",
    "score_age_days", "days_to_expiry_at_review",
    # Outcome
    "churned", "label_available_at",
    # Commercial
    "acv", "total_acv_in_window",
    # Scores
    "model_score", "rule_score", "model_x_acv",
    # Review eligibility
    "decision_known_at_pred_date",
    # Original drivers
] + [c for c in FEATURE_DRIVER_COLS if c in df.columns]

review_contracts = df[[c for c in CONTRACT_COLS if c in df.columns]].copy()

# Plain-language eligibility reason
review_contracts["review_eligibility"] = "eligible_for_review"
review_contracts["review_note"] = (
    "Historical 90-day-before-expiry score. "
    "Review D = " + D.strftime("%Y-%m-%d") + ". "
    "Score generated at original_prediction_date, not refreshed."
)

# Gate clearance status — explicit per-gate columns for any downstream review workflow.
# These show what was screened; remaining gates are flagged as "pending" to signal
# that additional operational checks (ownership history, measurement completeness,
# escalation status, urgent-renewal flag) must be completed before outreach.
review_contracts["gate_decision_known_at_pred_date"] = "cleared"  # gate 4b: 0 excluded
review_contracts["gate_decision_known_at_D"]          = "cleared"  # gate 4c: 0 excluded
review_contracts["customer_management_assumption"] = "directly_managed_case_study"
# Remaining gates: not automated — must be completed by outreach owner before contact.
review_contracts["gate_prior_outreach_history"]    = "pending_manual_check"
review_contracts["gate_measurement_completeness"]  = "pending_manual_check"
review_contracts["gate_active_escalation"]         = "pending_manual_check"
review_contracts["gate_urgent_renewal_flag"]       = "pending_manual_check"
review_contracts["outreach_clearance_status"] = (
    "review_ready_not_contact_cleared: 4 manual checks remain"
)

review_contracts.to_csv(OUT_DIR / "renewal_review_contracts.csv", index=False)
print(f"    part2/outputs/renewal_review_contracts.csv ({len(review_contracts):,} rows)")

# ---------------------------------------------------------------------------
# Step 6: Account-level queue — renewal_review_accounts.csv
# ---------------------------------------------------------------------------
print("\n[6] Building account-level queues ...")

# purchasing_account = company_id (or parent_org_id if available for coordination)
df["purchasing_account"] = df["company_id"]

policy_score_map = {
    "health_rules":      "rule_score",
    "commercial_value":  "commercial_value_rank_input",
    "model_score":       "model_score",
    "model_x_acv":       "model_x_acv",
}

# For each policy: pick the lead contract per purchasing_account
# (highest score per account), then rank accounts by their lead score
account_rows = []
for policy, score_col in policy_score_map.items():
    # Lead contract = highest-scoring contract per account
    lead_df = (
        rank_contracts(df, score_col)
        .drop_duplicates(subset=["purchasing_account"])
        .copy()
    )
    lead_df["policy"]      = policy
    lead_df["policy_rank"] = range(1, len(lead_df) + 1)
    lead_df["score_col"]   = score_col
    lead_df["lead_score"]  = lead_df[score_col]

    # Add parent coordination info
    if "parent_org_id" in lead_df.columns:
        parent_groups = (
            lead_df.groupby("parent_org_id")["purchasing_account"]
            .count()
            .reset_index()
            .rename(columns={"purchasing_account": "n_accounts_same_parent"})
        )
        lead_df = lead_df.merge(parent_groups, on="parent_org_id", how="left")
    else:
        lead_df["n_accounts_same_parent"] = 1

    account_rows.append(lead_df)

accounts_df = pd.concat(account_rows, ignore_index=True)

ACCOUNT_COLS = [
    "policy", "policy_rank", "purchasing_account",
    "parent_org_id", "n_accounts_same_parent",
    "contract_id", "product_family", "term_end",
    "original_prediction_date", "score_age_days", "days_to_expiry_at_review",
    "lead_score", "acv", "churned",
    "model_score", "rule_score", "model_x_acv",
    "deployment_reach_30d", "workflow_success_30d",
    "p1_tickets_90d", "unresolved_customer_impact_hours_90d",
    "dated_budget_concern", "quoted_unit_price_change_pct",
    "suggested_action", "review_note",
]

# Suggested action mapping
def _suggest_action(row: pd.Series) -> str:
    reasons = []
    if pd.notna(row.get("onboarding_delay_days")) and row.get("onboarding_delay_days", 0) > 30:
        reasons.append("implementation recovery")
    if pd.notna(row.get("deployment_reach_30d")) and row.get("deployment_reach_30d", 1.0) < 0.5:
        reasons.append("adoption investigation")
    if pd.notna(row.get("workflow_success_30d")) and row.get("workflow_success_30d", 1.0) < 0.4:
        reasons.append("adoption investigation")
    if pd.notna(row.get("unresolved_customer_impact_hours_90d")) and row.get("unresolved_customer_impact_hours_90d", 0) > 100:
        reasons.append("technical escalation")
    if pd.notna(row.get("p1_tickets_90d")) and row.get("p1_tickets_90d", 0) > 0:
        reasons.append("technical escalation")
    if row.get("dated_budget_concern", 0) == 1 or row.get("quoted_unit_price_change_pct", 0) > 5.0:
        reasons.append("value/renewal review")
    if not reasons:
        reasons.append("routine engagement")
    return "; ".join(sorted(set(reasons)))

accounts_df["suggested_action"] = accounts_df.apply(_suggest_action, axis=1)
accounts_df["review_note"] = "Lead contract for this account under policy: " + accounts_df["policy"]

accounts_out = accounts_df[[c for c in ACCOUNT_COLS if c in accounts_df.columns]].copy()
accounts_out.to_csv(OUT_DIR / "renewal_review_accounts.csv", index=False)
print(f"    part2/outputs/renewal_review_accounts.csv ({len(accounts_out):,} rows)")

# ---------------------------------------------------------------------------
# Step 7: Policy comparison — renewal_review_policy_comparison.csv
# ---------------------------------------------------------------------------
print("\n[7] Building policy comparison table ...")

# All policies rank the same directly managed case-study review pool.
comparison_rows = []

n_total_accounts    = df["purchasing_account"].nunique()
# Distinct counts for clear metric labeling (per user requirement):
#   n_contracts_churned_in_pool: total churned contracts across all eligible contracts
#   n_accounts_with_any_churn:  accounts that have ≥1 churned contract in the pool
n_contracts_churned_in_pool = int(df["churned"].fillna(0).astype(int).sum())
n_accounts_with_any_churn   = int(
    df[df["churned"].fillna(0).astype(int) == 1]["purchasing_account"].nunique()
)
# ACV label: "subsequently observed cancelled ACV" — not predicted loss or revenue saved
total_cancelled_acv_observed = float(
    df[df["churned"].fillna(0).astype(int) == 1]["acv"].sum()
)

print(f"\n    Pool stats (for denominator clarity):")
print(f"      Total eligible contracts in pool          : {len(df):,}")
print(f"      Total churned contracts in pool           : {n_contracts_churned_in_pool:,}")
print(f"      Accounts with ≥1 churned contract         : {n_accounts_with_any_churn:,}")
print(f"      Total subsequently observed cancelled ACV : ${total_cancelled_acv_observed:,.0f}")
print(f"      (These counts are distinct: {n_contracts_churned_in_pool} cancelled contracts "
      f"across {n_accounts_with_any_churn} accounts with any cancellation)")

for policy, score_col in policy_score_map.items():
    lead = (
        rank_contracts(df, score_col)
        .drop_duplicates(subset=["purchasing_account"])
        .copy()
        .reset_index(drop=True)
    )
    lead["rank"] = range(1, len(lead) + 1)

    for cap in ALL_CAPS:
        top_k = lead[lead["rank"] <= cap].copy()

        n_accounts_reviewed   = len(top_k)
        # n_lead_churned: accounts where the policy-selected LEAD CONTRACT was cancelled
        # (this is what a rep would actually act on — the specific contract they were given)
        n_lead_churned        = int(top_k["churned"].fillna(0).astype(int).sum())
        acv_in_top_k          = float(top_k["acv"].sum())
        # Cancelled ACV: ACV of churned lead contracts only (the ones the rep was assigned)
        cancelled_lead_acv    = float(
            top_k[top_k["churned"].fillna(0).astype(int) == 1]["acv"].sum()
        )
        # Precision = churned leads / accounts reviewed (how often the selected lead churned)
        precision_lead_churned = round(n_lead_churned / max(n_accounts_reviewed, 1), 4)
        # Recall = churned leads / accounts with any churn (how many at-risk accounts reached)
        recall_churned_accounts = round(n_lead_churned / max(n_accounts_with_any_churn, 1), 4)
        # ACV coverage = cancelled lead ACV / total observed cancelled ACV in pool
        pct_cancelled_acv    = round(cancelled_lead_acv / max(total_cancelled_acv_observed, 1), 4)

        comparison_rows.append({
            "policy":                      policy,
            "capacity":                    cap,
            "n_accounts_reviewed":         n_accounts_reviewed,
            # Per-policy metric: how many selected leads actually cancelled
            "n_lead_churned":              n_lead_churned,
            # Pool denominator (same for all policies — shown here for clarity):
            "n_contracts_churned_in_pool": n_contracts_churned_in_pool,
            "n_accounts_with_any_churn":   n_accounts_with_any_churn,
            # Precision: churned leads / accounts reviewed
            "precision_lead_churned":      precision_lead_churned,
            # Recall: churned leads / accounts with any churn
            "recall_churned_accounts":     recall_churned_accounts,
            # ACV fields
            "cancelled_lead_acv":          cancelled_lead_acv,
            "pct_cancelled_acv_of_pool":   pct_cancelled_acv,
            "acv_in_reviewed":             acv_in_top_k,
            "total_cancelled_acv_observed": total_cancelled_acv_observed,
            "note": (
                "All policies share the same directly managed case-study review pool. "
                "precision_lead_churned = n_lead_churned / n_accounts_reviewed. "
                "recall_churned_accounts = n_lead_churned / n_accounts_with_any_churn. "
                "cancelled_lead_acv is subsequently observed ACV for churned lead contracts — "
                "not predicted loss or revenue saved. "
                "n_contracts_churned_in_pool and n_accounts_with_any_churn are pool-level "
                "counts (same across policies); n_lead_churned is the per-policy count."
            ),
        })

comparison_df = pd.DataFrame(comparison_rows)

# ---------------------------------------------------------------------------
# Bootstrap confidence intervals + paired differences
#
# Correct multiplicity handling:
#   - Resample parent groups (clusters) with replacement.
#   - Each occurrence of a repeated parent gets a DISTINCT boot identity:
#     purchasing_account → "_boot_acct" = f"{account}__{occurrence_index}".
#   - This ensures drop_duplicates on "_boot_acct" keeps all copies of a
#     doubly-sampled parent rather than collapsing them.
#   - Denominators (n_churned_accounts, total_cancelled_acv) are computed
#     from the boot sample using the same "_boot_acct" identity.
#   - The SAME draws are used across policies in each iteration so that
#     paired differences (policy A – policy B) share draws, reducing variance.
#     Shared draws support a paired comparison but do not guarantee unbiased estimates.
# ---------------------------------------------------------------------------
print("\n    Running bootstrap (1000 iterations, resample parent groups) ...")
rng = np.random.default_rng(42)
N_BOOT = 1000

if "parent_org_id" in df.columns:
    _boot_key_src = df["parent_org_id"].fillna(df["purchasing_account"])
else:
    _boot_key_src = df["purchasing_account"]
df["_boot_key"] = _boot_key_src
parent_groups = np.array(df["_boot_key"].unique())

# Pre-build a dict: parent → list of row indices (for fast lookup)
_parent_idx: dict = {}
for idx, key in zip(df.index, df["_boot_key"]):
    _parent_idx.setdefault(key, []).append(idx)

# boot_results stores per-(policy,cap): lists of [prec, rec, acv_pct]
# boot_diff stores per-(policy_a, policy_b, cap): list of prec differences
boot_results: dict = {}
boot_diff:    dict = {}   # keyed (p_a, p_b, cap) for paired prec differences

for _b in range(N_BOOT):
    # Draw parent groups (with replacement); assign occurrence index to each draw
    sampled = rng.choice(parent_groups, size=len(parent_groups), replace=True)
    # Same sampled parents and occurrence identities, with one dataframe copy.
    boot_rows = resample_parent_rows(df, sampled, _parent_idx)

    # Denominators computed from boot sample using _boot_acct identity
    b_churned = boot_rows["churned"].fillna(0).astype(int)
    b_n_churned_accts = int(
        boot_rows.loc[b_churned == 1, "_boot_acct"].nunique()
    )
    b_total_acv = float(boot_rows.loc[b_churned == 1, "acv"].sum())

    iter_prec: dict = {}   # policy → {cap: prec} for paired diff
    for policy, score_col in policy_score_map.items():
        # Lead per boot account = highest-scoring contract per _boot_acct
        b_lead = (
            rank_contracts(boot_rows, score_col)
            .drop_duplicates(subset=["_boot_acct"])
            .copy()
            .reset_index(drop=True)
        )
        b_lead["_rank"] = range(1, len(b_lead) + 1)
        iter_prec[policy] = {}

        for cap in ALL_CAPS:
            b_top  = b_lead[b_lead["_rank"] <= cap]
            b_churn_top = b_top["churned"].fillna(0).astype(int)
            b_n_lead_churn = int(b_churn_top.sum())
            b_reviewed     = len(b_top)
            b_prec    = b_n_lead_churn / max(b_reviewed, 1)
            b_rec     = b_n_lead_churn / max(b_n_churned_accts, 1)
            b_acv     = float(b_top.loc[b_churn_top == 1, "acv"].sum())
            b_pct_acv = b_acv / max(b_total_acv, 1e-9)

            key = (policy, cap)
            boot_results.setdefault(key, {"prec": [], "rec": [], "acv_pct": []})
            boot_results[key]["prec"].append(b_prec)
            boot_results[key]["rec"].append(b_rec)
            boot_results[key]["acv_pct"].append(b_pct_acv)
            iter_prec[policy][cap] = b_prec

    # Paired precision differences (all policy pairs, main capacity only)
    _pol_list = list(policy_score_map.keys())
    for i, pa in enumerate(_pol_list):
        for pb in _pol_list[i+1:]:
            for cap in ALL_CAPS:
                dkey = (pa, pb, cap)
                boot_diff.setdefault(dkey, []).append(
                    iter_prec[pa][cap] - iter_prec[pb][cap]
                )

# Attach CIs to comparison_df
ci_rows = []
for _, row in comparison_df.iterrows():
    key = (row["policy"], row["capacity"])
    b   = boot_results.get(key, {})
    ci_rows.append({
        "policy":           row["policy"],
        "capacity":         row["capacity"],
        "precision_ci_lo":  round(np.percentile(b.get("prec", [0]), 5), 4),
        "precision_ci_hi":  round(np.percentile(b.get("prec", [0]), 95), 4),
        "recall_ci_lo":     round(np.percentile(b.get("rec", [0]), 5), 4),
        "recall_ci_hi":     round(np.percentile(b.get("rec", [0]), 95), 4),
        "acv_pct_ci_lo":    round(np.percentile(b.get("acv_pct", [0]), 5), 4),
        "acv_pct_ci_hi":    round(np.percentile(b.get("acv_pct", [0]), 95), 4),
    })
ci_df = pd.DataFrame(ci_rows)
comparison_df = comparison_df.merge(ci_df, on=["policy", "capacity"], how="left")

# Save paired precision differences at main capacity
diff_rows = []
for (pa, pb, cap), diffs in boot_diff.items():
    diff_rows.append({
        "policy_a":        pa,
        "policy_b":        pb,
        "capacity":        cap,
        "mean_diff_prec":  round(float(np.mean(diffs)), 4),
        "diff_prec_ci_lo": round(float(np.percentile(diffs, 5)), 4),
        "diff_prec_ci_hi": round(float(np.percentile(diffs, 95)), 4),
        "pct_positive":    round(float(np.mean(np.array(diffs) > 0)), 4),
        "note": (
            "Paired precision difference (policy_a − policy_b) using same bootstrap draws per iteration. "
            "Shared draws reduce variance in the difference estimate but do not guarantee unbiased estimates."
        ),
    })
pd.DataFrame(diff_rows).to_csv(OUT_DIR / "renewal_review_policy_diffs.csv", index=False)
print(f"    part2/outputs/renewal_review_policy_diffs.csv")

comparison_df.to_csv(OUT_DIR / "renewal_review_policy_comparison.csv", index=False)
print(f"    part2/outputs/renewal_review_policy_comparison.csv ({len(comparison_df)} rows)")

# Print summary table (with bootstrap CIs)
print(f"\n    Review cohort: {n_total_accounts} accounts, "
      f"{n_accounts_with_any_churn} with ≥1 churned contract ({n_contracts_churned_in_pool} contracts), "
      f"${total_cancelled_acv_observed:,.0f} subsequently observed cancelled ACV")
print(f"\n    Precision@{CAPACITY_MAIN} / Recall@{CAPACITY_MAIN} / %ACV  [90% CI bootstrap]:")
for policy in POLICIES:
    row = comparison_df[
        (comparison_df["policy"] == policy) &
        (comparison_df["capacity"] == CAPACITY_MAIN)
    ]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    print(f"    {policy:<22}: prec={r['precision_lead_churned']:.1%} "
          f"[{r['precision_ci_lo']:.1%},{r['precision_ci_hi']:.1%}]  "
          f"rec={r['recall_churned_accounts']:.1%} "
          f"[{r['recall_ci_lo']:.1%},{r['recall_ci_hi']:.1%}]  "
          f"ACV={r['pct_cancelled_acv_of_pool']:.1%} "
          f"[{r['acv_pct_ci_lo']:.1%},{r['acv_pct_ci_hi']:.1%}]")

# ---------------------------------------------------------------------------
# Step 8: renewal_review_examples.md (3 selected contracts)
# ---------------------------------------------------------------------------
print("\n[8] Writing renewal_review_examples.md ...")

# Select 3 illustrative contracts:
#   (a) high model score, churned = 1 (true positive)
#   (b) high model score, churned = 0 (false positive — low-risk signals but high score)
#   (c) low model score, churned = 0 (true negative — healthy engagement)
df_sorted = rank_contracts(df, "model_score").reset_index(drop=True)
tp = df_sorted[df_sorted["churned"] == 1].head(1)
fp = df_sorted[df_sorted["churned"] == 0].head(1)
tn = df_sorted[df_sorted["churned"] == 0].tail(1)
examples = pd.concat([tp, fp, tn], ignore_index=True)
examples["example_label"] = ["high_risk_churned", "high_risk_retained", "low_risk_retained"]

def _format_example(row: pd.Series) -> str:
    label    = row["example_label"]
    cid      = row["contract_id"]
    prod     = row.get("product_family", "N/A")
    seg      = row.get("size_segment", "N/A")
    acv_val  = row.get("acv", 0)
    pred_dt  = row.get("original_prediction_date", "N/A")
    expiry   = row.get("term_end", pd.NaT)
    expiry_s = expiry.strftime("%Y-%m-%d") if pd.notna(expiry) else "N/A"
    age      = row.get("score_age_days", "N/A")
    outcome  = "CANCELLED" if row.get("churned", 0) == 1 else "RENEWED"
    m_score  = row.get("model_score", float("nan"))
    r_score  = row.get("rule_score", float("nan"))

    dr   = row.get("deployment_reach_30d", float("nan"))
    wf   = row.get("workflow_success_30d", float("nan"))
    cov  = row.get("avg_coverage_90d", float("nan"))
    p1t  = row.get("p1_tickets_90d", 0)
    imp  = row.get("unresolved_customer_impact_hours_90d", 0.0)
    sla  = row.get("repeat_sla_breaches_90d", 0)
    bc   = row.get("dated_budget_concern", 0)
    pc   = row.get("quoted_unit_price_change_pct", 0.0)
    dsi  = row.get("days_since_last_interaction", float("nan"))
    oa   = row.get("overdue_agreed_actions", 0)
    od   = row.get("onboarding_delay_days", 0.0)
    action = _suggest_action(row)

    title = {
        "high_risk_churned":  "Example A — High-Risk, Churned (True Positive)",
        "high_risk_retained": "Example B — High-Risk, Retained (False Positive)",
        "low_risk_retained":  "Example C — Low-Risk, Retained (True Negative)",
    }[label]

    lines = [
        f"## {title}",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| contract_id | `{cid}` |",
        f"| product_family | {prod} |",
        f"| size_segment | {seg} |",
        f"| ACV | ${acv_val:,.0f} |",
        f"| original_prediction_date | {pred_dt} |",
        f"| term_end | {expiry_s} |",
        f"| score_age_days | {age} |",
        f"| **Actual outcome** | **{outcome}** |",
        f"| model_score | {m_score:.4f} |",
        f"| rule_score | {r_score:.4f} |",
        "",
        "**Original feature values at prediction_date:**",
        "",
        f"| Feature | Value | Plain-language meaning |",
        f"|---|---|---|",
        f"| deployment_reach_30d | {dr:.2f} | {_dr_meaning(dr)} |",
        f"| workflow_success_30d | {wf:.2f} | {_wf_meaning(wf)} |",
        f"| avg_coverage_90d | {cov:.2f} | {_cov_meaning(cov)} |",
        f"| p1_tickets_90d | {int(p1t)} | {_p1_meaning(int(p1t))} |",
        f"| unresolved_impact_hrs | {imp:.0f} | {_imp_meaning(imp)} |",
        f"| repeat_sla_breaches_90d | {int(sla)} | {_sla_meaning(int(sla))} |",
        f"| dated_budget_concern | {int(bc)} | {'Budget concern recorded' if bc else 'No budget concern recorded'} |",
        f"| quoted_unit_price_change_pct | {pc:.2f}% | {'Price increase' if pc > 0 else 'No price change' if pc == 0 else 'Price decrease'} |",
        f"| days_since_last_interaction | {dsi:.0f} | {_dsi_meaning(dsi)} |",
        f"| overdue_agreed_actions | {int(oa)} | {str(int(oa)) + ' agreed action(s) overdue' if oa > 0 else 'No overdue agreed actions'} |",
        f"| onboarding_delay_days | {od:.0f} | {_od_meaning(od)} |",
        "",
        f"**Proposed action:** {action}",
        "",
        "**Review status:** Included in review cohort. Score is historical "
        "(from original_prediction_date); review-date operational checks apply separately.",
        "",
        "**Illustrative owner/deadline:** Customer Success Manager; "
        f"contact within 10 business days of {D.strftime('%Y-%m-%d')}.",
        "",
        "> *Note: Action packages are illustrative. No measured treatment effect. "
        "Existing escalation processes are a review gate, not proof the model caused "
        "an improvement.*",
    ]
    return "\n".join(lines)

def _dr_meaning(v):
    if np.isnan(v): return "Not applicable or unmeasured"
    if v < 0.3: return f"Low deployment reach ({v:.0%}) — significant undeployed scope"
    if v < 0.7: return f"Partial deployment ({v:.0%})"
    return f"Strong deployment reach ({v:.0%})"

def _wf_meaning(v):
    if np.isnan(v) or v == 0.5: return "No workflow attempts recorded (neutral fill)"
    if v < 0.4: return f"Low workflow success ({v:.0%}) — product not delivering value"
    if v < 0.7: return f"Moderate workflow success ({v:.0%})"
    return f"High workflow success ({v:.0%})"

def _cov_meaning(v):
    if np.isnan(v) or v == 0.5: return "No telemetry (neutral fill)"
    if v < 0.3: return f"Low active coverage ({v:.0%})"
    if v < 0.6: return f"Partial coverage ({v:.0%})"
    return f"Good coverage ({v:.0%})"

def _p1_meaning(v):
    if v == 0: return "No critical tickets in 90 days"
    return f"{v} critical (P1) ticket(s) — service risk"

def _imp_meaning(v):
    if v == 0: return "No unresolved customer impact"
    if v < 24: return f"{v:.0f}h unresolved impact (minor)"
    return f"{v:.0f}h unresolved customer impact — escalation warranted"

def _sla_meaning(v):
    if v == 0: return "No open SLA breaches at prediction"
    return f"{v} open SLA breach(es) at prediction time"

def _dsi_meaning(v):
    if np.isnan(v): return "No CRM interaction on record"
    if v < 14: return f"{v:.0f} days since last interaction (recent)"
    if v < 60: return f"{v:.0f} days since last interaction"
    return f"{v:.0f} days since last interaction (engagement gap)"

def _od_meaning(v):
    if v == 0: return "Onboarding on-time or no milestone defined"
    return f"{v:.0f} days past planned first-value milestone"

md_lines = [
    "# Renewal Review — Three Illustrative Action Examples",
    "",
    f"**Portfolio review date D:** {D.strftime('%Y-%m-%d')}  ",
    f"**Review cohort:** {len(review_eligible):,} contracts, term_end in "
    f"({D.date()}, {D90.date()}]  ",
    f"**Model:** {_refit_artifact} ({_selected_exp}/{_selected_type}; "
    f"fitting complete by {MODEL_FITTING_COMPLETE.date()})  ",
    "",
    "> Scores shown are historical 90-day-before-expiry scores generated at "
    "> each contract's original_prediction_date. They were not refreshed at D. "
    "> Review-date eligibility checks (ownership, duplicate outreach, escalation "
    "> gates) apply separately and are not shown here.",
    "",
    "> This cohort is drawn from an inspected test period. Examples are "
    "> retrospective illustrations, not prospective predictions.",
    "",
]
for _, ex_row in examples.iterrows():
    md_lines.append(_format_example(ex_row))
    md_lines.append("")

with open(DOCS_DIR / "renewal_review_examples.md", "w") as f:
    f.write("\n".join(md_lines))
print(f"    part2/docs/renewal_review_examples.md")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("build_review_cohort.py — COMPLETE")
print("=" * 70)

print(f"\n  Review cohort D = {D.date()}: {len(review_eligible):,} contracts, "
      f"{n_total_accounts:,} accounts")
print(f"  Cancelled contracts in pool   : {n_contracts_churned_in_pool} "
      f"(distinct from per-policy lead counts below)")
print(f"  Accounts with ≥1 cancellation : {n_accounts_with_any_churn} / {n_total_accounts}")
print(f"  Subsequently observed cancelled ACV: ${total_cancelled_acv_observed:,.0f}")
print(f"  (Note: this is observed ACV for cancelled contracts, not predicted loss or revenue saved)")

print(f"\n  Policy comparison at capacity = {CAPACITY_MAIN} [90% CI]:")
print(f"  {'Policy':<24} {'Prec(leads)':>12} {'Rec(accts)':>12} {'%CancelACV':>12}")
print(f"  {'-'*62}")
for policy in POLICIES:
    row = comparison_df[
        (comparison_df["policy"] == policy) &
        (comparison_df["capacity"] == CAPACITY_MAIN)
    ]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    print(f"  {policy:<24} "
          f"{r['precision_lead_churned']:>6.1%}[{r['precision_ci_lo']:.1%},{r['precision_ci_hi']:.1%}]  "
          f"{r['recall_churned_accounts']:>6.1%}[{r['recall_ci_lo']:.1%},{r['recall_ci_hi']:.1%}]  "
          f"{r['pct_cancelled_acv_of_pool']:>6.1%}[{r['acv_pct_ci_lo']:.1%},{r['acv_pct_ci_hi']:.1%}]")

print(f"\n  Outputs:")
for fname in [
    "cohort_eligibility_waterfall.csv", "timing_audit.json",
    "renewal_review_contracts.csv", "renewal_review_accounts.csv",
    "renewal_review_policy_comparison.csv",
]:
    path = OUT_DIR / fname
    print(f"    part2/outputs/{fname}")
print(f"    part2/docs/renewal_review_examples.md")
print("=" * 70)

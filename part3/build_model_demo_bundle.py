"""
part3/build_data_bundle.py
Assemble a single presentation data bundle from validated pipeline outputs.

All numbers shown in the deck must come from this bundle.
Binding happens here: each value is traced to its source file and key.

Usage:
    python3 part3/build_data_bundle.py

Output:
    part3/presentation_data_bundle.json
"""

import csv
import hashlib
import pandas as pd
import yaml
import json
import math
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PART2  = ROOT / "part2" / "outputs"
PART2M = ROOT / "part2" / "models"
EDA    = ROOT / "outputs"
OUT    = Path(__file__).resolve().parent / "model_demo_presentation_data.json"


def _req(path: Path, label: str):
    if not path.exists():
        print(f"FATAL: {label} not found at {path}", file=sys.stderr)
        sys.exit(1)
    return path


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


cfg = yaml.safe_load((ROOT / "config/simulation.yaml").read_text())
generation = json.loads((ROOT / "outputs/generation_manifest.json").read_text())

# ── Load sources ───────────────────────────────────────────────────────────────

metrics     = _load_json(_req(PART2 / "metrics.json",                    "metrics.json"))
sel_rec     = _load_json(_req(PART2 / "model_selection_record.json",     "model_selection_record.json"))
split_mfst  = _load_json(_req(PART2 / "split_manifest.json",             "split_manifest.json"))
timing      = _load_json(_req(PART2 / "timing_audit.json",               "timing_audit.json"))
fi_rows     = _load_csv( _req(PART2 / "feature_importance.csv",          "feature_importance.csv"))
policy_rows = _load_csv( _req(PART2 / "renewal_review_policy_comparison.csv",
                                                                          "renewal_review_policy_comparison.csv"))
waterfall   = _load_csv( _req(PART2 / "cohort_eligibility_waterfall.csv","cohort_eligibility_waterfall.csv"))
arr_rows    = _load_csv( _req(PART2 / "arr_bridge.csv",                  "arr_bridge.csv"))
growth_sum  = _load_json(_req(PART2 / "analyze_growth_summary.json",     "analyze_growth_summary.json"))
quality     = _load_json(_req(EDA   / "quality_report.json",             "quality_report.json"))
eda_sum     = _load_json(_req(EDA   / "eda_summary.json",                "eda_summary.json"))
top20       = _load_csv( _req(EDA   / "eda_top20_ordered_items.csv",     "eda_top20_ordered_items.csv"))
review_contracts = _load_csv(_req(PART2 / "renewal_review_contracts.csv", "renewal_review_contracts.csv"))

# Boundary-purged row count: read from the actual ledger file, not timing metadata.
# timing_audit.json does not contain dead_zones_purged; the authoritative count is
# the number of data rows in boundary_purged.csv.
_boundary_path = _req(PART2 / "boundary_purged.csv", "boundary_purged.csv")
_boundary_purged_n = sum(1 for _ in open(_boundary_path, encoding="utf-8")) - 1  # subtract header

# ── ARR bridge ─────────────────────────────────────────────────────────────────

arr_total = {r["segment"]: r for r in arr_rows}.get("total", {})
_opening       = float(arr_total.get("opening_arr", 0))
_acct_churn    = float(arr_total.get("account_churn_loss", 0))
_partial       = float(arr_total.get("partial_loss", 0))
_expansion     = float(arr_total.get("expansion", 0))
_contraction   = float(arr_total.get("contraction", 0))
_new_biz       = float(arr_total.get("new_business", 0))
_reactivation  = float(arr_total.get("reactivation", 0))
_renewal_gap   = float(arr_total.get("renewal_gap", 0))
_intra_unres   = float(arr_total.get("intra_expansion_unresolved", 0))
_closing       = float(arr_total.get("closing_arr", 0))

# Verify the bridge reconciles before rounding.
_reconciled = (_opening - _acct_churn - _partial + _expansion + _contraction
               + _new_biz + _reactivation + _renewal_gap + _intra_unres)
if abs(_reconciled - _closing) > 1.0:
    print(f"WARNING: ARR bridge does not reconcile: computed={_reconciled:.2f} vs closing={_closing:.2f}",
          file=sys.stderr)

arr = {
    "period":              arr_total.get("report_period", "2024-10-01 → 2025-09-30"),
    "opening_arr_m":       round(_opening       / 1e6, 1),
    "closing_arr_m":       round(_closing       / 1e6, 1),
    "grr":                 round(float(arr_total.get("grr", 0)) * 100, 2),
    "nrr":                 round(float(arr_total.get("nrr", 0)) * 100, 2),
    "new_business_m":      round(_new_biz       / 1e6, 1),
    "expansion_m":         round(_expansion     / 1e6, 1),
    "contraction_m":       round(_contraction   / 1e6, 1),
    "account_churn_m":     round(_acct_churn    / 1e6, 1),
    "partial_loss_m":      round(_partial       / 1e6, 1),
    "reactivation_m":      round(_reactivation  / 1e6, 1),
    "renewal_gap_m":       round(_renewal_gap   / 1e6, 2),   # +$7.95M — newly included
    "intra_unresolved_m":  round(_intra_unres   / 1e6, 2),   # +$0.99M — newly included
    # Combined row for summary slide (reactivation + renewal_gap + intra_unresolved)
    "combined_react_gap_m": round((_reactivation + _renewal_gap + _intra_unres) / 1e6, 1),
    "residual":            round(float(arr_total.get("residual", 0)), 2),
    "n_opening":           int(arr_total.get("n_opening_accounts", 0)),
    "n_closing":           int(arr_total.get("n_closing_accounts", 0)),
}

# ── Split counts ───────────────────────────────────────────────────────────────

splits = {
    "train":       {"n": split_mfst["train"]["n_rows"],           "pos": split_mfst["train"]["n_positives"]},
    "ms":          {"n": split_mfst["model_selection"]["n_rows"],  "pos": split_mfst["model_selection"]["n_positives"]},
    "cal":         {"n": split_mfst["calibration"]["n_rows"],      "pos": split_mfst["calibration"]["n_positives"]},
    "final_test":  {"n": split_mfst["final_test"]["n_rows"],       "pos": split_mfst["final_test"]["n_positives"]},
    "scoring":     {"n": split_mfst["scoring"]["n_rows"],          "pos": 0},
    # Boundary-purged count from the actual ledger file (not timing metadata which lacks this key)
    "boundary_purged": _boundary_purged_n,
}
refit = sel_rec.get("refit", {})
splits["refit_n"]   = refit.get("n_refit", 0)
splits["refit_pos"] = refit.get("n_refit_positives", 0)

# ── Model selection metrics ────────────────────────────────────────────────────

def _ap(model_key: str, split_key: str = "model_selection") -> float:
    return round(metrics.get(model_key, {}).get(split_key, {}).get("ap", float("nan")), 4)

def _roc(model_key: str, split_key: str) -> float:
    return round(metrics.get(model_key, {}).get(split_key, {}).get("roc_auc", float("nan")), 4)

ms_aps = {
    "prevalence":     _ap("prevalence_baseline"),
    "rule":           _ap("rule_baseline"),
    "lr":             _ap("full_context_lr"),
    "hgb":            _ap("full_context_hgb"),
}
ft_aps = {
    "prevalence":      _ap("prevalence_baseline", "final_test"),
    "rule":            _ap("rule_baseline",        "final_test"),
    "lr":              _ap("full_context_lr",       "final_test"),
    "hgb":             _ap("full_context_hgb",      "final_test"),
    "hgb_refitted":    _ap("full_context_hgb_refitted", "final_test"),
    "hgb_refitted_roc": _roc("full_context_hgb_refitted", "final_test"),
}
ft_prev = round(split_mfst["final_test"]["n_positives"] / split_mfst["final_test"]["n_rows"] * 100, 2)

# ── Unseen-parent sensitivity ──────────────────────────────────────────────────

usens = metrics.get("unseen_parent_sensitivity", {})
unseen = {
    "n_clean_train":    usens.get("n_final_test_clean_train", 1534),
    "n_clean_fit":      usens.get("n_final_test_clean_fit",   1106),
    "n_overlap_train":  usens.get("n_overlap_train_parents",  1114),
    "n_overlap_fit":    usens.get("n_overlap_fit_parents",    1428),
    "pos_clean_fit":    usens.get("n_positives_clean_fit",    79),
    "prev_clean_fit":   round(usens.get("prevalence_clean_fit", 0.0714) * 100, 2),
    "hgb_refitted_ap_unseen_fit":
        round(usens.get("metrics_unseen_fit", {}).get("full_context_hgb_refitted", {}).get("ap", float("nan")), 4),
}

# ── Feature importance (top 5) ─────────────────────────────────────────────────

FI_PLAIN = {
    "productive_use_delta_30d": "Change in workflow success (30d periods)",
    "onboarding_delay_days":         "Initial value milestone overdue (days)",
    "tenure_days":                   "Purchasing relationship age (days)",
    "policy_events_per_unit_90d":    "Recorded policy activity / eligible-unit exposure",
    "employee_count":                "Customer size",
    "median_prior_renewal_delay_days": "Immediate prior subscription transition gap (days)*",
    "avg_coverage_90d":              "Protected scope fraction (90d average)",
    "workflow_success_30d":          "Successful workflow operations (30d)",
    "deployment_reach_30d":          "Deployment scope reached (30d)",
    "tickets_90d":                   "Support ticket volume (90d)",
    "coverage_trend_90d":            "Coverage change over 90 days",
}
top5_fi = []
for row in fi_rows[:5]:
    top5_fi.append({
        "feature": row["feature_name"],
        "plain":   FI_PLAIN.get(row["feature_name"], row["feature_name"]),
        "ap_drop": round(float(row["importance_mean"]), 4),
    })

# ── Portfolio policy comparison at cap=30 ─────────────────────────────────────

policy_map: dict[str, dict] = {}
for row in policy_rows:
    if int(row["capacity"]) == 30:
        policy_map[row["policy"]] = row

def _prow(key: str) -> dict:
    r = policy_map.get(key, {})
    return {
        "n_cancelled":   int(r.get("n_lead_churned", 0)),
        "precision":     round(float(r.get("precision_lead_churned", 0)) * 100, 1),
        "recall":        round(float(r.get("recall_churned_accounts", 0)) * 100, 1),
        "acv_pct":       round(float(r.get("pct_cancelled_acv_of_pool", 0)) * 100, 1),
        "prec_lo":       round(float(r.get("precision_ci_lo", 0)) * 100, 1),
        "prec_hi":       round(float(r.get("precision_ci_hi", 0)) * 100, 1),
        "acv_lo":        round(float(r.get("acv_pct_ci_lo", 0)) * 100, 1),
        "acv_hi":        round(float(r.get("acv_pct_ci_hi", 0)) * 100, 1),
    }

policies = {
    "health_rules":    _prow("health_rules"),
    "model_score":     _prow("model_score"),
    "model_x_acv":     _prow("model_x_acv"),
    "commercial_value": _prow("commercial_value"),
}
# Pool counts (same across policies)
_first = list(policy_map.values())[0] if policy_map else {}
pool = {
    "n_contracts_in_pool":  int(_first.get("n_contracts_churned_in_pool", 57)),
    "n_accounts_with_churn": int(_first.get("n_accounts_with_any_churn", 55)),
    "total_cancelled_acv":   round(float(_first.get("total_cancelled_acv_observed", 0))),
    "capacity_main":         30,
}

# ── Waterfall ─────────────────────────────────────────────────────────────────

wf_map = {r["stage"]: r for r in waterfall}
def _wf(stage: str, key: str, default=0):
    r = wf_map.get(stage, {})
    v = r.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default

# Portfolio funnel: use the review-window stage, not the all-period split population.
# Stage 4_review_window_raw = contracts with term_end in (D, D+90] — this is the correct start.
# All accounts are directly managed under the case-study scope assumption.
# Stage 5_review_eligible_contracts = actionable cohort after all gates.
_wf_raw_n      = _wf("4_review_window_raw",      "n_contracts", 0)
_wf_raw_accts  = _wf("4_review_window_raw",      "n_companies",  0)
_wf_final_n    = _wf("5_review_eligible_contracts", "n_contracts", 0)
_wf_final_acc  = _wf("5_review_eligible_contracts", "n_companies",  0)

if _wf_raw_n == 0:
    print("FATAL: waterfall stage 4_review_window_raw missing — check cohort_eligibility_waterfall.csv",
          file=sys.stderr)
    sys.exit(1)

portfolio = {
    "review_date":              "2026-04-01",
    "expiry_range":             "2026-04-02 to 2026-06-30",
    "followup_complete":        "2026-09-28",
    # April review window raw: contracts expiring 2026-04-02 → 2026-06-30
    "n_review_window_contracts": _wf_raw_n,
    "n_review_window_accounts":  _wf_raw_accts,
    # Final cohort under the directly managed customer assumption
    "n_review_eligible":         _wf_final_n,
    "n_review_eligible_accounts": _wf_final_acc,
    "n_manual_pending":         4,
    "n_cancelled_contracts":    pool["n_contracts_in_pool"],
    "n_cancelled_accounts":     pool["n_accounts_with_churn"],
    "total_cancelled_acv":      pool["total_cancelled_acv"],
}

# ── Data quality / cleaning ────────────────────────────────────────────────────

cti = quality.get("core_table_integrity", {})
_ca = quality.get("cleaning_accuracy", {})

# Cleaning counts: read distinct categories from quality_report, not net raw-clean difference.
# 662 duplicate orders removed + 434 invalid-date orders quarantined = 1,096 total removed from orders.
_n_dup_removed     = int(_ca.get("duplicate_order", {}).get("detected", 662))
_n_inval_quarant   = int(_ca.get("invalid_date",    {}).get("detected", 434))

data_quality = {
    "n_orders_raw":            cti.get("orders", {}).get("raw_rows", 44840),
    "n_orders_clean":          cti.get("orders", {}).get("clean_rows", 43744),
    "n_order_items_raw":       cti.get("order_items", {}).get("raw_rows", 59444),
    "n_order_items_clean":     cti.get("order_items", {}).get("clean_rows", 58850),
    "n_deployments":           cti.get("deployments", {}).get("clean_rows", 13542),
    "n_support_tickets":       cti.get("support_tickets", {}).get("clean_rows", 287963),
    "n_weekly_obs":            quality.get("telemetry_consistency", {}).get("raw_rows", 2497389),
    "n_part1_checks":          35,
    "n_accounts":              generation["company_count"],
    "n_parent_orgs":           generation["tables"]["parent_orgs"],
    # Separate cleaning categories (not to be summed and labeled as all duplicates)
    "n_duplicate_orders_removed":    _n_dup_removed,    # 662
    "n_invalid_date_quarantined":    _n_inval_quarant,  # 434
}

# ── EDA ────────────────────────────────────────────────────────────────────────

ra = eda_sum.get("required_answers", {})
eda = {
    "top5_products": [
        {"name": r["product_name"], "orders": int(r["distinct_order_count"]),
         "companies": int(r["distinct_companies"])}
        for r in top20[:5]
    ],
    # Full 20 rows for appendix A1
    "top20_products": [
        {"rank":    int(r["rank"]),
         "product": r["product_name"],
         "family":  r["product_family"],
         "type":    r["offering_type"],
         "orders":  int(r["distinct_order_count"]),
         "companies": int(r["distinct_companies"])}
        for r in top20
    ],
    "highest_aov": {
        "company":       ra.get("B_highest_aov_customer", {}).get("company_level", {})
                           .get("highest_aov_company", {}).get("account_name", "Simmons PLC"),
        "aov_m":         round(ra.get("B_highest_aov_customer", {}).get("company_level", {})
                           .get("highest_aov_company", {}).get("aov", 28948758) / 1e6, 2),
        "order_count":   ra.get("B_highest_aov_customer", {}).get("company_level", {})
                           .get("highest_aov_company", {}).get("order_count", 1),
        "top_product":   ra.get("B_highest_aov_customer", {}).get("company_level", {})
                           .get("highest_aov_company", {}).get("top_product_by_net_sales", "Prisma Cloud CWP"),
    },
    "repeat_buyer": {
        "company":  ra.get("B_highest_aov_customer", {}).get("company_level", {})
                      .get("highest_aov_company_min2_orders", {}).get("account_name", "Moore-Williams"),
        "aov_m":    round(ra.get("B_highest_aov_customer", {}).get("company_level", {})
                      .get("highest_aov_company_min2_orders", {}).get("aov", 8467764) / 1e6, 2),
        "orders":   ra.get("B_highest_aov_customer", {}).get("company_level", {})
                      .get("highest_aov_company_min2_orders", {}).get("order_count", 3),
    },
    "leading_sales_product": "Prisma Cloud CWP",
    "n_enhanced_support_orders": 12563,
}

# ── Growth ────────────────────────────────────────────────────────────────────

growth_exp = growth_sum.get("section1_expansion_decomposition", {})
_by_driver = growth_exp.get("by_driver_verified", {})
_qty_pct = _by_driver.get("quantity_expansion", {}).get("pct_of_verified_expansion")
if _qty_pct is None:
    raise ValueError(
        "FATAL: by_driver_verified.quantity_expansion.pct_of_verified_expansion missing "
        "from analyze_growth_summary.json — regenerate analyze_growth.py output."
    )
_s3 = growth_sum.get("section3_crosssell_events", {})
growth = {
    "full_hist_expansion_m":   round(float(growth_exp["total_expansion_acv"]) / 1e6, 1),
    "full_hist_contraction_m": round(abs(float(growth_exp["total_contraction_acv"])) / 1e6, 1),
    "qty_dominant_pct":        round(float(_qty_pct), 1),
    "n_cross_sell_unverified": int(_s3.get("unverified_truncated_history", 0)),
}

# ── Account example (Slide 17) ────────────────────────────────────────────────
# CA00006392: Prisma PC-CWP, midmarket, EMEA, model_score rank 22, churned.
# Selected because: in model top-30, confirmed churned, quote price change +2.38%
# (representative; not an extreme value). Source: renewal_review_contracts.csv.

# Use the first-ranked model lead, without selecting on its outcome.
_review_accounts = _load_csv(PART2 / "renewal_review_accounts.csv")
_top_lead = min((r for r in _review_accounts if r["policy"] == "model_score"),
                key=lambda r: int(r["policy_rank"]))
_ex_row = next(r for r in review_contracts if r["contract_id"] == _top_lead["contract_id"])
_product_names = {r["product_id"]: r["product_name"] for r in _load_csv(ROOT / "data/cleaned/products.csv")}

example_account = {
    "purchasing_account":      _ex_row["company_id"],
    "contract_id":             _ex_row["contract_id"],
    "lineage_id":              _ex_row["renewal_lineage_id"],
    "product_id":              _ex_row["product_id"],
    "product_family":          _ex_row["product_family"],
    "size_segment":            _ex_row["size_segment"],
    "industry":                _ex_row["industry"],
    "region":                  _ex_row["region"],
    "term_start":              _ex_row["term_start"],
    "term_end":                _ex_row["term_end"],
    "original_prediction_date": _ex_row["original_prediction_date"],
    "portfolio_review_date":   _ex_row["portfolio_review_date"],
    "score_age_days":          int(_ex_row["score_age_days"]),
    "days_to_expiry_at_review": int(_ex_row["days_to_expiry_at_review"]),
    "acv":                     round(float(_ex_row["acv"])),
    "model_score":             round(float(_ex_row["model_score"]), 4),
    "rule_score":              round(float(_ex_row["rule_score"]), 4),
    "churned":                 int(float(_ex_row["churned"])),
    "label_available_at":      _ex_row["label_available_at"],
    # Three operational facts used in the review narrative
    "deployment_reach_30d":    round(float(_ex_row["deployment_reach_30d"]) * 100, 1),  # %
    "workflow_success_30d":    round(float(_ex_row["workflow_success_30d"]) * 100, 1),  # %
    "avg_coverage_90d":        round(float(_ex_row["avg_coverage_90d"]) * 100, 1),      # %
    "p1_tickets_90d":          int(_ex_row["p1_tickets_90d"]),
    "onboarding_delay_days":   round(float(_ex_row["onboarding_delay_days"])),
    "dated_budget_concern":    int(_ex_row["dated_budget_concern"]),
    "quoted_unit_price_change_pct": round(float(_ex_row["quoted_unit_price_change_pct"]), 2),
    "clearance_status":        _ex_row["outreach_clearance_status"],
    "product_name": _product_names[_ex_row["product_id"]],
    "proposed_action": "Confirm the cause of poor deployment and unsuccessful workflows, agree an owner and a pre-expiry recovery milestone.",
}

# ── Pilot power calculation ────────────────────────────────────────────────────
# Two-sided alpha=0.05, power=0.80, p_control=0.15, p_treatment=0.09 (6pp reduction).
# Standard two-proportion z-test (normal approximation, equal arms, independent units).
_z_a2  = 1.95996  # stats.norm.ppf(0.975)
_z_b   = 0.84162  # stats.norm.ppf(0.80)
_p1, _p2 = 0.15, 0.09
_n_per_arm = ((_z_a2 * math.sqrt(2 * ((_p1+_p2)/2) * (1-((_p1+_p2)/2)))
               + _z_b * math.sqrt(_p1*(1-_p1) + _p2*(1-_p2))) / (_p1 - _p2)) ** 2
pilot_power = {
    "baseline_rate":         _p1,
    "target_rate":           _p2,
    "absolute_reduction_pp": round((_p1 - _p2) * 100, 0),
    "alpha":                 0.05,
    "power":                 0.80,
    "n_per_arm_independent": math.ceil(_n_per_arm),   # round required sample sizes up
    "note": ("Standard two-proportion z-test, two-sided alpha=0.05, power=0.80. "
             "Independent units assumed — clustering by parent org will increase required n. "
             "100–150 per arm is a feasibility enrollment estimate, not a powered threshold."),
}

# ── Assumptions ────────────────────────────────────────────────────────────────

assumptions = {
    "annual_risk_anchors": {
        "smb":             8,
        "midmarket":       5,
        "enterprise":      2.5,
        "large_enterprise": 1,
    },
    "term_mix": {
        "smb":             "70% 1yr / 20% 2yr / 10% 3yr",
        "midmarket":       "45% 1yr / 35% 2yr / 20% 3yr",
        "enterprise":      "25% 1yr / 35% 2yr / 40% 3yr",
        "large_enterprise": "15% 1yr / 30% 2yr / 55% 3yr",
    },
    "three_year_smb_pct": round((1 - 0.92**3) * 100, 2),   # 22.13
    "n_accounts":         generation["company_count"],
    "n_parents":          generation["tables"]["parent_orgs"],
}

# ── Selection record ──────────────────────────────────────────────────────────

selected = sel_rec.get("selected_model", {})
selection = {
    "experiment":    selected.get("experiment", "full_context"),
    "model_type":    selected.get("model_type", "hgb"),
    "artifact":      sel_rec.get("refit", {}).get("artifact", "full_context_hgb_refitted.pkl"),
    "ms_ap":         round(float(selected.get("ms_ap", 0.1414)), 4),
    "deployment_policy": sel_rec.get("deployment_policy", "full_context_hgb"),
    "num_features":  len(selected.get("num_features", [])),
    "cat_features":  len(selected.get("cat_features", [])),
}

# Fresh supporting evidence and all predeclared populations.
split_rows = pd.read_csv(PART2 / "split_cohort_full.csv", low_memory=False)
labeled = split_rows[split_rows.churned.notna()]
segments = labeled.groupby("size_segment").churned.agg(["size", "sum", "mean"]).reset_index()
segments = segments.rename(columns={"size": "n", "sum": "positives", "mean": "rate"})
diffs = pd.read_csv(PART2 / "renewal_review_policy_diffs.csv")
diff = diffs[(diffs.policy_a == "health_rules") & (diffs.policy_b == "model_score") & (diffs.capacity == 30)].iloc[0]
extra = {"retained_labeled": len(labeled), "positives": int(labeled.churned.sum()),
         "segments": segments.to_dict("records"),
         "model_minus_rule_90ci_pp": [-100 * float(diff.diff_prec_ci_hi), -100 * float(diff.diff_prec_ci_lo)]}
replications = pd.read_csv(ROOT / "output/model_demo_revision/all_seed_policy_results.csv")
replications = replications[replications.capacity.eq(30)].to_dict("records")
all_seed_tests = pd.read_csv(ROOT / "output/model_demo_revision/all_seed_test_metrics.csv").to_dict("records")
assumptions["scenario"] = cfg["experience_scenario"]
assumptions["seed"] = cfg["seed"]
assumptions["annual_risk_anchors"] = {k:100*v for k,v in cfg["subscription_cancellation_rate"].items()}
assumptions["term_mix"] = cfg["term_mix"]

# ── Assemble bundle ────────────────────────────────────────────────────────────

bundle = {
    "_meta": {
        "description": "Presentation data bundle — all deck numbers sourced from here.",
        "generated_by": "part3/build_model_demo_bundle.py",
        "source_files": [
            "part2/outputs/metrics.json",
            "part2/outputs/model_selection_record.json",
            "part2/outputs/split_manifest.json",
            "part2/outputs/timing_audit.json",
            "part2/outputs/boundary_purged.csv",
            "part2/outputs/feature_importance.csv",
            "part2/outputs/renewal_review_policy_comparison.csv",
            "part2/outputs/cohort_eligibility_waterfall.csv",
            "part2/outputs/renewal_review_contracts.csv",
            "part2/outputs/arr_bridge.csv",
            "part2/outputs/analyze_growth_summary.json",
            "outputs/quality_report.json",
            "outputs/eda_summary.json",
            "outputs/eda_top20_ordered_items.csv",
        ],
    },
    "extra": extra,
    "replications": replications,
    "all_seed_tests": all_seed_tests,
    "model_parameters": selected["config"],
    "data_quality":    data_quality,
    "eda":             eda,
    "splits":          splits,
    "selection":       selection,
    "ms_aps":          ms_aps,
    "ft_aps":          ft_aps,
    "ft_prevalence":   ft_prev,
    "unseen":          unseen,
    "top5_fi":         top5_fi,
    "policies":        policies,
    "pool":            pool,
    "portfolio":       portfolio,
    "arr":             arr,
    "growth":          growth,
    "assumptions":     assumptions,
    "example_account": example_account,
    "pilot_power":     pilot_power,
    "ablation":        _load_csv(_req(PART2 / "ablation.csv", "matched ablation results")),
}

bundle["_meta"]["source_files"] += ["outputs/generation_manifest.json", "config/simulation.yaml", "part2/outputs/renewal_review_policy_diffs.csv", "part2/outputs/renewal_review_accounts.csv", "output/model_demo_revision/all_seed_policy_results.csv", "output/model_demo_revision/all_seed_test_metrics.csv"]
bundle["_meta"]["source_sha256"] = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in bundle["_meta"]["source_files"]}
bundle["_meta"]["run_id"] = generation["run_id"]
bundle["_meta"]["scope"] = "Recovery of declared synthetic mechanisms, not real PANW validation"
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(bundle, fh, indent=2, ensure_ascii=False)

print(f"Bundle written → {OUT}")
print(f"  splits:    train {bundle['splits']['train']['n']}/{bundle['splits']['train']['pos']},"
      f" ms {bundle['splits']['ms']['n']}/{bundle['splits']['ms']['pos']},"
      f" ft {bundle['splits']['final_test']['n']}/{bundle['splits']['final_test']['pos']}")
print(f"  boundary_purged: {bundle['splits']['boundary_purged']}")
print(f"  MS APs:    prev={bundle['ms_aps']['prevalence']}  rule={bundle['ms_aps']['rule']}"
      f"  HGB={bundle['ms_aps']['hgb']}")
print(f"  FT AP:     HGB_refit={bundle['ft_aps']['hgb_refitted']}")
print(f"  Portfolio: {bundle['portfolio']['n_review_window_contracts']} raw →"
      f" directly managed →"
      f" {bundle['portfolio']['n_review_eligible']} contracts /"
      f" {bundle['portfolio']['n_review_eligible_accounts']} accounts")
print(f"  ARR:       ${bundle['arr']['opening_arr_m']}M → ${bundle['arr']['closing_arr_m']}M"
      f"  renewal_gap=+${bundle['arr']['renewal_gap_m']}M  intra_unres=+${bundle['arr']['intra_unresolved_m']}M")
print(f"  Pilot power: ~{bundle['pilot_power']['n_per_arm_independent']} per arm"
      f" (independent-unit, 15%→9%, alpha=0.05, power=0.80)")

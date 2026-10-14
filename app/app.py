"""
PANW Subscription Renewal Intelligence — Model Demo
Built from panw-case-study-model-demo outputs.

Pages
-----
1. Dashboard      — ARR bridge, GRR/NRR, model headline AP/AUC, 3-seed robustness
2. Portfolio      — April review cohort, policy comparison (single-seed + cross-seed)
3. Live Scoring   — Feature inputs → cancellation probability from full_context_hgb_refitted
4. Model Card     — Experiment table, feature importance, ablation, replication summary

Run:
    streamlit run app/app.py
    (from the panw-case-study-model-demo root)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
PART2   = ROOT / "part2"
MODELS  = PART2 / "models"
OUTPUTS = PART2 / "outputs"
REPLS   = ROOT / "output" / "model_demo_revision"

sys.path.insert(0, str(PART2 / "src"))
from _features import (  # noqa: E402
    add_indicator_cols, apply_imputation_values,
    FULL_CONTEXT_NUM,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_pct(v) -> str:
    try:
        return f"{float(v):.2%}"
    except (TypeError, ValueError):
        return "—"

def fmt_4f(v) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "—"

def fmt_dollar(v) -> str:
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return "—"

def fmt_cols(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Return a copy of df with columns formatted as strings per rules dict."""
    out = df.copy()
    for col, fn in rules.items():
        if col in out.columns:
            out[col] = out[col].apply(fn)
    return out

# ── Loaders ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODELS / "full_context_hgb_refitted.pkl")

@st.cache_data
def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def _load_all():
    return (
        load_json(OUTPUTS / "metrics.json"),
        load_json(OUTPUTS / "model_selection_record.json"),
        load_json(OUTPUTS / "imputation_values.json"),
        load_csv(OUTPUTS / "feature_importance.csv"),
        load_csv(OUTPUTS / "arr_bridge.csv"),
        load_csv(OUTPUTS / "renewal_review_contracts.csv"),
        load_csv(OUTPUTS / "renewal_review_policy_comparison.csv"),
        load_csv(OUTPUTS / "ablation.csv"),
        load_csv(REPLS   / "all_seed_test_metrics.csv"),
        load_csv(REPLS   / "all_seed_policy_results.csv"),
    )

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PANW Renewal Intelligence",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🔒 PANW Renewal Intel")
st.sidebar.caption("Model demo · Synthetic data")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Portfolio Review", "Live Scoring", "Model Card"],
)
st.sidebar.divider()
st.sidebar.caption(
    "Model: `full_context_hgb_refitted`  \n"
    "Seeds: 137, 991, 2027  \n"
    "Data: synthesised for case study"
)

metrics, sel_rec, imp_vals, fi, bridge, contracts, policy_comp, ablation, \
    seed_metrics, seed_policies = _load_all()
model = load_model()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("Revenue Health Dashboard")
    st.caption("FY Oct 2024 – Sep 2025 · Synthetic data")

    total   = bridge[bridge["segment"] == "total"].iloc[0]
    opening = total["opening_arr"] / 1e6
    closing = total["closing_arr"] / 1e6
    grr     = total["grr"]
    nrr     = total["nrr"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Opening ARR",  f"${opening:.1f}M")
    k2.metric("Closing ARR",  f"${closing:.1f}M",  delta=f"+${closing-opening:.1f}M")
    k3.metric("GRR",           f"{grr:.2%}",         delta=f"{grr-1:.2%}")
    k4.metric("NRR",           f"{nrr:.2%}",         delta=f"{nrr-1:.2%}")

    st.subheader("ARR Bridge")
    seg = bridge[bridge["segment"] == "total"].iloc[0]
    movements = [
        ("Opening ARR",                   seg["opening_arr"]),
        ("Account churn",                -seg.get("account_churn", 0)),
        ("Partial loss",                 -seg.get("partial_loss", 0)),
        ("Contraction",                   seg.get("contraction", 0)),
        ("Expansion",                     seg.get("expansion", 0)),
        ("New business",                  seg.get("new_business", 0)),
        ("Reactivation",                  seg.get("reactivation", 0)),
        ("Renewal gap",                   seg.get("renewal_gap", 0)),
        ("Intra-expansion unresolved",    seg.get("intra_expansion_unresolved", 0)),
        ("Closing ARR",                   seg["closing_arr"]),
    ]
    bdf = pd.DataFrame(movements, columns=["Movement", "Amount ($M)"])
    bdf["Amount ($M)"] = bdf["Amount ($M)"] / 1e6
    bdf["Amount ($M)"] = bdf["Amount ($M)"].apply(
        lambda v: f"+{v:.2f}" if v > 0 and v != opening and v != closing else f"{v:.2f}"
    )
    st.dataframe(bdf, use_container_width=True, height=370)

    st.subheader("GRR / NRR by Segment")
    seg_df = bridge[bridge["segment"] != "total"][["segment","opening_arr","grr","nrr"]].copy()
    seg_df = fmt_cols(seg_df, {
        "opening_arr": lambda v: f"${v/1e6:.1f}M",
        "grr": fmt_pct,
        "nrr": fmt_pct,
    })
    seg_df.columns = ["Segment","Opening ARR","GRR","NRR"]
    st.dataframe(seg_df, use_container_width=True)

    st.divider()
    st.subheader("Model Headline — Full Context HGB")

    hgb_ms   = metrics["full_context_hgb"]["model_selection"]
    refit_ft = metrics["full_context_hgb_refitted"]["final_test"]
    rule_ms  = metrics["rule_baseline"]["model_selection"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Refitted HGB AP (final test)",  f"{refit_ft['ap']:.4f}")
    m2.metric("Refitted HGB ROC-AUC",          f"{refit_ft['roc_auc']:.4f}")
    m3.metric("Rule baseline AP (MS)",         f"{rule_ms['ap']:.4f}",
              delta=f"{hgb_ms['ap'] - rule_ms['ap']:+.4f} HGB lift")
    brier = refit_ft.get("brier_score")
    m4.metric("Brier score (refitted FT)",     f"{brier:.4f}" if brier else "—")

    st.subheader("AP Robustness — 3 Random Seeds")
    ms_model = seed_metrics[seed_metrics["policy"] == "model_score"].copy()
    ms_rules = seed_metrics[seed_metrics["policy"] == "health_rules"].copy()

    r1, r2 = st.columns(2)
    with r1:
        st.caption("Model score AP across seeds")
        rd = fmt_cols(ms_model[["seed","ap","roc_auc"]].copy(),
                      {"ap": fmt_4f, "roc_auc": fmt_4f})
        rd.columns = ["Seed","AP","ROC-AUC"]
        st.dataframe(rd, use_container_width=True)
        st.caption(f"Mean AP {ms_model['ap'].mean():.4f} ± {ms_model['ap'].std():.4f}")

    with r2:
        st.caption("Health rules AP across seeds")
        rd2 = fmt_cols(ms_rules[["seed","ap","roc_auc"]].copy(),
                       {"ap": fmt_4f, "roc_auc": fmt_4f})
        rd2.columns = ["Seed","AP","ROC-AUC"]
        st.dataframe(rd2, use_container_width=True)
        st.caption(f"Mean AP {ms_rules['ap'].mean():.4f} ± {ms_rules['ap'].std():.4f}")

    st.info(
        "All metrics are on the **final-test split** (contracts expiring Apr–Oct 2026). "
        "Synthetic data only — do not generalise to live PANW cohorts."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PORTFOLIO REVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio Review":
    st.title("April 2026 Portfolio Review")
    st.caption("Seed 137 primary cohort · 3-seed cross-validation shown below")

    n_churn   = int(contracts["churned"].fillna(0).astype(int).sum())
    n_accts   = contracts["company_id"].nunique() if "company_id" in contracts.columns else "—"
    total_acv = contracts["acv"].sum() / 1e6 if "acv" in contracts.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Contracts in pool",      f"{len(contracts):,}")
    c2.metric("Unique accounts",        f"{n_accts:,}")
    c3.metric("Pool ACV",               f"${total_acv:.1f}M" if total_acv else "—")
    c4.metric("Observed cancellations", f"{n_churn} ({n_churn/len(contracts):.1%})")

    st.divider()
    st.subheader("Policy Comparison — Seed 137, Capacity = 30")

    cap30 = policy_comp[policy_comp["capacity"] == 30].copy()
    cap30["Precision [90% CI]"] = cap30.apply(
        lambda r: f"{r['precision_lead_churned']:.1%}  [{r['precision_ci_lo']:.1%}, {r['precision_ci_hi']:.1%}]",
        axis=1,
    )
    cap30["Cancelled ACV"] = cap30["cancelled_lead_acv"].apply(lambda v: f"${v/1e3:.0f}k")
    cap30 = cap30.rename(columns={
        "policy":                    "Policy",
        "n_accounts_reviewed":       "Leads",
        "n_lead_churned":            "Cancelled",
        "recall_churned_accounts":   "Recall (accts)",
        "pct_cancelled_acv_of_pool": "% Cancel ACV",
    })
    cap30["Recall (accts)"] = cap30["Recall (accts)"].apply(fmt_pct)
    cap30["% Cancel ACV"]   = cap30["% Cancel ACV"].apply(fmt_pct)
    st.dataframe(
        cap30[["Policy","Leads","Cancelled","Precision [90% CI]",
               "Recall (accts)","Cancelled ACV","% Cancel ACV"]],
        use_container_width=True, height=200,
    )
    st.caption(
        "Precision = cancelled leads / leads reviewed.  "
        "Cancelled ACV is subsequently observed, not predicted revenue saved."
    )

    st.subheader("Precision by Capacity (Seed 137)")
    pivot = policy_comp.pivot_table(
        index="capacity", columns="policy", values="precision_lead_churned"
    ).reset_index()
    st.line_chart(pivot.set_index("capacity"), height=280)

    st.divider()
    st.subheader("Cross-Seed Precision at Capacity = 30")

    sp30 = seed_policies[seed_policies["capacity"] == 30].copy()
    heat = sp30.pivot(index="policy", columns="seed", values="precision_lead_churned").copy()
    heat.columns = [f"Seed {c}" for c in heat.columns]
    for col in heat.columns:
        heat[col] = heat[col].apply(fmt_pct)
    st.dataframe(heat, use_container_width=True)

    agg = sp30.groupby("policy")["precision_lead_churned"].agg(["mean","std"]).reset_index()
    agg.columns = ["Policy","Mean precision","Std"]
    agg["Mean precision"] = agg["Mean precision"].apply(fmt_pct)
    agg["Std"]            = agg["Std"].apply(fmt_pct)
    st.dataframe(agg, use_container_width=True)
    st.caption("Mean ± std across seeds 137, 991, 2027 at capacity=30.")

    st.divider()
    st.subheader("Contract Details")

    fam_opts = ["All"] + sorted(contracts["product_family"].dropna().unique().tolist()) \
        if "product_family" in contracts.columns else ["All"]
    col_f, col_s = st.columns(2)
    fam = col_f.selectbox("Filter by product family", fam_opts)
    srt = col_s.selectbox("Sort by", ["model_score","acv","days_to_expiry_at_review"])

    disp = contracts.copy()
    if fam != "All":
        disp = disp[disp["product_family"] == fam]
    if srt in disp.columns:
        disp = disp.sort_values(srt, ascending=(srt == "days_to_expiry_at_review"))

    show_cols = [c for c in [
        "contract_id","company_id","product_family","size_segment",
        "acv","model_score","churned","days_to_expiry_at_review","term_end",
    ] if c in disp.columns]
    out = disp[show_cols].head(200).copy()
    out = fmt_cols(out, {
        "acv":   fmt_dollar,
        "model_score": fmt_4f,
        "days_to_expiry_at_review": lambda v: f"{v:.0f}",
    })
    st.dataframe(out, use_container_width=True, height=400)
    st.caption(f"Showing up to 200 of {len(disp):,} contracts.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LIVE SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Scoring":
    st.title("Live Contract Scoring")
    st.caption(
        "Inputs → `full_context_hgb_refitted` → cancellation probability.  "
        "All values represent the contract state 90 days before expiry."
    )

    PRODUCT_FAMILIES = ["Cortex","Prisma","Network Security","Cloud Security",
                        "Security Operations","AI-Powered SOC"]
    SIZE_SEGMENTS    = ["smb","midmarket","enterprise","strategic"]
    INDUSTRIES       = ["Technology","Financial Services","Healthcare","Retail",
                        "Government","Energy","Manufacturing","Other"]
    REGIONS          = ["AMER","EMEA","APAC","LATAM"]

    with st.form("score_form"):
        st.subheader("Deployment & Value Signals")
        a1, a2, a3 = st.columns(3)
        deployment_reach_30d      = a1.slider("Deployment reach (30d)", 0.0, 1.0, 0.5, 0.01)
        workflow_success_30d      = a2.slider("Workflow success (30d)", 0.0, 1.0, 0.6, 0.01,
                                              help="Top feature — most influential predictor")
        productive_use_delta_30d  = a3.number_input("Productive use delta (30d)", -0.5, 0.5, 0.0, 0.01)
        avg_coverage_90d          = a1.slider("Avg coverage (90d)", 0.0, 1.0, 0.7, 0.01,
                                              help="Second most influential feature")
        coverage_trend_90d        = a2.number_input("Coverage trend (90d)", -0.1, 0.1, 0.0, 0.001)
        obs_completeness_90d      = a3.slider("Obs completeness (90d)", 0.0, 1.0, 0.8, 0.01)

        st.subheader("Support & Service")
        b1, b2, b3 = st.columns(3)
        p1_tickets_90d                        = b1.number_input("P1 tickets (90d)", 0, 50, 0)
        open_tickets_90d                      = b2.number_input("Open tickets (90d)", 0, 200, 2)
        tickets_90d                           = b3.number_input("Total tickets (90d)", 0, 500, 5)
        repeat_sla_breaches_90d               = b1.number_input("Repeat SLA breaches (90d)", 0, 20, 0)
        unresolved_customer_impact_hours_90d  = b2.number_input("Unresolved impact hrs (90d)", 0.0, 5000.0, 0.0)
        avg_open_age_hrs_90d                  = b3.number_input("Avg open ticket age (hrs)", 0.0, 2000.0, 48.0)
        avg_resolution_hrs_90d                = b1.number_input("Avg resolution hrs", 0.0, 5000.0, 24.0)

        st.subheader("Relationship & Commercial")
        c1, c2, c3 = st.columns(3)
        days_since_last_interaction     = c1.number_input("Days since last interaction", 0, 730, 45)
        executive_interactions_180d     = c2.number_input("Exec interactions (180d)", 0, 50, 2)
        marketing_engagements_180d      = c3.number_input("Marketing engagements (180d)", 0, 50, 1)
        overdue_agreed_actions          = c1.number_input("Overdue agreed actions", 0, 20, 0)
        days_without_confirmed_owner    = c2.number_input("Days without confirmed owner", 0, 730, 0)
        quoted_unit_price_change_pct    = c3.number_input("Quoted unit price change (%)",
                                                           -20.0, 20.0, 2.0, 0.1)
        dated_budget_concern            = c1.selectbox("Budget concern recorded", [0, 1])
        median_prior_renewal_delay_days = c2.number_input("Median prior renewal delay (days)", 0, 365, 0)

        st.subheader("Commercial & Portfolio Context")
        d1, d2, d3 = st.columns(3)
        product_family      = d1.selectbox("Product family", PRODUCT_FAMILIES)
        size_segment        = d2.selectbox("Size segment", SIZE_SEGMENTS, index=1)
        industry            = d3.selectbox("Industry", INDUSTRIES)
        region              = d1.selectbox("Region", REGIONS)
        acv                 = d2.number_input("ACV ($)", 10_000, 5_000_000, 200_000, 1_000)
        total_acv_in_window = d3.number_input("Total ACV in window ($)", 10_000, 20_000_000, 400_000, 1_000)
        employee_count      = d1.number_input("Employee count", 1, 500_000, 5_000)
        n_products_in_window  = d2.number_input("Products in window", 1, 20, 2)
        n_families_in_window  = d3.number_input("Families in window", 1, 10, 1)

        st.subheader("Breadth & Operational Context")
        e1, e2, e3 = st.columns(3)
        n_active_products_in_family      = e1.number_input("Active products in family", 0, 10, 1)
        family_sibling_cancellations_1yr = e2.number_input("Family sibling cancellations (1yr)", 0, 10, 0)
        family_adoption_completeness     = e3.slider("Family adoption completeness", 0.0, 1.0, 0.5, 0.01)
        days_to_renewal                  = e1.number_input("Days to renewal", 0, 365, 90)
        tenure_days                      = e2.number_input("Tenure (days)", 0, 5000, 730)
        policy_events_per_unit_90d       = e3.number_input("Policy events per unit (90d)", 0.0, 100.0, 5.0)
        remediation_velocity_90d         = e1.number_input("Remediation velocity (90d)", 0.0, 50.0, 2.0)
        automation_adoption_90d          = e2.slider("Automation adoption (90d)", 0.0, 1.0, 0.3, 0.01)
        scan_success_rate_90d            = e3.slider("Scan success rate (90d)", 0.0, 1.0, 1.0, 0.01,
                                                     help="Set 1.0 for non-Prisma products")

        submitted = st.form_submit_button("Score →", type="primary", use_container_width=True)

    if submitted:
        row = {
            "deployment_reach_30d":               deployment_reach_30d,
            "onboarding_delay_days":              0.0,
            "workflow_success_30d":               workflow_success_30d,
            "productive_use_delta_30d":           productive_use_delta_30d,
            "unresolved_customer_impact_hours_90d": unresolved_customer_impact_hours_90d,
            "repeat_sla_breaches_90d":            float(repeat_sla_breaches_90d),
            "p1_tickets_90d":                     float(p1_tickets_90d),
            "open_tickets_90d":                   float(open_tickets_90d),
            "avg_open_age_hrs_90d":               avg_open_age_hrs_90d,
            "avg_resolution_hrs_90d":             avg_resolution_hrs_90d,
            "avg_coverage_90d":                   avg_coverage_90d,
            "coverage_trend_90d":                 coverage_trend_90d,
            "obs_completeness_90d":               obs_completeness_90d,
            "quoted_unit_price_change_pct":       quoted_unit_price_change_pct,
            "dated_budget_concern":               float(dated_budget_concern),
            "days_without_confirmed_owner":       float(days_without_confirmed_owner),
            "overdue_agreed_actions":             float(overdue_agreed_actions),
            "days_to_renewal":                    float(days_to_renewal),
            "tenure_days":                        float(tenure_days),
            "employee_count":                     float(employee_count),
            "acv":                                float(acv),
            "total_acv_in_window":                float(total_acv_in_window),
            "n_products_in_window":               float(n_products_in_window),
            "n_families_in_window":               float(n_families_in_window),
            "n_active_products_in_family":        float(n_active_products_in_family),
            "family_sibling_cancellations_1yr":   float(family_sibling_cancellations_1yr),
            "family_adoption_completeness":       family_adoption_completeness,
            "executive_interactions_180d":        float(executive_interactions_180d),
            "marketing_engagements_180d":         float(marketing_engagements_180d),
            "days_since_last_interaction":        float(days_since_last_interaction),
            "median_prior_renewal_delay_days":    float(median_prior_renewal_delay_days),
            "tickets_90d":                        float(tickets_90d),
            "policy_events_per_unit_90d":         policy_events_per_unit_90d,
            "remediation_velocity_90d":           remediation_velocity_90d,
            "automation_adoption_90d":            automation_adoption_90d,
            "scan_success_rate_90d":              scan_success_rate_90d,
            "product_family":                     product_family,
            "size_segment":                       size_segment,
            "industry":                           industry,
            "region":                             region,
        }
        df_in = pd.DataFrame([row])
        df_in = add_indicator_cols(df_in)
        df_in = apply_imputation_values(df_in, imp_vals)
        prob  = model.predict_proba(df_in)[0, 1]

        if prob >= 0.25:
            tier, icon, advice = "HIGH",   "🔴", "Immediate CSM outreach"
        elif prob >= 0.12:
            tier, icon, advice = "MEDIUM", "🟠", "Include in next review cycle"
        else:
            tier, icon, advice = "LOW",    "🟢", "Routine monitoring"

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Cancellation probability", f"{prob:.3f}")
        r2.metric("Risk tier",                f"{icon} {tier}")
        r3.metric("Recommendation",           advice)
        st.progress(min(prob / 0.5, 1.0),
                    text=f"Score {prob:.3f}  ·  pool prevalence ≈4.5%")

        fi_top = fi[fi["model"] == "full_context_hgb"].sort_values(
            "importance_mean", ascending=False
        ).head(10)
        if not fi_top.empty:
            st.subheader("Top 10 Features (permutation importance)")
            st.bar_chart(fi_top.set_index("feature_name")["importance_mean"], horizontal=True)

        if "model_score" in contracts.columns:
            pct_rank = (contracts["model_score"] < prob).mean()
            st.caption(
                f"This score is higher than **{pct_rank:.0%}** of the April review pool "
                f"({len(contracts):,} contracts)."
            )

        st.caption(
            "⚠️ Synthetic data only. Probabilities reflect the case-study cohort, "
            "not live PANW renewal outcomes."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL CARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Card":
    st.title("Model Card")
    st.caption("`full_context_hgb_refitted.pkl` · Selected on MS AP · Evaluated on final-test split")

    st.subheader("Experiment AP Comparison")
    exp_order = [
        "prevalence_baseline","rule_baseline",
        "experience_led_lr","experience_led_hgb",
        "size_value_lr","size_value_hgb",
        "full_context_lr","full_context_hgb",
    ]
    rows = []
    for k in exp_order:
        if k not in metrics:
            continue
        ms = metrics[k].get("model_selection", {})
        ft = metrics[k].get("final_test", {})
        rows.append({
            "Experiment":  k,
            "MS AP":       fmt_4f(ms.get("ap")),
            "MS AUC":      fmt_4f(ms.get("roc_auc")),
            "FT AP":       fmt_4f(ft.get("ap")),
            "FT AUC":      fmt_4f(ft.get("roc_auc")),
        })
    refit_ft = metrics.get("full_context_hgb_refitted", {}).get("final_test", {})
    rows.append({
        "Experiment": "full_context_hgb_refitted ★",
        "MS AP": "—", "MS AUC": "—",
        "FT AP":  fmt_4f(refit_ft.get("ap")),
        "FT AUC": fmt_4f(refit_ft.get("roc_auc")),
    })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=330)

    st.subheader("3-Seed Replication Results")
    seed_disp = seed_metrics.copy()
    seed_disp["ap"]        = seed_disp["ap"].apply(fmt_4f)
    seed_disp["roc_auc"]   = seed_disp["roc_auc"].apply(fmt_4f)
    seed_disp["prevalence"]= seed_disp["prevalence"].apply(fmt_4f)
    seed_disp.columns      = [c.upper() for c in seed_disp.columns]
    st.dataframe(seed_disp, use_container_width=True)
    st.caption("Independent train/test splits per seed — tests AP stability.")

    st.divider()
    st.subheader("Permutation Feature Importance (top 20)")
    fi_top20 = fi[fi["model"] == "full_context_hgb"].sort_values("importance_mean", ascending=False).head(20)
    if not fi_top20.empty:
        st.bar_chart(fi_top20.set_index("feature_name")["importance_mean"], horizontal=True, height=420)
    st.caption(
        "**workflow_success_30d** and **avg_coverage_90d** are the dominant drivers — "
        "workflow health explains most of the AP lift over the rule baseline."
    )

    st.subheader("Block Ablation")
    st.caption("Negative ap_drop = removing that block increases AP (block hurts on this cohort).")
    abl_disp = ablation[["block","baseline_ap","ablated_ap","ap_drop"]].copy()
    abl_disp["baseline_ap"] = abl_disp["baseline_ap"].apply(fmt_4f)
    abl_disp["ablated_ap"]  = abl_disp["ablated_ap"].apply(fmt_4f)
    abl_disp["ap_drop"]     = abl_disp["ap_drop"].apply(lambda v: f"{v:+.4f}")
    st.dataframe(abl_disp, use_container_width=True)

    st.subheader("Model Specification")
    sel = sel_rec.get("selected_model", {})
    for label, val in [
        ("Experiment",       sel_rec.get("deployment_policy")),
        ("Config",           sel.get("config")),
        ("Num features",     len(sel.get("num_features", []))),
        ("Cat features",     len(sel.get("cat_features", []))),
        ("MS AP",            f"{sel.get('ms_ap', 0):.6f}"),
        ("Artifact",         sel_rec.get("refit", {}).get("artifact")),
        ("Selection split",  "model_selection (2025-04-01 cut)"),
        ("Final-test split", "2026-04-01 cut"),
    ]:
        st.text(f"  {label:<22} {val}")

    with st.expander("Numeric feature list"):
        for f_name in sel.get("num_features", []):
            st.text(f"  {f_name}")
    with st.expander("Categorical feature list"):
        for f_name in sel.get("cat_features", []):
            st.text(f"  {f_name}")

    st.subheader("Caveats")
    st.warning(
        "**Synthetic data.** AP and AUC figures reflect the case-study cohort only.  \n\n"
        "**Retrospective comparison.** Precision figures compare policies against "
        "subsequently observed cancellations — not prevented cancellations.  \n\n"
        "**Value block dominance.** workflow_success_30d and avg_coverage_90d account "
        "for most of the AP; other blocks contribute marginally on this cohort.  \n\n"
        "**3 seeds only.** AP range is ~0.22–0.25 across seeds; "
        "more seeds would narrow the estimate."
    )

"""
WP3 — Retention and ARR bridge analysis.

Produces descriptive analytics showing where revenue is being lost and retained.
Consumes outputs from build_features.py (WP2).

Outputs
-------
part2/outputs/cohort_retention.csv          — churn rates by quarter/family/segment
part2/outputs/arr_bridge.csv                — ARR movement bridge (total + by size_segment)
part2/outputs/partial_loss.csv              — companies with partial product cancellations
part2/outputs/value_concentration.csv       — Lorenz-style ACV concentration
part2/outputs/breadth_retention_assoc.csv   — product breadth vs retention (association only)

Figures
-------
part2/figures/cohort_retention_by_family.png — bar chart: churn rate by product_family
part2/figures/arr_bridge.png                 — waterfall: ARR movements
part2/figures/value_concentration.png        — Lorenz curve: ACV concentration
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Business-logic library (importable, no side-effects)
sys.path.insert(0, str(Path(__file__).parent))
from _arr_bridge import (
    build_period_bridge, build_lineage_reconciliation,
    DEFAULT_PERIOD_START, DEFAULT_PERIOD_END,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
CLEANED = ROOT / "data" / "cleaned"
ANALYSIS = ROOT / "data" / "analysis"
PART2_OUT = ROOT / "part2" / "outputs"
FIGURES = ROOT / "part2" / "figures"

PART2_OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PANW brand colours & plot defaults
# ---------------------------------------------------------------------------
PANW_BLUE = "#003087"
PANW_RED = "#E1251B"
PANW_TEAL = "#00A2A1"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("WP3 — analyze_cohorts.py")
print("=" * 70)

cohort_path = ANALYSIS / "renewal_cohort_subscriptions.csv"
if not cohort_path.exists():
    print(
        f"\nERROR: {cohort_path} not found.\n"
        "Run part2/src/build_features.py to generate the per-contract modeling cohort.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"\n[1] Loading current per-contract renewal_cohort_subscriptions.csv from {ANALYSIS}")
cohort = pd.read_csv(cohort_path, low_memory=False)
print(f"    Loaded {len(cohort):,} rows, {cohort.shape[1]} columns")

print(f"\n[2] Loading company_summary.csv from {CLEANED}")
cs = pd.read_csv(CLEANED / "company_summary.csv", low_memory=False)
print(f"    Loaded {len(cs):,} rows, {cs.shape[1]} columns")

print(f"\n[2b] Loading order_item_fact.csv from {CLEANED}")
oif = pd.read_csv(CLEANED / "order_item_fact.csv", low_memory=False)
print(f"     Loaded {len(oif):,} rows, {oif.shape[1]} columns")

# Parse dates
for col in ("term_start", "term_end", "order_date"):
    oif[col] = pd.to_datetime(oif[col], errors="coerce")

oif_sub = oif[oif["offering_type"] == "subscription"].copy()

# Join size_segment from company_summary so per-segment bridges can filter oif_sub directly
seg_map = cs.set_index("company_id")["size_segment"].to_dict()
oif_sub["size_segment"] = oif_sub["company_id"].map(seg_map)

# Coerce types
cohort["term_end"] = pd.to_datetime(cohort["term_end"], errors="coerce")
cohort["churned"] = pd.to_numeric(cohort["churned"], errors="coerce")

# ---------------------------------------------------------------------------
# Analysis 1: Retention cohorts
# ---------------------------------------------------------------------------
print("\n[3] Analysis 1 — Retention cohorts by quarter / family / segment")

sub = cohort[
    (cohort["offering_type"] == "subscription")
    & cohort["churned"].notna()
].copy()
print(f"    Subscription-eligible rows: {len(sub):,}")

# Renewal quarter from term_end
sub["renewal_quarter"] = (
    sub["term_end"].dt.to_period("Q").astype(str)
)

GROUP_COLS = ["renewal_quarter", "product_family", "size_segment"]

retention_rows = []
for keys, grp in sub.groupby(GROUP_COLS, dropna=False):
    n = len(grp)
    n_pos = int(grp["churned"].sum())
    rate = n_pos / n if n > 0 else np.nan
    total_acv = grp["acv"].sum() if "acv" in grp.columns else np.nan

    if n < 5:
        rate_display = "suppressed — n<5"
        n_pos_display = "suppressed — n<5"
    else:
        rate_display = round(rate, 4)
        n_pos_display = n_pos

    row = dict(zip(GROUP_COLS, keys if isinstance(keys, tuple) else (keys,)))
    row.update(
        {
            "n_contracts": n,
            "n_positives": n_pos_display,
            "churn_rate": rate_display,
            "total_acv": round(total_acv, 2) if not np.isnan(total_acv) else np.nan,
        }
    )
    retention_rows.append(row)

retention_df = pd.DataFrame(retention_rows)
out_ret = PART2_OUT / "cohort_retention.csv"
retention_df.to_csv(out_ret, index=False)
print(f"    Saved {len(retention_df):,} group rows -> part2/outputs/cohort_retention.csv")

# Quick summary: groups suppressed vs shown
n_shown = (retention_df["churn_rate"] != "suppressed — n<5").sum()
n_supp = (retention_df["churn_rate"] == "suppressed — n<5").sum()
print(f"      Groups shown     : {n_shown:>5,}")
print(f"      Groups suppressed: {n_supp:>5,}")

# ---------------------------------------------------------------------------
# Analysis 2: ARR movement bridge
# Reporting period: Study Reporting Year (2024-10-01 → 2025-09-30)
# All components derived from subscription OIF for the same period and scope.
# GRR/NRR denominator = opening cohort (accounts active at period start).
# Movements restricted to the opening cohort and the reporting window.
# ---------------------------------------------------------------------------
print("\n[4] Analysis 2 — ARR movement bridge (Study Reporting Year: 2024-10-01 → 2025-09-30)")
print(f"    Reporting period: {DEFAULT_PERIOD_START.date()} → {DEFAULT_PERIOD_END.date()}")
print(f"    Scope: subscription OIF only (consistent across opening, movements, closing)")

# ---- Acceptance check 1: contraction renewal → GRR=NRR=80%, residual=0 ----
_ps = DEFAULT_PERIOD_START
_pe = DEFAULT_PERIOD_END
_ren_date = _ps + pd.Timedelta(days=30)  # strictly after period_start
test_oif_1 = pd.DataFrame([
    {"company_id":"T1","contract_id":"T_PRED","predecessor_contract_id":None,
     "net_acv":100.,"prior_net_acv":None,
     "term_start":_ps-pd.Timedelta(days=180),"term_end":_ps+pd.Timedelta(days=180),
     "order_type":"new_logo","order_date":_ps-pd.Timedelta(days=180),"offering_type":"subscription"},
    {"company_id":"T1","contract_id":"T_CURR","predecessor_contract_id":"T_PRED",
     "net_acv":80.,"prior_net_acv":100.,
     "term_start":_ren_date,"term_end":_ren_date+pd.Timedelta(days=365),
     "order_type":"renewal_contraction","order_date":_ren_date,"offering_type":"subscription"},
])
test_oif_1["size_segment"] = "test"
_tr1 = build_period_bridge(test_oif_1, "ac1", _ps, _pe)
assert abs(_tr1["grr"] - 0.80) < 1e-6, f"GRR check failed: {_tr1['grr']}"
assert abs(_tr1["nrr"] - 0.80) < 1e-6, f"NRR check failed: {_tr1['nrr']}"
assert abs(_tr1["residual"]) < 0.01,   f"Residual check failed: {_tr1['residual']}"
print("    ACCEPTANCE CHECK 1 PASSED: $100→$80 renewal → GRR=80%, NRR=80%, residual≈0")

# ---- Acceptance check 2: partial product loss → GRR=NRR=66.67%, residual=0 --
# Company T2: two subscriptions ($100+$50). Only $100 survives to closing.
# GRR = NRR = $100/$150 = 66.67%.
test_oif_2 = pd.DataFrame([
    {"company_id":"T2","contract_id":"SUB_A","predecessor_contract_id":None,
     "net_acv":100.,"prior_net_acv":None,
     "term_start":_ps-pd.Timedelta(days=180),"term_end":_ps+pd.Timedelta(days=400),
     "order_type":"new_logo","order_date":_ps-pd.Timedelta(days=180),"offering_type":"subscription"},
    {"company_id":"T2","contract_id":"SUB_B","predecessor_contract_id":None,
     "net_acv":50.,"prior_net_acv":None,
     "term_start":_ps-pd.Timedelta(days=180),"term_end":_ps+pd.Timedelta(days=180),
     "order_type":"new_logo","order_date":_ps-pd.Timedelta(days=180),"offering_type":"subscription"},
])
test_oif_2["size_segment"] = "test"
_tr2 = build_period_bridge(test_oif_2, "ac2", _ps, _pe)
assert abs(_tr2["grr"] - (100./150.)) < 1e-4, f"GRR partial-loss check failed: {_tr2['grr']}"
assert abs(_tr2["nrr"] - (100./150.)) < 1e-4, f"NRR partial-loss check failed: {_tr2['nrr']}"
assert abs(_tr2["residual"]) < 0.01,           f"Residual partial-loss check failed: {_tr2['residual']}"
print("    ACCEPTANCE CHECK 2 PASSED: partial product loss → GRR=NRR=66.67%, residual≈0")

# ---- Acceptance check 3: opening-day renewal must NOT double-count ----------
# Old contract expires exactly at period_start (term_end = period_start, NOT in opening).
# Renewal starts on period_start (in opening). ACV goes $100 → $120.
# Opening = closing = $120. Expected: GRR = NRR = 100%, expansion = 0, residual = 0.
test_oif_3 = pd.DataFrame([
    {"company_id":"T3","contract_id":"OLD","predecessor_contract_id":None,
     "net_acv":100.,"prior_net_acv":None,
     "term_start":_ps-pd.Timedelta(days=365),"term_end":_ps,   # term_end == ps → NOT in opening
     "order_type":"new_logo","order_date":_ps-pd.Timedelta(days=365),"offering_type":"subscription"},
    {"company_id":"T3","contract_id":"RENEW","predecessor_contract_id":"OLD",
     "net_acv":120.,"prior_net_acv":100.,
     "term_start":_ps,"term_end":_ps+pd.Timedelta(days=365),   # in opening AND in closing
     "order_type":"renewal_expansion","order_date":_ps,    # order_date == ps → excluded from movements
     "offering_type":"subscription"},
])
test_oif_3["size_segment"] = "test"
_tr3 = build_period_bridge(test_oif_3, "ac3", _ps, _pe)
assert abs(_tr3["opening_arr"] - 120.) < 0.01, f"Opening check: {_tr3['opening_arr']}"
assert abs(_tr3["closing_arr"] - 120.) < 0.01, f"Closing check: {_tr3['closing_arr']}"
assert abs(_tr3["expansion"]) < 0.01,           f"Expansion should be 0, got {_tr3['expansion']}"
assert abs(_tr3["grr"] - 1.0) < 1e-6,           f"GRR should be 100%, got {_tr3['grr']}"
assert abs(_tr3["residual"]) < 0.01,            f"Residual should be 0, got {_tr3['residual']}"
print("    ACCEPTANCE CHECK 3 PASSED: opening-day renewal → no double-count, GRR=100%")

# ---- Build per-segment bridge -----------------------------------------------
bridge_rows = [build_period_bridge(oif_sub, "total")]
for seg, seg_oif in oif_sub.groupby("size_segment", dropna=False):
    bridge_rows.append(build_period_bridge(seg_oif.copy(), str(seg)))

bridge_df = pd.DataFrame(bridge_rows)
out_bridge = PART2_OUT / "arr_bridge.csv"
bridge_df.to_csv(out_bridge, index=False)
print(f"    Saved {len(bridge_df):,} rows -> part2/outputs/arr_bridge.csv")

total_row = bridge_df[bridge_df["segment"] == "total"].iloc[0]
resid_pct = total_row["residual"] / total_row["opening_arr"] * 100 if total_row["opening_arr"] else float("nan")
print(f"    residual = {resid_pct:.1f}% of opening ARR")
print(f"    (see lineage_reconciliation.csv for contract-level fate breakdown)")

print(f"\n    ARR Bridge Summary (Study Reporting Year):")
hdr = (f"      {'Segment':<20} {'Opening':>12} {'AcctChurn':>11} {'PartLoss':>10} "
       f"{'Exp':>10} {'Con':>10} {'NewBiz':>10} {'Reactiv':>9} {'Closing':>12} "
       f"{'Resid':>9} {'GRR':>7} {'NRR':>8} {'cNRR':>7}")
print(hdr)
print("      " + "-" * len(hdr))
for _, row in bridge_df.iterrows():
    grr_s  = f"{row['grr']:.1%}"          if not (isinstance(row['grr'], float)          and np.isnan(row['grr']))          else "N/A"
    nrr_s  = f"{row['nrr']:.1%}"          if not (isinstance(row['nrr'], float)          and np.isnan(row['nrr']))          else "N/A"
    cnrr_s = f"{row['contract_nrr']:.1%}" if not (isinstance(row['contract_nrr'], float) and np.isnan(row['contract_nrr'])) else "N/A"
    print(
        f"      {row['segment']:<20} {row['opening_arr']:>12,.0f} "
        f"{row['account_churn_loss']:>11,.0f} {row['partial_loss']:>10,.0f} "
        f"{row['expansion']:>10,.0f} {row['contraction']:>10,.0f} "
        f"{row['new_business']:>10,.0f} {row['reactivation']:>9,.0f} "
        f"{row['closing_arr']:>12,.0f} {row['residual']:>9,.0f} "
        f"{grr_s:>7} {nrr_s:>8} {cnrr_s:>7}"
    )

# ---- Lineage reconciliation table -------------------------------------------
print("\n    Building lineage reconciliation table...")
lineage_recon = build_lineage_reconciliation(oif_sub, _ps, _pe)
out_lineage = PART2_OUT / "lineage_reconciliation.csv"
lineage_recon.to_csv(out_lineage, index=False)

fate_summary = lineage_recon.groupby("fate").agg(
    n_contracts=("contract_id", "count"),
    sum_opening_acv=("opening_net_acv", "sum"),
    sum_closing_acv=("closing_net_acv", "sum"),
).reset_index()
print(f"    Lineage reconciliation ({len(lineage_recon):,} rows):")
for _, fr in fate_summary.iterrows():
    open_v = f"${fr['sum_opening_acv']:,.0f}" if not np.isnan(fr['sum_opening_acv']) else "—"
    close_v = f"${fr['sum_closing_acv']:,.0f}" if not np.isnan(fr['sum_closing_acv']) else "—"
    print(f"      {fr['fate']:<40} n={fr['n_contracts']:>5,}  opening={open_v:<14} closing={close_v}")

# Churn evidence breakdown for loss fates
loss_rows = lineage_recon[lineage_recon["fate"].isin(["account_churn", "partial_loss"])].copy()
if not loss_rows.empty and "churn_evidence" in loss_rows.columns:
    ev_summary = loss_rows.groupby(["fate", "churn_evidence"]).agg(
        n=("contract_id", "count"),
        acv=("opening_net_acv", "sum"),
    ).reset_index()
    print(f"\n    Churn evidence breakdown (snapshot loss ≠ confirmed churn):")
    for _, ev in ev_summary.iterrows():
        print(f"      {ev['fate']:<18} {ev['churn_evidence']:<28} n={ev['n']:>4}  ACV=${ev['acv']:>12,.0f}")
    n_unconfirmed = int(loss_rows[loss_rows["churn_evidence"] == "snapshot_gap_only"].shape[0])
    n_subsequently = int(loss_rows[loss_rows["churn_evidence"] == "subsequently_renewed"].shape[0])
    print(f"\n    NOTE: {n_unconfirmed} loss contracts have no cancellation event or post-period renewal")
    print(f"          (snapshot_gap_only). {n_subsequently} subsequently renewed after period_end.")
    print(f"          See analysis_contract.md: 'No confirmed churn from snapshot absence'.")
print(f"    Saved -> part2/outputs/lineage_reconciliation.csv")

# ---------------------------------------------------------------------------
# Analysis 3: Partial loss
# ---------------------------------------------------------------------------
print("\n[5] Analysis 3 — Partial loss analysis")

# Partial loss: has cancellations AND still has positive remaining active ACV
partial = cs[(cs["has_cancellation"] == 1) & (cs["total_active_acv"] > 0)].copy()
# Complete loss: has cancellations AND zero remaining active ACV
complete_loss = cs[(cs["has_cancellation"] == 1) & (cs["total_active_acv"] == 0)].copy()

n_partial = len(partial)
n_complete_loss = len(complete_loss)
partial_acv = partial["total_active_acv"].sum()
total_book_acv = cs["total_active_acv"].sum()
pct_of_book = partial_acv / total_book_acv if total_book_acv != 0 else np.nan

# Save both groups to partial_loss.csv with a type column
partial["loss_type"] = "partial_loss"
complete_loss["loss_type"] = "complete_loss"
partial_combined = pd.concat([partial, complete_loss], ignore_index=True)

out_partial = PART2_OUT / "partial_loss.csv"
partial_combined.to_csv(out_partial, index=False)
print(f"    Companies with partial loss (cancellation AND positive remaining ACV): {n_partial:,}")
print(f"    Companies with complete loss (cancellation AND zero remaining ACV)   : {n_complete_loss:,}")
print(f"    Partial loss total ACV : ${partial_acv:,.0f}")
print(f"    % of total book        : {pct_of_book:.1%}" if not np.isnan(pct_of_book) else "    % of total book        : N/A")
print(f"    Saved both groups -> part2/outputs/partial_loss.csv")

# ---------------------------------------------------------------------------
# Analysis 4: Value concentration (Lorenz)
# ---------------------------------------------------------------------------
print("\n[6] Analysis 4 — Value concentration (Lorenz)")

acv_sorted = cs["total_active_acv"].dropna().sort_values().reset_index(drop=True)
cumulative_acv = acv_sorted.cumsum()
total_acv_sum = acv_sorted.sum()
n_companies = len(acv_sorted)

# Population quantiles
pct_companies = np.arange(1, n_companies + 1) / n_companies
pct_acv = cumulative_acv / total_acv_sum if total_acv_sum > 0 else cumulative_acv

# Concentration at top 10%, 20%, 50%
def top_x_pct(pct_threshold: float) -> float:
    """Fraction of total ACV held by top `pct_threshold` of accounts."""
    n_top = max(1, int(np.ceil(n_companies * pct_threshold)))
    top_acv = acv_sorted.tail(n_top).sum()
    return top_acv / total_acv_sum if total_acv_sum > 0 else np.nan


top10_pct = top_x_pct(0.10)
top20_pct = top_x_pct(0.20)
top50_pct = top_x_pct(0.50)

lorenz_df = pd.DataFrame(
    {
        "pct_companies_ranked_low_to_high": np.round(pct_companies, 6),
        "cumulative_pct_acv": np.round(pct_acv.values, 6),
    }
)
# Add concentration summary rows at end
summary_rows = pd.DataFrame(
    [
        {
            "pct_companies_ranked_low_to_high": "top_10pct_accounts",
            "cumulative_pct_acv": round(top10_pct, 4),
        },
        {
            "pct_companies_ranked_low_to_high": "top_20pct_accounts",
            "cumulative_pct_acv": round(top20_pct, 4),
        },
        {
            "pct_companies_ranked_low_to_high": "top_50pct_accounts",
            "cumulative_pct_acv": round(top50_pct, 4),
        },
    ]
)
conc_out = pd.concat([lorenz_df, summary_rows], ignore_index=True)
out_conc = PART2_OUT / "value_concentration.csv"
conc_out.to_csv(out_conc, index=False)
print(f"    Top 10% of accounts hold {top10_pct:.1%} of total ACV")
print(f"    Top 20% of accounts hold {top20_pct:.1%} of total ACV")
print(f"    Top 50% of accounts hold {top50_pct:.1%} of total ACV")
print(f"    Saved Lorenz data ({len(lorenz_df):,} rows + 3 summary) -> part2/outputs/value_concentration.csv")

# ---------------------------------------------------------------------------
# Analysis 5: Product breadth association (NOT causal)
# ---------------------------------------------------------------------------
print("\n[7] Analysis 5 — Product breadth vs retention (ASSOCIATION ONLY)")
print("    NOTE: breadth/retention association — selection effects may explain it")

def breadth_bucket(n: float) -> str:
    if pd.isna(n):
        return "unknown"
    n = int(n)
    if n == 0:
        return "0"
    elif n == 1:
        return "1"
    elif n <= 3:
        return "2-3"
    else:
        return "4+"


cs["n_families_bucket"] = cs["n_families_active"].apply(breadth_bucket)

breadth_order = ["0", "1", "2-3", "4+"]

breadth_rows = []
for bucket in breadth_order:
    grp = cs[cs["n_families_bucket"] == bucket]
    if len(grp) == 0:
        continue
    row = {
        "n_families_bucket": bucket,
        "n_companies": len(grp),
        "mean_grr": round(grp["grr"].mean(), 4) if grp["grr"].notna().any() else np.nan,
        "mean_nrr": round(grp["nrr"].mean(), 4) if grp["nrr"].notna().any() else np.nan,
        "pct_with_cancellation": round(grp["has_cancellation"].mean(), 4),
        "note": "association only — selection effects may explain differences",
    }
    breadth_rows.append(row)

breadth_df = pd.DataFrame(breadth_rows)
out_breadth = PART2_OUT / "breadth_retention_assoc.csv"
breadth_df.to_csv(out_breadth, index=False)

print(f"\n    {'Bucket':<8} {'N':>6} {'Mean GRR':>10} {'Mean NRR':>10} {'Pct w/ Cancel':>14}")
print("    " + "-" * 54)
for _, row in breadth_df.iterrows():
    grr_str = f"{row['mean_grr']:.4f}" if not (isinstance(row['mean_grr'], float) and np.isnan(row['mean_grr'])) else "N/A"
    nrr_str = f"{row['mean_nrr']:.4f}" if not (isinstance(row['mean_nrr'], float) and np.isnan(row['mean_nrr'])) else "N/A"
    print(
        f"    {row['n_families_bucket']:<8} {row['n_companies']:>6,} "
        f"{grr_str:>10} {nrr_str:>10} {row['pct_with_cancellation']:>14.1%}"
    )
print("    breadth/retention association — selection effects may explain it")
print(f"    Saved -> part2/outputs/breadth_retention_assoc.csv")

# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
print("\n[8] Generating figures")

# ---- Figure 1: Churn rate by product_family --------------------------------
print("    Figure 1: cohort_retention_by_family.png")

# Compute family-level churn from subscription cohort (using numeric values)
family_stats = (
    sub.groupby("product_family", dropna=False)
    .agg(n_contracts=("churned", "count"), n_positives=("churned", "sum"))
    .reset_index()
)
family_stats["churn_rate"] = family_stats["n_positives"] / family_stats["n_contracts"]
family_stats = family_stats[family_stats["n_contracts"] >= 5].sort_values(
    "churn_rate", ascending=False
)

fig1, ax1 = plt.subplots(figsize=(9, 5))
bars = ax1.bar(
    range(len(family_stats)),
    family_stats["churn_rate"],
    color=PANW_RED,
    edgecolor="white",
    linewidth=0.6,
    alpha=0.90,
)
ax1.set_xticks(range(len(family_stats)))
ax1.set_xticklabels(
    [
        f"{fam}\n(n={n:,})"
        for fam, n in zip(family_stats["product_family"], family_stats["n_contracts"])
    ],
    fontsize=9,
)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax1.set_ylabel("Churn Rate")
ax1.set_title("Subscription Churn Rate by Product Family\n(denominator n shown below bar)", color=PANW_BLUE)
ax1.axhline(
    family_stats["churn_rate"].mean(),
    color=PANW_TEAL,
    linestyle="--",
    linewidth=1.2,
    label=f"Overall avg {family_stats['churn_rate'].mean():.1%}",
)
ax1.legend(fontsize=9)
# Annotate rate on each bar
for bar, rate in zip(bars, family_stats["churn_rate"]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{rate:.1%}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=PANW_BLUE,
    )
fig1.tight_layout()
fig1.savefig(FIGURES / "cohort_retention_by_family.png", bbox_inches="tight")
plt.close(fig1)
print(f"      Saved -> part2/figures/cohort_retention_by_family.png")

# ---- Figure 2: ARR bridge waterfall ----------------------------------------
print("    Figure 2: arr_bridge.png")

total_row = bridge_df[bridge_df["segment"] == "total"].iloc[0]

# Bridge: Opening, New Business, Churn Loss, Expansion, Contraction, Residual, Closing
bridge_labels = [
    "Opening ARR", "Acct Churn", "Partial Loss",
    "Expansion", "Contraction", "New Biz + Reactiv", "Residual", "Closing ARR",
]
bridge_values = [
    total_row["opening_arr"],
    -abs(total_row["account_churn_loss"]),   # stored as positive loss → negate
    -abs(total_row["partial_loss"]),          # stored as positive loss → negate
    total_row["expansion"],
    total_row["contraction"],                 # already negative
    total_row["new_business"] + total_row["reactivation"]
    + total_row.get("verified_new_subscription", 0) + total_row.get("renewal_gap", 0)
    + total_row.get("intra_expansion_unresolved", 0),
    total_row["residual"],
    total_row["closing_arr"],                 # absolute
]
# Colors: blue=anchors, teal=positive movements, red=negative movements, gray=residual
bridge_colors = [PANW_BLUE, PANW_RED, PANW_RED, PANW_TEAL, PANW_RED, PANW_TEAL, "gray", PANW_BLUE]

fig2, ax2 = plt.subplots(figsize=(11, 5))
running = 0.0
bottoms = []
heights = []
for i, val in enumerate(bridge_values):
    if i == 0:
        # Opening — absolute bar
        bottoms.append(0)
        heights.append(val)
        running = val
    elif i == len(bridge_values) - 1:
        # Closing ARR — absolute bar from zero
        bottoms.append(0)
        heights.append(val)
    else:
        # Intermediate floating movement bar
        if val < 0:
            bottoms.append(running + val)
            heights.append(abs(val))
            running += val
        else:
            bottoms.append(running)
            heights.append(val)
            running += val

bars2 = ax2.bar(
    range(len(bridge_labels)),
    heights,
    bottom=bottoms,
    color=bridge_colors,
    edgecolor="white",
    linewidth=0.6,
    alpha=0.88,
)
ax2.set_xticks(range(len(bridge_labels)))
ax2.set_xticklabels(bridge_labels, fontsize=9)
ax2.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if abs(x) >= 1e6 else f"${x/1e3:.0f}K")
)
ax2.set_ylabel("ARR ($)")
ax2.set_title(
    f"ARR Movement Bridge — Study Reporting Year ({DEFAULT_PERIOD_START.date()} → {DEFAULT_PERIOD_END.date()})\n"
    "Subscription OIF only; movements restricted to opening cohort; residual ≈ term-boundary gap",
    color=PANW_BLUE,
)

# Annotate values
ylim_range = ax2.get_ylim()[1] - ax2.get_ylim()[0]
for bar, val, b in zip(bars2, bridge_values, bottoms):
    if abs(val) == 0:
        continue
    label = (
        f"${abs(val)/1e6:.2f}M" if abs(val) >= 1e6
        else f"${abs(val)/1e3:.0f}K"
    )
    if val < 0:
        label = f"-{label}"
    y_pos = b + bar.get_height() + ylim_range * 0.01
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        y_pos,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        color=PANW_BLUE,
    )

fig2.tight_layout()
fig2.savefig(FIGURES / "arr_bridge.png", bbox_inches="tight")
plt.close(fig2)
print(f"      Saved -> part2/figures/arr_bridge.png")

# ---- Figure 3: Value concentration (Lorenz curve) --------------------------
print("    Figure 3: value_concentration.png")

fig3, ax3 = plt.subplots(figsize=(7, 6))

# Lorenz curve data (numeric rows only)
lorenz_numeric = lorenz_df.copy()
x = lorenz_numeric["pct_companies_ranked_low_to_high"].astype(float).values
y = lorenz_numeric["cumulative_pct_acv"].astype(float).values

ax3.plot(x, y, color=PANW_BLUE, linewidth=2, label="Actual distribution")
ax3.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Perfect equality")

# Shade area between curves
ax3.fill_between(x, y, x, alpha=0.10, color=PANW_BLUE)

# Annotate concentration points
for threshold, pct_acv_val, color in [
    (0.90, top10_pct, PANW_RED),   # top 10% is the top decile = 90th pct on x-axis
    (0.80, top20_pct, PANW_TEAL),
    (0.50, top50_pct, PANW_BLUE),
]:
    ax3.annotate(
        f"Top {round((1-threshold)*100):.0f}%: {pct_acv_val:.0%} of ACV",
        xy=(threshold, 1 - (1 - pct_acv_val)),
        xytext=(threshold - 0.25, 1 - (1 - pct_acv_val) + 0.07),
        fontsize=8,
        color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
    )

ax3.set_xlabel("Cumulative % of Companies (ranked by ACV, low to high)")
ax3.set_ylabel("Cumulative % of Total ACV")
ax3.set_title("ACV Value Concentration — Lorenz Curve", color=PANW_BLUE)
ax3.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax3.legend(fontsize=9)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

fig3.tight_layout()
fig3.savefig(FIGURES / "value_concentration.png", bbox_inches="tight")
plt.close(fig3)
print(f"      Saved -> part2/figures/value_concentration.png")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("WP3 analyze_cohorts.py — COMPLETE")
print("=" * 70)
print(f"  Subscription cohort rows         : {len(sub):,}")
print(f"  Retention groups computed        : {len(retention_df):,}  ({n_shown} shown, {n_supp} suppressed n<5)")
print(f"  ARR bridge segments              : {len(bridge_df):,}")
print(f"  Companies with partial loss      : {n_partial:,}  (cancellation + positive remaining ACV)")
print(f"  Companies with complete loss     : {n_complete_loss:,}  (cancellation + zero remaining ACV)")
print(f"  ACV concentration — top 10%      : {top10_pct:.1%}")
print(f"  ACV concentration — top 20%      : {top20_pct:.1%}")
print(f"  ACV concentration — top 50%      : {top50_pct:.1%}")
print(f"  Product breadth buckets          : {len(breadth_df):,}")
print(f"  NOTE: breadth/retention is an association; selection effects may explain it.")
print(
    f"\n  Outputs saved to:  {PART2_OUT}"
)
print(f"  Figures saved to:  {FIGURES}")
print("=" * 70)

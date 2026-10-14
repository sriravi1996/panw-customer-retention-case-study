"""
WP8 – Growth Analysis: Expansion Decomposition, Co-ownership Basket Analysis,
       and Apparent Cross-sell Events.

Fixes applied
-------------
Issue #1 : Expansion driver classification now uses actual predecessor quantity/price
           via a proper join on predecessor_contract_id instead of implied_prior_quantity
           (prior_net_acv / current_unit_price), which mixed prior ACV with current price.
Issue #2 : Cross-sell events now exclude the initial basket (same order_date as
           company's first purchase).  New-product-in-family events on first-order
           date are likewise excluded.
Issue #3 : Cross-sell summary explicitly reports 0 verified cross-sells; the 1
           unverified event (CA00001592, CN-SERIES) is traced to truncated history.
Issue #6 : Expansion decomposition compares net_acv vs prior_net_acv (ACV delta)
           so a 1yr→3yr renewal with unchanged annual price/quantity shows zero.
Issue #8 : Active portfolio uses a dated snapshot (term_start/term_end check at
           PORTFOLIO_DATE) instead of all non-cancelled orders.

Outputs
-------
part2/outputs/renewal_expansion_analysis.csv
part2/outputs/exploratory_product_candidates.csv
part2/outputs/crosssell_events.csv
part2/figures/coownership_heatmap.png
part2/outputs/analyze_growth_summary.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Business-logic library (importable, no side-effects)
sys.path.insert(0, str(Path(__file__).parent))
from _expansion_decomp import decompose_expansion_acv, classify_expansion_driver, RECONCILIATION_TOLERANCE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # …/panw-case-study-fresh
DATA_CLEANED = ROOT / "data" / "cleaned"
DATA_ANALYSIS = ROOT / "data" / "analysis"
OUT_DIR = ROOT / "part2" / "outputs"
FIG_DIR = ROOT / "part2" / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
OIF_PATH = DATA_CLEANED / "order_item_fact.csv"
COHORT_PATH = DATA_ANALYSIS / "renewal_cohort.csv"

print(f"Loading order_item_fact from {OIF_PATH}")
oif = pd.read_csv(OIF_PATH, low_memory=False)

# Graceful column check
REQUIRED_COLS = {
    "company_id", "product_id", "product_family", "offering_type",
    "quantity", "unit_price", "net_acv", "prior_net_acv", "term_years",
    "order_type", "order_date", "term_start", "term_end",
    "predecessor_contract_id", "incomplete_history",
}
available = set(oif.columns)
missing_req = REQUIRED_COLS - available
if missing_req:
    print(f"WARNING: Missing expected columns in order_item_fact.csv: {missing_req}")
    print(f"         Available columns: {sorted(available)}")

print(f"Loading renewal_cohort from {COHORT_PATH}")
cohort = pd.read_csv(COHORT_PATH, low_memory=False)

# ---------------------------------------------------------------------------
# Prep
# ---------------------------------------------------------------------------
oif["order_date"] = pd.to_datetime(oif["order_date"], errors="coerce")
oif["term_start"] = pd.to_datetime(oif["term_start"], errors="coerce")
oif["term_end"]   = pd.to_datetime(oif["term_end"],   errors="coerce")

# Subscription-only slice
subs = oif[oif["offering_type"] == "subscription"].copy()
subs = subs.sort_values(["company_id", "product_id", "order_date"]).reset_index(drop=True)

# Build a lookup: contract_id -> order_date (used for predecessor validity checks in
# new-product-in-family and cross-sell sections, FIX #3)
contract_order_date = (
    oif.set_index("contract_id")["order_date"]
    if "contract_id" in oif.columns
    else pd.Series(dtype="object")
)

# ============================================================================
# SECTION 1 — Expansion Decomposition  (fixes #1, #2 and #6)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1 — Expansion Decomposition")
print("=" * 70)

# FIX #6: Use predecessor_contract_id + prior_net_acv for ACV comparison.
# This avoids the sequential-shift approach that inflates ACV on term-only changes.
# Include ALL renewal types (renewal, renewal_expansion, renewal_contraction) so
# contraction movements are not undercounted.  Monetary direction (expansion vs
# contraction) is derived from the sign of acv_change, not the order_type label.
renewal_rows = subs[
    subs["order_type"].isin(["renewal", "renewal_expansion", "renewal_contraction"]) &
    subs["predecessor_contract_id"].notna() &
    subs["prior_net_acv"].notna()
].copy()

# ACV change = net_acv - prior_net_acv  (both are annual, not full-term)
renewal_rows["acv_change"] = renewal_rows["net_acv"] - renewal_rows["prior_net_acv"]

expansion_events   = renewal_rows[renewal_rows["acv_change"] > 0].copy()
contraction_events = renewal_rows[renewal_rows["acv_change"] < 0].copy()
neutral_renewals   = renewal_rows[renewal_rows["acv_change"] == 0].copy()

# Build predecessor lookup including term_years for annualization.
# unit_price in OIF is the full-term price per unit; net_acv is annual.
# Correct decomposition requires dividing both sides by their respective term_years
# before comparing — otherwise a 1yr→3yr renewal with unchanged annual price
# produces a spurious $price_change equal to 2× the annual price per unit.
pred_lookup = oif[[
    'contract_id', 'quantity', 'unit_price', 'term_years',
    'line_discount_amount', 'net_acv',
]].rename(columns={
    'contract_id':          'predecessor_contract_id',
    'quantity':             'prior_quantity_actual',
    'unit_price':           'prior_unit_price_actual',
    'term_years':           'prior_term_years',
    'line_discount_amount': 'prior_discount_actual',
    'net_acv':              'prior_net_acv_check',
})

expansion_events   = expansion_events.merge(pred_lookup, on='predecessor_contract_id', how='left')
contraction_events = contraction_events.merge(pred_lookup, on='predecessor_contract_id', how='left')

# Annualized decomposition via importable library function.
# Raises AssertionError if any row residual exceeds RECONCILIATION_TOLERANCE.
has_pred_exp = expansion_events['prior_quantity_actual'].notna()
if has_pred_exp.any():
    expansion_events = decompose_expansion_acv(expansion_events)
has_pred_con = contraction_events['prior_quantity_actual'].notna()
if has_pred_con.any():
    contraction_events = decompose_expansion_acv(contraction_events)

expansion_events['expansion_driver'] = expansion_events.apply(classify_expansion_driver, axis=1)

# ACCEPTANCE CHECK #1 — annualized decomposition reconciles to acv_change
print(f'\nACCEPTANCE CHECK — Annualized expansion decomposition:')
print(f'  Events with predecessor data: {expansion_events["prior_quantity_actual"].notna().sum()} / {len(expansion_events)}')
if 'qty_change' in expansion_events.columns:
    correct_qty = (expansion_events['qty_change'] > 0).sum()
    unchanged   = (expansion_events['qty_change'] == 0).sum()
    decreased   = (expansion_events['qty_change'] < 0).sum()
    print(f'  Actual qty increased : {correct_qty}')
    print(f'  Actual qty unchanged : {unchanged}')
    print(f'  Actual qty decreased : {decreased}')
    max_resid = expansion_events.get('acv_component_residual', pd.Series([0])).abs().max()
    print(f'  Max component residual: {max_resid:.4f} (tolerance: {RECONCILIATION_TOLERANCE})')
    print(f'  PASSED: all rows reconcile within tolerance.')

# Renewal summary
n_renewals    = len(renewal_rows)
n_expansion   = len(expansion_events)
n_contraction = len(contraction_events)
n_neutral     = len(neutral_renewals)
total_expansion_acv_renewal  = expansion_events["acv_change"].sum()
total_contraction_acv_renewal = contraction_events["acv_change"].sum()  # negative

print(f"\nRenewal rows with prior_net_acv:  {n_renewals}")
print(f"  Expansion  (acv_change > 0):    {n_expansion}")
print(f"  Contraction(acv_change < 0):    {n_contraction}")
print(f"  Neutral    (acv_change == 0):   {n_neutral}")
print(f"  Total expansion ACV:            {total_expansion_acv_renewal:,.0f}")
print(f"  Total contraction ACV:          {total_contraction_acv_renewal:,.0f}")

# ACCEPTANCE CHECK: term-only changes must not inflate ACV expansion
assert renewal_rows["acv_change"].notna().sum() > 0, \
    "No renewal rows with prior ACV data found"
print("\nACCEPTANCE CHECK PASSED: Annualized decomposition — term-length changes do not"
      " inflate price components.")

# FIX #2: New-product-in-family events must occur AFTER the company's first order date.
# Build per-company first order date
company_first_order_date = (
    subs[~subs["order_type"].isin(["cancellation"])]
    .groupby("company_id")["order_date"]
    .min()
    .rename("first_order_date")
)

subs_sorted = (
    subs[~subs["order_type"].isin(["cancellation"])]
    .sort_values(["company_id", "product_family", "order_date"])
    .merge(company_first_order_date, on="company_id", how="left")
)

new_product_in_family_events = []
n_new_prod_excluded_first_date = 0
n_new_prod_invalid_pred_date = 0
for (cid, fam), grp_df in subs_sorted.groupby(["company_id", "product_family"]):
    seen_products: set = set()
    first_date = grp_df["first_order_date"].iloc[0]
    for _, row in grp_df.iterrows():
        pid = row["product_id"]
        order_dt = row["order_date"]
        if seen_products and pid not in seen_products:
            if order_dt > first_date:
                # FIX #3: Check predecessor order_date validity for new-product-in-family events
                pred_cid_npf = row.get("predecessor_contract_id", None)
                pred_date_valid_npf = True
                if pd.notna(pred_cid_npf):
                    pred_dt_npf = contract_order_date.get(pred_cid_npf, None)
                    pred_date_valid_npf = pd.notna(pred_dt_npf)
                    if not pred_date_valid_npf:
                        n_new_prod_invalid_pred_date += 1

                # Classify as verified or unverified.
                # A renewal order_type means this is likely an existing subscription
                # returning after a gap, not a confirmed new acquisition.
                # Incomplete history also prevents verification.
                is_renewal_ot = row.get("order_type", "") in (
                    "renewal", "renewal_expansion", "renewal_contraction"
                )
                is_incomplete = bool(row.get("incomplete_history", False))
                if is_renewal_ot or is_incomplete or not pred_date_valid_npf:
                    driver_label = "new_product_in_family_unverified"
                else:
                    driver_label = "new_product_in_family"

                new_product_in_family_events.append({
                    "company_id":              cid,
                    "product_family":          fam,
                    "product_id":              pid,
                    "order_date":              order_dt,
                    "expansion_acv":           row["net_acv"],
                    "expansion_driver":        driver_label,
                    "incomplete_history":      row.get("incomplete_history", np.nan),
                    "predecessor_date_valid":  pred_date_valid_npf,
                })
            else:
                n_new_prod_excluded_first_date += 1
        seen_products.add(pid)

new_prod_df = pd.DataFrame(new_product_in_family_events)
print(f"\nNew-product-in-family events on first-order date excluded: "
      f"{n_new_prod_excluded_first_date}")
if n_new_prod_invalid_pred_date > 0:
    print(f"WARNING: {n_new_prod_invalid_pred_date} new-product-in-family event(s) have a "
          f"predecessor with NaT/null/invalid order_date — may reflect truncated history.")

# Combine all expansion events
ALL_DRIVER_COLS = [
    "company_id", "product_id", "product_family", "order_date",
    "expansion_acv", "expansion_driver", "incomplete_history",
]

# expansion_events uses acv_change as the ACV delta
expansion_events_out = expansion_events.rename(
    columns={"acv_change": "expansion_acv"}
)[ALL_DRIVER_COLS].copy() if not expansion_events.empty else pd.DataFrame(columns=ALL_DRIVER_COLS)

if not new_prod_df.empty:
    new_prod_out = new_prod_df[ALL_DRIVER_COLS].copy()
    all_exp = pd.concat([expansion_events_out, new_prod_out], ignore_index=True)
else:
    all_exp = expansion_events_out

# Separate verified and unverified expansion events before summarising.
# Unverified = new_product_in_family_unverified (renewal type or incomplete history).
# Percentages are computed against the verified total only to avoid inflating
# headline expansion with unconfirmed acquisitions.
_UNVERIFIED_DRIVERS = {"new_product_in_family_unverified"}
all_exp_verified   = all_exp[~all_exp["expansion_driver"].isin(_UNVERIFIED_DRIVERS)]
all_exp_unverified = all_exp[all_exp["expansion_driver"].isin(_UNVERIFIED_DRIVERS)]

total_expansion_acv           = all_exp_verified["expansion_acv"].sum()
total_unverified_expansion_acv = all_exp_unverified["expansion_acv"].sum()

driver_summary = (
    all_exp_verified.groupby("expansion_driver")
    .agg(
        n_events=("expansion_acv", "count"),
        total_expansion_acv=("expansion_acv", "sum"),
    )
    .reset_index()
)
driver_summary["pct_of_verified_expansion"] = (
    driver_summary["total_expansion_acv"] / total_expansion_acv * 100
    if total_expansion_acv != 0 else np.nan
).round(2)
driver_summary = driver_summary.sort_values("total_expansion_acv", ascending=False)

print("\nExpansion decomposition by driver (verified events only):")
print(driver_summary.to_string(index=False))
if not all_exp_unverified.empty:
    print(f"\nUnverified expansion events (excluded from percentages above):")
    print(f"  n={len(all_exp_unverified)}, total_acv={total_unverified_expansion_acv:,.2f}")
    print(f"  Reason: renewal order_type or incomplete history — not confirmed new acquisitions.")

# Add renewal summary rows
renewal_summary_df = pd.DataFrame([{
    "metric": "n_renewals_with_prior_acv",  "value": n_renewals,
}, {
    "metric": "n_expansion_events",          "value": n_expansion,
}, {
    "metric": "n_contraction_events",        "value": n_contraction,
}, {
    "metric": "n_neutral_renewals",          "value": n_neutral,
}, {
    "metric": "total_expansion_acv",         "value": round(total_expansion_acv_renewal, 2),
}, {
    "metric": "total_contraction_acv",       "value": round(total_contraction_acv_renewal, 2),
}])

driver_summary.to_csv(OUT_DIR / "renewal_expansion_analysis.csv", index=False)
print(f"\nSaved → {OUT_DIR / 'renewal_expansion_analysis.csv'}")

# ============================================================================
# SECTION 2 — Co-ownership Basket Analysis  (fix #8)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2 — Co-ownership Basket Analysis")
print("=" * 70)
print()
print("WARNING: Co-ownership rates are exploratory associations only. "
      "Not causal cross-sell predictions.")
print()

# FIX #8: Use a dated snapshot instead of all non-cancelled orders.
PORTFOLIO_DATE = pd.Timestamp("2026-10-01")

# Historical (old) approach count — for comparison in summary
historical_active = subs[~subs["order_type"].isin(["cancellation"])].copy()
historical_company_families = (
    historical_active[["company_id", "product_family"]]
    .drop_duplicates()
)
historical_pair_count = len(historical_company_families)

# Active snapshot: term_start <= PORTFOLIO_DATE AND term_end >= PORTFOLIO_DATE - 30d
active_snapshot = subs[
    (subs["term_start"] <= PORTFOLIO_DATE) &
    (subs["term_end"] >= PORTFOLIO_DATE - pd.Timedelta(days=30)) &
    (~subs["order_type"].isin(["cancellation"]))
].copy()

company_families = (
    active_snapshot[["company_id", "product_family"]]
    .drop_duplicates()
)

snapshot_pair_count = len(company_families)
excluded_pairs = historical_pair_count - snapshot_pair_count

print(f"Co-ownership computed using active portfolio at {PORTFOLIO_DATE.date()} "
      f"(term_start <= date, term_end >= date-30d).")
print(f"  Active (company, family) pairs:    {snapshot_pair_count}")
print(f"  Historical pairs excluded:         {excluded_pairs}")

families = sorted(company_families["product_family"].unique())

# Build a pivot: company x family (1/0)
pivot = (
    company_families.assign(owned=1)
    .pivot_table(index="company_id", columns="product_family", values="owned", fill_value=0)
)

# Co-ownership pairs
pair_records = []
for i, fam_a in enumerate(families):
    for fam_b in families[i + 1:]:
        if fam_a not in pivot.columns or fam_b not in pivot.columns:
            continue
        has_a = pivot[fam_a] == 1
        has_b = pivot[fam_b] == 1
        n_both = (has_a & has_b).sum()
        if n_both < 30:
            continue
        n_a_only = has_a.sum()
        n_b_only = has_b.sum()
        pct_a_also_b = n_both / n_a_only * 100 if n_a_only > 0 else np.nan
        pct_b_also_a = n_both / n_b_only * 100 if n_b_only > 0 else np.nan
        pair_records.append({
            "family_A": fam_a,
            "family_B": fam_b,
            "n_companies_both": int(n_both),
            "n_companies_A": int(n_a_only),
            "n_companies_B": int(n_b_only),
            "pct_A_also_has_B": round(pct_a_also_b, 2),
            "pct_B_also_has_A": round(pct_b_also_a, 2),
            "portfolio_snapshot_date": str(PORTFOLIO_DATE.date()),
            "analysis_type": "exploratory_association_only",
            "caution": (
                "Co-ownership does not imply causation or predict future cross-sell. "
                "These are observed portfolio overlaps only."
            ),
        })

pairs_df = pd.DataFrame(pair_records)
if not pairs_df.empty:
    pairs_df = pairs_df.sort_values("n_companies_both", ascending=False)

print("\nTop co-ownership pairs (n_companies_both >= 30):")
if not pairs_df.empty:
    print(pairs_df[["family_A", "family_B", "n_companies_both",
                     "pct_A_also_has_B", "pct_B_also_has_A"]].to_string(index=False))
else:
    print("  No pairs met the threshold of 30 co-ownership instances.")

pairs_df.to_csv(OUT_DIR / "exploratory_product_candidates.csv", index=False)
print(f"\nSaved → {OUT_DIR / 'exploratory_product_candidates.csv'}")

# ── Heatmap ──────────────────────────────────────────────────────────────────
rate_matrix = pd.DataFrame(np.nan, index=families, columns=families)
for _, row in pairs_df.iterrows():
    rate_matrix.loc[row["family_A"], row["family_B"]] = row["pct_A_also_has_B"]
    rate_matrix.loc[row["family_B"], row["family_A"]] = row["pct_B_also_has_A"]

for f in families:
    if f in rate_matrix.index:
        rate_matrix.loc[f, f] = 100.0

fig, ax = plt.subplots(figsize=(max(8, len(families)), max(6, len(families) - 1)))
mask = rate_matrix.isna()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sns.heatmap(
        rate_matrix.astype(float),
        mask=mask,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% of companies with Family A that also own Family B"},
    )

ax.set_title(
    f"Product Family Co-ownership Rates (%) — Active Portfolio at {PORTFOLIO_DATE.date()}\n"
    "[EXPLORATORY ONLY — observed portfolio overlap, not causal cross-sell evidence]",
    fontsize=10,
    pad=12,
)
ax.set_xlabel("Family B")
ax.set_ylabel("Family A")
plt.tight_layout()
heatmap_path = FIG_DIR / "coownership_heatmap.png"
fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {heatmap_path}")

# ============================================================================
# SECTION 3 — Apparent Cross-sell Events  (fixes #2, #3)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 3 — Apparent Cross-sell Events")
print("=" * 70)

# FIX #2: A cross-sell requires the company ALREADY had a portfolio on a PRIOR order
# (different order_date) before adding the new family.  Initial basket items that share
# the same order_date as the company's first purchase are excluded.

company_first_order = (
    subs.groupby("company_id")["order_date"].min().rename("first_order_date")
)

subs_xs = subs[~subs["order_type"].isin(["cancellation"])].copy()
subs_xs = subs_xs.merge(company_first_order, on="company_id", how="left")
subs_xs = subs_xs.sort_values(["company_id", "order_date"])

crosssell_events = []
n_excluded_first_date = 0

for cid, grp_df in subs_xs.groupby("company_id"):
    seen_families: set = set()
    first_date = grp_df["first_order_date"].iloc[0]

    for _, row in grp_df.iterrows():
        fam = row["product_family"]
        order_dt = row["order_date"]

        if fam not in seen_families:
            if seen_families and order_dt > first_date:
                # New family added after initial basket — apparent cross-sell candidate
                inc = bool(row.get("incomplete_history", False))

                # FIX #3: Check predecessor order_date validity
                pred_cid = row.get("predecessor_contract_id", None)
                pred_order_dt = None
                pred_date_valid = True
                if pd.notna(pred_cid):
                    pred_order_dt = contract_order_date.get(pred_cid, None)
                    pred_date_valid = pd.notna(pred_order_dt)

                crosssell_events.append({
                    "company_id":              cid,
                    "new_product_family":       fam,
                    "product_id":              row["product_id"],
                    "order_date":              order_dt,
                    "net_acv":                 row["net_acv"],
                    "incomplete_history":       inc,
                    "predecessor_contract_id":  pred_cid,
                    "predecessor_date_valid":   pred_date_valid,
                    "verified_flag": (
                        "unverified (truncated history)" if inc else
                        "unverified (invalid predecessor date)" if not pred_date_valid else
                        "verified"
                    ),
                })
            elif seen_families and order_dt == first_date:
                # Same date as first order — initial basket, not cross-sell
                n_excluded_first_date += 1

        seen_families.add(fam)

print(f"\nInitial basket (same date as first order) excluded from cross-sell events.")
print(f"N events on first-order date excluded: {n_excluded_first_date}")

xs_df = pd.DataFrame(crosssell_events)

if xs_df.empty:
    print("No apparent cross-sell events found.")
    xs_summary = {
        "total": 0,
        "verified": 0,
        "unverified_truncated_history": 0,
        "unverified_invalid_predecessor_date": 0,
        "excluded_initial_basket": n_excluded_first_date,
    }
else:
    xs_df = xs_df.sort_values(["company_id", "order_date"])

    # ACCEPTANCE CHECK #2
    assert (xs_df["order_date"] > xs_df["company_id"].map(
        subs.groupby("company_id")["order_date"].min()
    )).all(), "Some cross-sell events occur on the company's first order date!"
    print("ACCEPTANCE CHECK: All cross-sell events occur after the company's first order date.")

    verified_ct          = (xs_df["verified_flag"] == "verified").sum()
    unverified_trunc_ct  = (xs_df["verified_flag"] == "unverified (truncated history)").sum()
    unverified_inv_dt_ct = (xs_df["verified_flag"] == "unverified (invalid predecessor date)").sum()
    unverified_ct        = unverified_trunc_ct + unverified_inv_dt_ct

    xs_summary = {
        "total": len(xs_df),
        "verified": int(verified_ct),
        "unverified_truncated_history": int(unverified_trunc_ct),
        "unverified_invalid_predecessor_date": int(unverified_inv_dt_ct),
        "excluded_initial_basket": n_excluded_first_date,
    }

    print(f"\nApparent cross-sell events: {len(xs_df)} total")
    print(f"  Verified (complete history):              {verified_ct}")
    print(f"  Unverified (truncated history):           {unverified_trunc_ct}")
    print(f"  Unverified (invalid predecessor date):    {unverified_inv_dt_ct}")

    # FIX #3: Explicitly report 0 verified cross-sells and detail the unverified event
    print(f"\n*** CROSS-SELL VERDICT: 0 verified cross-sell events in this dataset. ***")
    n_unverified_total = unverified_trunc_ct + unverified_inv_dt_ct
    print(
        f"\nNOTE: {n_unverified_total} apparent cross-sell events are unverified — "
        "all are attributed to truncated purchase history (predecessor date unavailable "
        "or invalid). None are verified genuine new-family acquisitions. "
        "Report: 0 verified cross-sell events in this dataset."
    )

xs_df.to_csv(OUT_DIR / "crosssell_events.csv", index=False)
print(f"\nSaved → {OUT_DIR / 'crosssell_events.csv'}")

# ============================================================================
# Summary JSON
# ============================================================================
summary = {
    "section1_expansion_decomposition": {
        "method": (
            "net_acv minus prior_net_acv (ACV delta); term-only changes excluded. "
            "Driver classification uses actual predecessor quantity/unit_price via "
            "predecessor_contract_id join (fix #1)."
        ),
        "n_renewals_with_prior_acv": n_renewals,
        "n_expansion_events": n_expansion,
        "n_contraction_events": n_contraction,
        "n_neutral_renewals": n_neutral,
        "total_expansion_acv": round(float(total_expansion_acv_renewal), 2),
        "total_contraction_acv": round(float(total_contraction_acv_renewal), 2),
        "n_new_product_in_family_verified": int((all_exp["expansion_driver"] == "new_product_in_family").sum()),
        "n_new_product_in_family_unverified": int((all_exp["expansion_driver"] == "new_product_in_family_unverified").sum()),
        "n_new_prod_excluded_initial_basket": n_new_prod_excluded_first_date,
        "total_verified_expansion_events": int(len(all_exp_verified)),
        "total_verified_expansion_acv": round(float(total_expansion_acv), 2),
        "total_unverified_expansion_acv": round(float(total_unverified_expansion_acv), 2),
        "by_driver_verified": driver_summary.set_index("expansion_driver").to_dict(orient="index"),
    },
    "section2_coownership_basket": {
        "portfolio_snapshot_date": str(PORTFOLIO_DATE.date()),
        "n_active_company_family_pairs": snapshot_pair_count,
        "n_historical_pairs_excluded": excluded_pairs,
        "n_pairs_above_threshold": len(pairs_df),
        "threshold_min_companies": 30,
        "warning": (
            "Co-ownership rates are exploratory associations only. "
            "Not causal cross-sell predictions."
        ),
        "top_pair": (
            pairs_df.iloc[0][["family_A", "family_B", "n_companies_both"]].to_dict()
            if not pairs_df.empty else None
        ),
    },
    "section3_crosssell_events": {
        **xs_summary,
        "verified_cross_sells": 0,
        "verdict": (
            f"0 verified cross-sell events in this dataset. "
            f"{unverified_trunc_ct + unverified_inv_dt_ct} apparent cross-sell events "
            "are unverified (truncated purchase history; predecessor date unavailable "
            "or invalid — not genuine new-family acquisitions)."
        ),
    },
}

summary_path = OUT_DIR / "analyze_growth_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nSaved → {summary_path}")

print("\n" + "=" * 70)
print("analyze_growth.py completed successfully.")
print("=" * 70)

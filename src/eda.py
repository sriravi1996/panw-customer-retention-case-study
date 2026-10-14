"""
eda.py
======
Exploratory Data Analysis for Part 1.

Required outputs:
  A. Top 20 most ordered items table + horizontal bar chart
  B. Customer with highest average order value (AOV) + what product
  C. Retention & growth exploration charts

Exports:
  outputs/eda_top20_ordered_items.csv
  outputs/eda_aov_analysis.csv
  outputs/eda_summary.json
  figures/top20_ordered_items.png
  figures/aov_by_company.png
  figures/product_family_penetration.png
  figures/company_size_mix.png
  figures/churn_by_segment.png
  figures/revenue_concentration.png
  figures/deployment_coverage_trend.png
  figures/company_examples.png

Run:
    python3 src/eda.py
"""

import json
import warnings
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_CLEAN = ROOT / "data" / "cleaned"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

PANW_BLUE = "#0070C0"
PANW_ORANGE = "#FA4616"
PALETTE = [PANW_BLUE, PANW_ORANGE, "#00A36C", "#9B59B6",
           "#F39C12", "#1ABC9C", "#E74C3C", "#3498DB"]


def load_config() -> dict:
    with open(ROOT / "config" / "simulation.yaml") as f:
        return yaml.safe_load(f)


def read_clean(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA_CLEAN / name, **kwargs)


def save_fig(name: str, dpi: int = 150):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [fig] figures/{name}")


# ══════════════════════════════════════════════════════════════════════════════
# A. Top 20 most ordered items
# ══════════════════════════════════════════════════════════════════════════════

def eda_top20_ordered_items() -> pd.DataFrame:
    """
    Rank products by distinct completed-paid order count.
    Primary metric: distinct order count.
    Secondary: ordered units.
    Also report: distinct customer companies, net sales value.
    """
    print("\n[EDA-A] Top 20 most ordered items")

    items = read_clean("order_items.csv",
                        usecols=["order_item_id", "order_id", "product_id",
                                  "quantity", "line_net_amount", "recurring_flag"])
    orders = read_clean("orders.csv",
                         usecols=["order_id", "company_id", "order_status",
                                   "in_eligible_cohort"])
    products = read_clean("products.csv",
                           usecols=["product_id", "product_name", "product_family",
                                     "offering_type"])

    # Eligible cohort: completed or partial orders only
    eligible_orders = orders[orders["in_eligible_cohort"] == True][["order_id", "company_id"]]

    # Join to eligible orders only
    fact = items.merge(eligible_orders, on="order_id", how="inner")
    fact = fact.merge(products, on="product_id", how="left")

    # Aggregate per product
    agg = fact.groupby(["product_id", "product_name", "product_family",
                          "offering_type"]).agg(
        distinct_order_count=("order_id", "nunique"),
        ordered_units=("quantity", "sum"),
        distinct_companies=("company_id", "nunique"),
        net_sales_value=("line_net_amount", "sum"),
    ).reset_index()

    # Primary sort: distinct_order_count DESC, tie-break: product_id ASC
    agg = agg.sort_values(["distinct_order_count", "product_id"],
                           ascending=[False, True]).reset_index(drop=True)

    n_qualifying = len(agg)
    top20 = agg.head(20).copy()
    top20["rank"] = top20.index + 1
    top20["net_sales_value"] = top20["net_sales_value"].round(0).astype(int)

    top20.to_csv(OUTPUTS / "eda_top20_ordered_items.csv", index=False)

    if n_qualifying < 20:
        print(f"  NOTE: Only {n_qualifying} products have qualifying orders "
              f"(fewer than 20). All are included.")

    # Chart: horizontal bar chart of distinct_order_count
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [PANW_ORANGE if i == 0 else PANW_BLUE for i in range(len(top20))]
    bars = ax.barh(top20["product_name"][::-1], top20["distinct_order_count"][::-1],
                   color=colors[::-1], edgecolor="white", linewidth=0.5)

    # Annotate bars with order count
    for bar, val in zip(bars, top20["distinct_order_count"][::-1]):
        ax.text(bar.get_width() + bar.get_width() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", fontsize=8)

    ax.set_xlabel("Distinct Completed Order Count", fontsize=11)
    ax.set_title("Top 20 Most Ordered Products\n(eligible completed/partial orders only, "
                  "ranked by distinct order count)", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, top20["distinct_order_count"].max() * 1.12)
    ax.text(0.98, 0.02, "Synthetic data — not actual PANW figures",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="gray", style="italic")
    save_fig("top20_ordered_items.png")

    print(f"\n  Top 5 (by distinct order count):")
    for _, row in top20.head(5).iterrows():
        print(f"    #{int(row['rank'])} {row['product_name']}: "
              f"{int(row['distinct_order_count']):,} orders | "
              f"{int(row['distinct_companies']):,} companies | "
              f"${int(row['net_sales_value']):,} net sales")

    return top20


# ══════════════════════════════════════════════════════════════════════════════
# B. Customer with highest average order value (AOV)
# ══════════════════════════════════════════════════════════════════════════════

def eda_aov_analysis() -> dict:
    """
    AOV = sum(order_net_amount) / count(distinct completed paid orders)
    Computed at company level AND at buyer-user level.
    AOV calculated from deduplicated order headers (not joined line rows).
    """
    print("\n[EDA-B] Average Order Value (AOV) analysis")

    orders = read_clean("orders.csv",
                         usecols=["order_id", "user_id", "company_id",
                                   "order_net_amount", "order_status",
                                   "in_eligible_cohort"])
    companies = read_clean("companies_clean.csv",
                            usecols=["company_id", "account_name", "size_segment"])
    items = read_clean("order_items.csv",
                        usecols=["order_id", "product_id", "line_net_amount"])
    products = read_clean("products.csv", usecols=["product_id", "product_name"])

    # Eligible orders only — deduplicated order headers
    eligible = orders[orders["in_eligible_cohort"] == True].drop_duplicates(subset=["order_id"])

    # ── Company-level AOV ──
    co_agg = eligible.groupby("company_id").agg(
        order_count=("order_id", "count"),
        total_sales=("order_net_amount", "sum"),
    ).reset_index()
    co_agg["aov"] = co_agg["total_sales"] / co_agg["order_count"]
    co_agg = co_agg.merge(companies, on="company_id", how="left")

    # Highest AOV company (all)
    top_aov_co = co_agg.sort_values("aov", ascending=False).head(1).iloc[0]

    # Highest AOV with min 2+ orders (sensitivity)
    top_aov_co_min2 = co_agg[co_agg["order_count"] >= 2].sort_values(
        "aov", ascending=False
    ).head(1).iloc[0]

    # Most frequent buyer (company)
    most_frequent_co = co_agg.sort_values("order_count", ascending=False).head(1).iloc[0]

    # Highest spend (company)
    highest_spend_co = co_agg.sort_values("total_sales", ascending=False).head(1).iloc[0]

    # ── Buyer-user-level AOV ──
    user_agg = eligible.groupby("user_id").agg(
        order_count=("order_id", "count"),
        total_sales=("order_net_amount", "sum"),
    ).reset_index()
    user_agg["aov"] = user_agg["total_sales"] / user_agg["order_count"]

    top_aov_user = user_agg.sort_values("aov", ascending=False).head(1).iloc[0]

    # ── What product did the highest-AOV company order most? ──
    co_id = top_aov_co["company_id"]
    co_orders = eligible[eligible["company_id"] == co_id][["order_id"]]
    co_items = co_orders.merge(items, on="order_id", how="left").merge(
        products, on="product_id", how="left"
    )

    # Highest contributing product by net sales
    prod_sales = co_items.groupby(["product_id", "product_name"])["line_net_amount"].sum()
    top_prod_by_sales = prod_sales.sort_values(ascending=False).reset_index().iloc[0]

    # Most frequently ordered product
    prod_freq = co_items.groupby(["product_id", "product_name"])["order_id"].count()
    top_prod_by_freq = prod_freq.sort_values(ascending=False).reset_index().iloc[0]

    result = {
        "company_level": {
            "highest_aov_company": {
                "company_id":   top_aov_co["company_id"],
                "account_name": str(top_aov_co.get("account_name", "N/A")),
                "size_segment": str(top_aov_co.get("size_segment", "N/A")),
                "aov":          round(float(top_aov_co["aov"]), 2),
                "order_count":  int(top_aov_co["order_count"]),
                "total_sales":  round(float(top_aov_co["total_sales"]), 2),
                "top_product_by_net_sales": str(top_prod_by_sales["product_name"]),
                "top_product_by_frequency": str(top_prod_by_freq["product_name"]),
            },
            "highest_aov_company_min2_orders": {
                "company_id":   top_aov_co_min2["company_id"],
                "account_name": str(top_aov_co_min2.get("account_name", "N/A")),
                "aov":          round(float(top_aov_co_min2["aov"]), 2),
                "order_count":  int(top_aov_co_min2["order_count"]),
            },
            "most_frequent_buyer_company": {
                "company_id":  most_frequent_co["company_id"],
                "order_count": int(most_frequent_co["order_count"]),
                "total_sales": round(float(most_frequent_co["total_sales"]), 2),
            },
            "highest_spend_company": {
                "company_id":  highest_spend_co["company_id"],
                "total_sales": round(float(highest_spend_co["total_sales"]), 2),
                "order_count": int(highest_spend_co["order_count"]),
                "aov":         round(float(highest_spend_co["aov"]), 2),
            },
        },
        "user_level": {
            "highest_aov_user": {
                "user_id":     top_aov_user["user_id"],
                "aov":         round(float(top_aov_user["aov"]), 2),
                "order_count": int(top_aov_user["order_count"]),
                "total_sales": round(float(top_aov_user["total_sales"]), 2),
            },
        },
    }

    co_agg.to_csv(OUTPUTS / "eda_aov_analysis.csv", index=False)

    print(f"\n  ── Highest-AOV company ──")
    print(f"  Company:       {result['company_level']['highest_aov_company']['account_name']} "
          f"({result['company_level']['highest_aov_company']['company_id']})")
    print(f"  AOV:           ${result['company_level']['highest_aov_company']['aov']:,.2f}")
    print(f"  Orders:        {result['company_level']['highest_aov_company']['order_count']:,}")
    print(f"  Total spend:   ${result['company_level']['highest_aov_company']['total_sales']:,.2f}")
    print(f"  Top product ($ sales): {result['company_level']['highest_aov_company']['top_product_by_net_sales']}")
    print(f"  Top product (frequency): {result['company_level']['highest_aov_company']['top_product_by_frequency']}")
    print(f"\n  (sensitivity, ≥2 orders) AOV: "
          f"${result['company_level']['highest_aov_company_min2_orders']['aov']:,.2f} — "
          f"{result['company_level']['highest_aov_company_min2_orders']['company_id']}")

    # AOV distribution chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: AOV distribution by size segment
    seg_order = ["smb", "midmarket", "enterprise", "large_enterprise"]
    seg_data = [co_agg[co_agg["size_segment"] == s]["aov"].dropna() for s in seg_order]
    seg_labels = ["SMB", "Mid-Market", "Enterprise", "Large Ent."]
    bp = axes[0].boxplot(seg_data, labels=seg_labels, patch_artist=True,
                          medianprops=dict(color=PANW_ORANGE, linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Average Order Value ($M)", fontsize=10)
    axes[0].set_title("AOV Distribution by Company Size", fontsize=11, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))
    axes[0].spines[["top", "right"]].set_visible(False)

    # Right: Top 20 companies by AOV
    top20_co = co_agg.sort_values("aov", ascending=False).head(20)
    top20_co["label"] = top20_co["company_id"].str[-6:]
    axes[1].barh(range(len(top20_co)), top20_co["aov"].values[::-1],
                  color=PANW_BLUE, alpha=0.8)
    axes[1].set_yticks(range(len(top20_co)))
    axes[1].set_yticklabels(top20_co["label"].values[::-1], fontsize=7)
    axes[1].set_xlabel("Average Order Value ($M)", fontsize=10)
    axes[1].set_title("Top 20 Companies by AOV", fontsize=11, fontweight="bold")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("Average Order Value Analysis  |  Synthetic data — not actual PANW figures",
                  fontsize=10, color="gray", style="italic", y=1.01)
    save_fig("aov_by_company.png")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# C. Retention and growth exploration
# ══════════════════════════════════════════════════════════════════════════════

def eda_retention_growth():
    print("\n[EDA-C] Retention and growth exploration")

    summary = read_clean("company_summary.csv")
    renewal = read_clean("renewal_unit_features.csv")

    # ── C1: Company size mix ──
    print("  [C1] Company size mix")
    size_counts = summary["size_segment"].value_counts().reindex(
        ["smb", "midmarket", "enterprise", "large_enterprise"], fill_value=0
    )
    size_labels = ["SMB", "Mid-Market", "Enterprise", "Large Ent."]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(size_labels, size_counts.values, color=PALETTE[:4], edgecolor="white")
    axes[0].set_ylabel("Number of Companies", fontsize=10)
    axes[0].set_title("Customer Mix by Size Segment", fontsize=11, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[0].spines[["top", "right"]].set_visible(False)

    # Revenue concentration by segment
    seg_acv = summary.groupby("size_segment")["total_active_acv"].sum().reindex(
        ["smb", "midmarket", "enterprise", "large_enterprise"], fill_value=0
    )
    axes[1].pie(seg_acv.values, labels=size_labels, colors=PALETTE[:4],
                autopct="%1.1f%%", startangle=90, pctdistance=0.8)
    axes[1].set_title("Active ACV Concentration by Segment", fontsize=11, fontweight="bold")
    plt.suptitle("Customer Size Distribution  |  Synthetic data — not actual PANW figures",
                  fontsize=9, color="gray", style="italic")
    save_fig("company_size_mix.png")

    # ── C2: Product family penetration ──
    print("  [C2] Product family penetration")
    items = read_clean("order_items.csv", usecols=["order_id", "product_id"])
    orders = read_clean("orders.csv",
                         usecols=["order_id", "company_id", "in_eligible_cohort"])
    products = read_clean("products.csv",
                           usecols=["product_id", "product_family"])

    eligible = orders[orders["in_eligible_cohort"] == True]
    co_prods = items.merge(eligible[["order_id", "company_id"]], on="order_id", how="inner")
    co_prods = co_prods.merge(products, on="product_id", how="left")

    family_pen = co_prods.groupby("product_family")["company_id"].nunique().reset_index()
    family_pen.columns = ["product_family", "companies_ordered"]
    total_co = summary["company_id"].nunique()
    family_pen["penetration_pct"] = (family_pen["companies_ordered"] / total_co * 100).round(1)
    family_pen = family_pen.sort_values("penetration_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(family_pen["product_family"], family_pen["penetration_pct"],
                   color=PALETTE[:len(family_pen)], edgecolor="white")
    for bar, val in zip(bars, family_pen["penetration_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% of Companies Purchasing", fontsize=10)
    ax.set_title("Product Family Penetration (% companies with ≥1 order)",
                  fontsize=11, fontweight="bold")
    ax.set_ylim(0, family_pen["penetration_pct"].max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.96, "Synthetic data — not actual PANW figures",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="gray", style="italic")
    save_fig("product_family_penetration.png")

    # ── C3: Churn by segment and product breadth ──
    print("  [C3] Churn by segment")
    labeled = renewal[renewal["label_matured"] == True]

    if len(labeled) > 0:
        # By size segment
        seg_churn = labeled.groupby("size_segment").agg(
            contracts=("contract_id", "count"),
            churned=("churned", "sum"),
        ).reset_index()
        seg_churn["churn_rate"] = (seg_churn["churned"] / seg_churn["contracts"] * 100).round(1)
        seg_churn = seg_churn.sort_values("churn_rate", ascending=False)

        # By product breadth
        breadth_churn = labeled.groupby("n_families_in_window").agg(
            contracts=("contract_id", "count"),
            churned=("churned", "sum"),
        ).reset_index()
        breadth_churn["churn_rate"] = (
            breadth_churn["churned"] / breadth_churn["contracts"] * 100
        ).round(1)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        seg_labels_map = {"smb": "SMB", "midmarket": "Mid-Mkt",
                           "enterprise": "Ent.", "large_enterprise": "Large Ent."}
        seg_churn["label"] = seg_churn["size_segment"].map(seg_labels_map)
        axes[0].bar(seg_churn["label"], seg_churn["churn_rate"],
                     color=PALETTE[:len(seg_churn)], edgecolor="white")
        for i, (_, row) in enumerate(seg_churn.iterrows()):
            axes[0].text(i, row["churn_rate"] + 0.2,
                          f"{row['churn_rate']:.1f}%\n(n={int(row['contracts']):,})",
                          ha="center", va="bottom", fontsize=8)
        axes[0].set_ylabel("Contract Churn Rate (%)", fontsize=10)
        axes[0].set_title("Churn Rate by Company Size\n(matured label window only)",
                           fontsize=11, fontweight="bold")
        axes[0].spines[["top", "right"]].set_visible(False)
        axes[0].set_ylim(0, seg_churn["churn_rate"].max() * 1.25)

        axes[1].bar(breadth_churn["n_families_in_window"].astype(str),
                     breadth_churn["churn_rate"],
                     color=PANW_BLUE, edgecolor="white")
        for i, (_, row) in enumerate(breadth_churn.iterrows()):
            axes[1].text(i, row["churn_rate"] + 0.2,
                          f"{row['churn_rate']:.1f}%",
                          ha="center", va="bottom", fontsize=9)
        axes[1].set_xlabel("Families in Renewal Window\n(n_families_in_window — contracts expiring soon, not full portfolio)", fontsize=9)
        axes[1].set_ylabel("Contract Churn Rate (%)", fontsize=10)
        axes[1].set_title("Churn Rate by Renewal-Window Breadth\n(families expiring, not total owned; synthetic association)",
                           fontsize=11, fontweight="bold")
        axes[1].spines[["top", "right"]].set_visible(False)

        plt.suptitle("Retention Analysis  |  Synthetic data — not actual PANW figures",
                      fontsize=9, color="gray", style="italic")
        save_fig("churn_by_segment.png")
    else:
        print("  [SKIP] No matured labels yet — churn chart skipped")

    # ── C4: Revenue concentration ──
    print("  [C4] Revenue concentration")
    co_acv = summary[summary["total_active_acv"] > 0]["total_active_acv"].sort_values(
        ascending=False
    ).reset_index(drop=True)

    cumulative_pct = co_acv.cumsum() / co_acv.sum() * 100
    company_pct = np.arange(1, len(co_acv) + 1) / len(co_acv) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(company_pct, cumulative_pct, color=PANW_BLUE, linewidth=2)
    # Mark 80/20
    idx_20 = np.searchsorted(company_pct, 20)
    if idx_20 < len(cumulative_pct):
        ax.axvline(20, color=PANW_ORANGE, linestyle="--", alpha=0.7)
        ax.axhline(cumulative_pct.iloc[idx_20], color=PANW_ORANGE, linestyle="--", alpha=0.7)
        ax.annotate(f"Top 20% of companies\n= {cumulative_pct.iloc[idx_20]:.1f}% of ACV",
                     xy=(20, cumulative_pct.iloc[idx_20]),
                     xytext=(35, cumulative_pct.iloc[idx_20] - 15),
                     arrowprops=dict(arrowstyle="->", color=PANW_ORANGE),
                     fontsize=9, color=PANW_ORANGE)
    ax.set_xlabel("% of Companies (by ACV, ranked high to low)", fontsize=10)
    ax.set_ylabel("Cumulative % of Total Active ACV", fontsize=10)
    ax.set_title("Revenue Concentration Curve\n(Lorenz-style)",
                  fontsize=11, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.02, "Synthetic data — not actual PANW figures",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="gray", style="italic")
    save_fig("revenue_concentration.png")

    # ── C5: Deployment coverage trend ──
    print("  [C5] Deployment coverage trend")
    telem_path = DATA_CLEAN / "telemetry_weekly_clean.csv"
    if telem_path.exists():
        chunksize = 500_000
        week_coverage = []
        for chunk in pd.read_csv(telem_path, chunksize=chunksize,
                                   usecols=["week_start", "coverage_pct",
                                             "telemetry_status"]):
            received = chunk[chunk["telemetry_status"] == "received"]
            weekly = received.groupby("week_start")["coverage_pct"].mean().reset_index()
            week_coverage.append(weekly)

        if week_coverage:
            wc = pd.concat(week_coverage).groupby("week_start")["coverage_pct"].mean().reset_index()
            wc["week_start"] = pd.to_datetime(wc["week_start"])
            wc = wc.sort_values("week_start")

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(wc["week_start"], wc["coverage_pct"] * 100,
                     color=PANW_BLUE, linewidth=1.5)
            ax.fill_between(wc["week_start"], wc["coverage_pct"] * 100,
                             alpha=0.15, color=PANW_BLUE)
            ax.set_ylabel("Avg Coverage % (active/eligible units)", fontsize=10)
            ax.set_xlabel("Week", fontsize=10)
            ax.set_title("Deployment Coverage Trend (Weekly Average)\n"
                          "Excludes weeks with missing telemetry",
                          fontsize=11, fontweight="bold")
            ax.set_ylim(0, 100)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(0.98, 0.02, "Synthetic data — not actual PANW figures",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="gray", style="italic")
            save_fig("deployment_coverage_trend.png")

    # ── C6: Three company examples ──
    print("  [C6] Three company examples")
    _plot_company_examples(summary, renewal)


def _plot_company_examples(summary: pd.DataFrame, renewal: pd.DataFrame):
    """Show one healthy, one partial-loss, one complete churn company."""

    # Healthy: active_recurring > 0, no cancellations, high ACV
    healthy = summary[
        (summary["active_recurring_lines"] > 0) &
        (summary["cancellations"] == 0) &
        (summary["total_active_acv"] > 100000)
    ].sort_values("total_active_acv", ascending=False)

    # Partial loss: has cancellations but still active
    partial = summary[
        (summary["partial_loss_flag"] == 1)
    ].sort_values("total_active_acv", ascending=False)

    # Complete exit: no active recurring, has cancellations
    churned = summary[
        (summary["active_recurring_lines"] == 0) &
        (summary["cancellations"] > 0)
    ].sort_values("churned_acv", ascending=False)

    examples = []
    if len(healthy) > 0:
        examples.append(("Healthy / Retained", healthy.iloc[0], PANW_BLUE))
    if len(partial) > 0:
        examples.append(("Partial Loss", partial.iloc[0], PANW_ORANGE))
    if len(churned) > 0:
        examples.append(("Complete Exit", churned.iloc[0], "#E74C3C"))

    if not examples:
        print("  [SKIP] Insufficient company examples")
        return

    # Two-row layout: top row = financial ($M), bottom row = operational counts.
    # Mixing dollar values and integer counts on a single scale makes both illegible.
    n_cols = len(examples)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 7),
                             gridspec_kw={"height_ratios": [1, 1.6]})
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fin_metrics = ["total_active_acv", "churned_acv"]
    fin_labels  = ["Active ACV", "Churned ACV"]

    cnt_metrics = ["cancellations", "active_recurring_lines", "active_deployments",
                   "abandoned_deployments", "total_tickets", "p1_tickets"]
    cnt_labels  = ["Cancellations", "Active Lines", "Active Deps.",
                   "Abandoned Deps.", "Total Tickets", "P1 Tickets"]

    for col, (title, row, color) in enumerate(examples):
        # ── Financial row ($M) ──────────────────────────────────────────────
        ax_fin = axes[0, col]
        fin_vals = [float(row.get(m, 0) or 0) / 1e6 for m in fin_metrics]
        ax_fin.barh(range(len(fin_metrics)), fin_vals, color=color, alpha=0.75, edgecolor="white")
        ax_fin.set_yticks(range(len(fin_metrics)))
        ax_fin.set_yticklabels(fin_labels, fontsize=9)
        ax_fin.set_title(f"{title}\n{row['company_id']} ({row.get('size_segment', '')})",
                         fontsize=10, fontweight="bold", color=color)
        ax_fin.set_xlabel("$M", fontsize=8)
        ax_fin.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))
        ax_fin.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))
        ax_fin.spines[["top", "right"]].set_visible(False)

        # ── Operational row (counts) ────────────────────────────────────────
        ax_cnt = axes[1, col]
        cnt_vals = [float(row.get(m, 0) or 0) for m in cnt_metrics]
        ax_cnt.barh(range(len(cnt_metrics)), cnt_vals, color=color, alpha=0.6, edgecolor="white")
        ax_cnt.set_yticks(range(len(cnt_metrics)))
        ax_cnt.set_yticklabels(cnt_labels, fontsize=9)
        ax_cnt.set_xlabel("Count", fontsize=8)
        ax_cnt.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_cnt.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4, prune="both"))
        ax_cnt.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Company Profile Examples: Healthy / Partial Loss / Complete Exit\n"
                  "Synthetic data — not actual PANW figures",
                  fontsize=10, color="gray", style="italic")
    save_fig("company_examples.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    top20 = eda_top20_ordered_items()
    aov_results = eda_aov_analysis()
    eda_retention_growth()

    # ── Compute key findings stats (written to JSON so README claims stay current) ─
    findings_stats = {}
    try:
        _labeled = pd.read_csv(DATA_CLEAN / "renewal_unit_features_labeled.csv")
        if len(_labeled) > 0:
            # Tenure bands
            _labeled["tenure_yr"] = _labeled["tenure_days"] / 365.25
            _t = _labeled.copy()
            _t["band"] = pd.cut(_t["tenure_yr"],
                                bins=[0, 2, 3, float("inf")],
                                labels=["under_2yr", "2_to_3yr", "over_3yr"])
            _tenure = (_t.groupby("band")["churned"].mean() * 100).round(1).to_dict()

            # P1 ticket cohorts (90-day lookback)
            _has_p1  = _labeled[_labeled["p1_tickets_90d"] > 0]["churned"].mean() * 100
            _no_p1   = _labeled[_labeled["p1_tickets_90d"] == 0]["churned"].mean() * 100
            _n_has_p1 = int((_labeled["p1_tickets_90d"] > 0).sum())
            _n_no_p1  = int((_labeled["p1_tickets_90d"] == 0).sum())

            findings_stats = {
                "tenure_churn_pct": _tenure,
                "p1_ticket_churn_pct": round(float(_has_p1), 1),
                "no_p1_ticket_churn_pct": round(float(_no_p1), 1),
                "n_contracts_with_p1": _n_has_p1,
                "n_contracts_without_p1": _n_no_p1,
            }
    except Exception:
        pass  # findings_stats stays {} if labeled data unavailable

    summary = {
        "run_timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "key_findings_stats": findings_stats,
        "required_answers": {
            "A_top20_most_ordered_items": {
                "ranking_metric": "distinct_order_count",
                "eligible_cohort": "completed and partial orders only",
                "export": "outputs/eda_top20_ordered_items.csv",
                "chart": "figures/top20_ordered_items.png",
                "top1": {
                    "product":      str(top20.iloc[0]["product_name"]),
                    "order_count":  int(top20.iloc[0]["distinct_order_count"]),
                    "companies":    int(top20.iloc[0]["distinct_companies"]),
                    "net_sales":    int(top20.iloc[0]["net_sales_value"]),
                },
            },
            "B_highest_aov_customer": aov_results,
        },
        "charts_exported": [
            "figures/top20_ordered_items.png",
            "figures/aov_by_company.png",
            "figures/product_family_penetration.png",
            "figures/company_size_mix.png",
            "figures/churn_by_segment.png",
            "figures/revenue_concentration.png",
            "figures/deployment_coverage_trend.png",
            "figures/company_examples.png",
        ],
    }

    with open(OUTPUTS / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ EDA complete")
    print(f"  Required answer A (top 20): outputs/eda_top20_ordered_items.csv")
    print(f"  Required answer B (AOV):    outputs/eda_aov_analysis.csv")
    print(f"  Summary JSON:               outputs/eda_summary.json")
    print(f"  Charts:                     figures/ ({len(summary['charts_exported'])} files)")


if __name__ == "__main__":
    main()

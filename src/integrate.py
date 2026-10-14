"""
integrate.py
============
Builds four analytical datasets from the cleaned core tables:

  data/cleaned/order_item_fact.csv        — one row per order line (purchasing grain)
  data/cleaned/renewal_unit_features.csv  — one row per contract renewal unit × as_of_date
  data/cleaned/company_summary.csv        — one row per company (executive EDA grain)
  data/cleaned/deployment_week_features.csv — one row per deployment × week

Also exports:
  outputs/integration_lineage.json        — join keys, cardinalities, row counts

Run:
    python3 src/integrate.py

Point-in-time rule: as_of_date = scoring_date (2026-10-01).
No forward-outcome information is used to construct features.
Label window: 90 days after each contract term_end.

CORRECTIONS (v2):
  - Label join via expiring_contract_id (Bug #1 fix)
  - Telemetry filtered on available_at <= as_of_date (Bug #4 fix)
  - Multiple historical scoring dates from config
  - GRR/NRR computed in company_summary
"""

import fcntl
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "cleaned"
OUTPUTS = ROOT / "outputs"


def load_config() -> dict:
    with open(ROOT / "config" / "simulation.yaml") as f:
        return yaml.safe_load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_clean(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA_CLEAN / name, **kwargs)


def cardinality_check(left: pd.DataFrame, right: pd.DataFrame,
                       left_key: str, right_key: str, label: str,
                       allow_unmatched: bool = False) -> dict:
    """
    Verify expected many-to-one join cardinality using merge indicator.

    Checks:
      - Duplicate right keys (right is not unique on right_key → fanout in join)
      - Unmatched left rows (left_key has no match in right → broken FK)

    Raises ValueError on either condition unless allow_unmatched=True.
    allow_unmatched=True suppresses the unmatched-rows check (use for optional left joins).
    Fanout (duplicate right keys) always raises — it silently multiplies rows.
    """
    right_dupes = right[right_key].duplicated().sum()
    if right_dupes > 0:
        raise ValueError(
            f"[cardinality_check] {label}: {right_dupes} duplicate right-side keys on "
            f"'{right_key}' — join would fan out left rows, inflating counts."
        )

    merged = left[[left_key]].merge(
        right[[right_key]].drop_duplicates(subset=[right_key]),
        left_on=left_key, right_on=right_key,
        how="left", indicator=True,
    )
    n_unmatched = (merged["_merge"] == "left_only").sum()

    result = {
        "join": label,
        "left_rows": len(left),
        "right_rows": len(right),
        "right_duplicate_keys": int(right_dupes),
        "unmatched_left_rows": int(n_unmatched),
        "match_rate_pct": round(100 * (1 - n_unmatched / max(len(left), 1)), 2),
    }

    if n_unmatched > 0 and not allow_unmatched:
        sample = left[~left[left_key].isin(right[right_key])][left_key].head(5).tolist()
        raise ValueError(
            f"[cardinality_check] {label}: {n_unmatched} unmatched left-side keys on "
            f"'{left_key}' — FK integrity broken. Sample: {sample}"
        )
    return result


def _parse_datetime_validated(series: pd.Series, field_name: str,
                               max_new_null_pct: float = 0.01) -> pd.Series:
    """
    Parse a datetime string series with strict loss detection.

    Strategy:
      1. Try ISO 8601 with explicit format (handles YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS).
         pandas `format="ISO8601"` covers both without ambiguity.
      2. Where that still leaves NaTs on non-null input, try inferred format as fallback.
      3. Count newly-introduced NaTs vs original nulls; raise if over threshold.
    """
    original_null = series.isna().sum()

    # Step 1: ISO 8601 parse (covers date-only and datetime strings alike)
    try:
        parsed = pd.to_datetime(series, format="ISO8601", errors="coerce")
    except (ValueError, TypeError):
        parsed = pd.to_datetime(series, errors="coerce")

    # Step 2: fallback — try inferred parse for any remaining NaTs on non-null values
    still_null_mask = ~series.isna() & parsed.isna()
    if still_null_mask.any():
        fallback = pd.to_datetime(series[still_null_mask], errors="coerce", infer_datetime_format=True)
        parsed = parsed.copy()
        parsed[still_null_mask] = fallback

    new_nulls = parsed.isna().sum() - original_null
    if new_nulls > 0:
        bad_pct = new_nulls / max(len(series), 1)
        sample_bad = series[~series.isna() & parsed.isna()].head(5).tolist()
        msg = (
            f"[date parse] '{field_name}': {new_nulls} valid values became NaT "
            f"({bad_pct:.2%} of rows). Sample unparseable values: {sample_bad}"
        )
        if bad_pct > max_new_null_pct:
            raise ValueError(msg + f" (exceeds {max_new_null_pct:.2%} threshold)")
        # Log below-threshold losses (wrong even if not fatal)
        import warnings
        warnings.warn(msg, stacklevel=3)
    return parsed


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ── Churn label builder ───────────────────────────────────────────────────────

def build_churn_labels(events: pd.DataFrame) -> pd.DataFrame:
    """
    FIX (Bug #1): Cancellation events carry expiring_contract_id pointing to the
    contract being cancelled.  We resolve churn labels via that foreign key, not
    via the cancellation's own contract_id.

    Returns a DataFrame keyed by 'expiring_contract_id' with:
      churned (0/1), label_matured (bool)
    """
    cancel = events[events["event_type"] == "cancellation"].copy()
    # expiring_contract_id = the contract that was terminated
    cancel_labels = cancel.groupby("expiring_contract_id").agg(
        churned=("churned", "max"),
        label_matured=("label_matured", "max"),
    ).reset_index().rename(columns={"expiring_contract_id": "contract_id"})

    # For non-churned contracts (they renewed), churned = 0 and label_matured can be
    # set from their renewal event.
    renew = events[events["event_type"].isin(
        ["renewal", "renewal_expansion", "renewal_contraction"]
    )].copy()
    renew_labels = renew.groupby("expiring_contract_id").agg(
        churned=("churned", "max"),         # will be 0 for renewals
        label_matured=("label_matured", "max"),
    ).reset_index().rename(columns={"expiring_contract_id": "contract_id"})

    # Merge: cancellation label takes precedence over renewal label for same contract
    all_labels = pd.concat([cancel_labels, renew_labels], ignore_index=True)
    # If a contract appears in both (shouldn't, but guard), churn=1 wins
    labels = all_labels.sort_values("churned", ascending=False).drop_duplicates(
        subset=["contract_id"], keep="first"
    )
    return labels


# ── Dataset 1: order_item_fact ────────────────────────────────────────────────

def build_order_item_fact(lineage: list) -> pd.DataFrame:
    print("[integrate] Building order_item_fact...")

    items = read_clean("order_items.csv")
    orders = read_clean("orders.csv")
    users = read_clean("users.csv", usecols=["user_id", "company_id", "user_role", "primary_team_id"])
    products = read_clean("products.csv")

    n_items = len(items)

    # Many-to-one: items -> orders
    lineage.append(cardinality_check(items, orders, "order_id", "order_id",
                                      "order_items -> orders"))

    # Join items -> orders (bring in order metadata)
    fact = items.merge(
        orders[["order_id", "user_id", "company_id", "order_date",
                "order_type", "order_status", "sales_channel",
                "currency", "order_net_amount", "in_eligible_cohort"]],
        on="order_id", how="left"
    )

    # Many-to-one: orders -> users
    lineage.append(cardinality_check(
        fact[["user_id"]].drop_duplicates(), users, "user_id", "user_id",
        "orders -> users"
    ))

    fact = fact.merge(
        users[["user_id", "user_role", "primary_team_id"]].rename(
            columns={"primary_team_id": "buyer_primary_team_id"}
        ),
        on="user_id", how="left"
    )

    # Many-to-one: items -> products
    lineage.append(cardinality_check(items, products, "product_id", "product_id",
                                      "order_items -> products"))

    fact = fact.merge(
        products[["product_id", "product_name", "product_family",
                   "offering_type", "billing_unit", "recurring_flag",
                   "annual_catalog_price"]],
        on="product_id", how="left", suffixes=("", "_prod")
    )

    # Use net_acv directly from order_items (computed in generate_data.py with correct TCV formula)
    # Fallback recompute only if net_acv is missing
    has_net_acv = "net_acv" in fact.columns and fact["net_acv"].notna().any()
    if has_net_acv:
        fact["acv"] = fact["net_acv"]
    else:
        fact["term_start_dt"] = pd.to_datetime(fact["term_start"], errors="coerce")
        fact["term_end_dt"] = pd.to_datetime(fact["term_end"], errors="coerce")
        fact["term_days"] = (fact["term_end_dt"] - fact["term_start_dt"]).dt.days
        fact["term_months"] = (fact["term_days"] / 30.44).round(1)
        fact["acv"] = np.where(
            fact["recurring_flag_prod"] & fact["term_months"].notna() & (fact["term_months"] > 0),
            (fact["line_net_amount"] / fact["term_months"] * 12).round(2),
            None
        )
        fact = fact.drop(columns=["term_start_dt", "term_end_dt"], errors="ignore")

    # Buyer company must match order company — verify
    n_mismatch = (fact["company_id_x"] != fact["company_id_y"]).sum() \
        if "company_id_y" in fact.columns else 0
    if n_mismatch > 0:
        print(f"  [WARN] {n_mismatch} order/user company mismatches")

    # Clean up
    if "company_id_y" in fact.columns:
        fact = fact.drop(columns=["company_id_y"]).rename(columns={"company_id_x": "company_id"})
    if "recurring_flag_prod" in fact.columns:
        fact = fact.rename(columns={"recurring_flag_prod": "is_recurring"})

    n_fact = len(fact)
    lineage.append({
        "dataset": "order_item_fact",
        "rows_before_join": n_items,
        "rows_after_join": n_fact,
        "note": "One row per order item; no fanout expected",
    })

    fact.to_csv(DATA_CLEAN / "order_item_fact.csv", index=False)
    print(f"  order_item_fact: {n_fact:,} rows")
    return fact


# ── Adoption case classification ─────────────────────────────────────────────

def classify_adoption_case(
    obs_completeness_90d: float,
    coverage_recent_30d: float,
    coverage_delta_30d: float,
    p1_tickets_90d: int = 0,
    completeness_min: float = 0.70,
    coverage_adequate: float = 0.70,
    delta_stable: float = 0.10,
    delta_declining: float = -0.15,
    coverage_low: float = 0.40,
) -> str:
    """
    Route a renewal unit to one of four adoption health cases.

    Priority order (first matching rule wins):
      1. unknown_measurement  — obs_completeness_90d < completeness_min
         Telemetry is too sparse to assess adoption state reliably.
         Action: instrumentation / data engineering review.

      2. declining_adoption   — coverage_delta_30d <= delta_declining
         Coverage is measurably falling over a reliable observation window.
         Action: investigate root cause (champion turnover, workload change, etc.).

      3. healthy_low_touch    — coverage_recent_30d >= coverage_adequate
                                AND |coverage_delta_30d| <= delta_stable
                                AND p1_tickets_90d == 0
         Adequate and stable coverage with no active escalation.
         Action: standard renewal motion — no adoption alert required.

      4. persistent_under_adoption — coverage_recent_30d < coverage_low
                                     AND coverage_delta_30d > delta_declining (stable)
         Coverage is stably inadequate — a structural adoption gap, not a trend.
         Action: adoption/implementation review with the customer.

      5. review_needed  — all other combinations (adequate coverage with minor
         volatility, or moderate shortfall that doesn't meet thresholds).

    Parameters use NaN-safety: any NaN input for completeness or coverage
    returns unknown_measurement so the row is not silently misclassified.
    """
    import math

    # NaN guard: missing completeness or coverage → cannot classify
    if coverage_recent_30d is None or (isinstance(coverage_recent_30d, float) and math.isnan(coverage_recent_30d)):
        return "unknown_measurement"
    if obs_completeness_90d is None or (isinstance(obs_completeness_90d, float) and math.isnan(obs_completeness_90d)):
        return "unknown_measurement"
    if coverage_delta_30d is None or (isinstance(coverage_delta_30d, float) and math.isnan(coverage_delta_30d)):
        return "unknown_measurement"

    # Rule 1: insufficient telemetry
    if obs_completeness_90d < completeness_min:
        return "unknown_measurement"

    # Rule 2: declining with reliable data
    if coverage_delta_30d <= delta_declining:
        return "declining_adoption"

    # Rule 3: healthy — adequate, stable, no escalation
    if (coverage_recent_30d >= coverage_adequate
            and abs(coverage_delta_30d) <= delta_stable
            and p1_tickets_90d == 0):
        return "healthy_low_touch"

    # Rule 4: persistently low but not declining
    if coverage_recent_30d < coverage_low and coverage_delta_30d > delta_declining:
        return "persistent_under_adoption"

    return "review_needed"


# ── Engagement feature aggregation ───────────────────────────────────────────

def build_engagement_features(as_of_date: pd.Timestamp,
                               lookback_days: int = 90) -> pd.DataFrame:
    """
    Aggregate per-lineage engagement signals for the lookback window ending at
    as_of_date, reading directly from telemetry_weekly_clean.csv (the full
    5-year source) with strict point-in-time filtering:

      week_start in [as_of_date - lookback_days, as_of_date]
      available_at <= as_of_date  (no future data; null available_at = missing)

    This works for every historical scoring date, not only the current window.

    Returns one row per renewal_lineage_id with columns:
      avg_coverage_90d          — mean coverage_ratio (null-safe; NaN if all missing)
      coverage_trend_90d        — OLS slope of weekly coverage (+ve = improving)
      obs_completeness_90d      — fraction of expected weeks with non-missing telemetry
      policy_events_per_unit_90d— mean (policy_events / active_units) for active weeks
      remediation_velocity_90d  — mean remediation_events per week
      automation_adoption_90d   — fraction of weeks with automation_events > 0
      scan_success_rate_90d     — mean scan_success_rate (Prisma; NaN for others)
      missing_telemetry_pct_90d — fraction of weeks with missing telemetry
    """
    window_start = as_of_date - timedelta(days=lookback_days)

    telem_path = DATA_CLEAN / "telemetry_weekly_clean.csv"
    deps_path  = DATA_CLEAN / "deployments_clean.csv"
    if not telem_path.exists() or not deps_path.exists():
        return pd.DataFrame()

    deps = pd.read_csv(deps_path, usecols=["deployment_id", "renewal_lineage_id"])
    lin_map = deps.set_index("deployment_id")["renewal_lineage_id"].to_dict()

    needed_cols = ["deployment_id", "week_start", "available_at",
                   "active_units", "eligible_units", "policy_events",
                   "remediation_events", "automation_events",
                   "scan_success_rate", "telemetry_status", "is_late"]
    telem_cols = pd.read_csv(telem_path, nrows=0).columns.tolist()
    use_cols = [c for c in needed_cols if c in telem_cols]

    # Load ALL rows whose reporting PERIOD IS CLOSED at as_of_date.
    # A weekly report is "due" when its 7-day period has fully elapsed:
    #   week_start + 7 days <= as_of_date
    # Periods still in progress (week_start + 7 > as_of_date) are PENDING —
    # they are not overdue, and excluding them keeps the denominator honest.
    # Do NOT filter on available_at here: we need overdue rows (available_at >
    # as_of_date) in the denominator so that "1 received, 3 overdue" gives
    # 25% completeness rather than 100%.
    all_chunks, received_chunks = [], []
    for chunk in pd.read_csv(telem_path, chunksize=200_000, usecols=use_cols):
        chunk["week_start_dt"]   = pd.to_datetime(chunk["week_start"],   errors="coerce")
        chunk["available_at_dt"] = pd.to_datetime(chunk["available_at"], errors="coerce")
        in_window = (
            (chunk["week_start_dt"] >= window_start) &
            (chunk["week_start_dt"] + pd.Timedelta(days=7) <= as_of_date)
        )
        c_all = chunk[in_window].copy()
        if len(c_all):
            all_chunks.append(c_all)
        # Received: available by as_of_date (or missing telemetry — null available_at)
        c_recv = c_all[
            c_all["available_at_dt"].isna() | (c_all["available_at_dt"] <= as_of_date)
        ]
        if len(c_recv):
            received_chunks.append(c_recv)

    if not all_chunks:
        return pd.DataFrame()

    df_all  = pd.concat(all_chunks,      ignore_index=True)
    df      = pd.concat(received_chunks, ignore_index=True) if received_chunks else pd.DataFrame(columns=df_all.columns)

    for d in [df_all, df]:
        d["renewal_lineage_id"] = d["deployment_id"].map(lin_map)
    df_all = df_all.dropna(subset=["renewal_lineage_id"])
    df     = df.dropna(subset=["renewal_lineage_id"])
    if df_all.empty:
        return pd.DataFrame()

    # Classify observation status for each row in df_all
    # received   : available_at <= as_of_date (and not telemetry_status='missing')
    # missing    : telemetry_status == 'missing' (no sensor data generated)
    # overdue    : available_at > as_of_date (generated but not yet arrived)
    df_all["obs_status"] = np.where(
        df_all["telemetry_status"] == "missing", "missing",
        np.where(
            df_all["available_at_dt"].isna() | (df_all["available_at_dt"] > as_of_date),
            "overdue",
            "received",
        ),
    )
    # obs_completeness_90d: fraction of expected observations that have arrived.
    # Denominator = all rows in window (received + missing + overdue).
    # A single received row among many delayed rows gives correct low completeness.
    completeness = (
        df_all.groupby("renewal_lineage_id")
        .apply(lambda g: (g["obs_status"] == "received").sum() / max(len(g), 1))
        .reset_index(name="obs_completeness_90d")
    )

    # For coverage and signal aggregation, use only received rows (df).
    if df.empty:
        agg = completeness.copy()
        for col in ["avg_coverage_90d", "policy_events_per_unit_90d",
                    "remediation_velocity_90d", "automation_adoption_90d",
                    "scan_success_rate_90d", "missing_telemetry_pct_90d"]:
            agg[col] = np.nan
        agg["coverage_trend_90d"] = np.nan
        return agg

    df["is_missing"] = (df["telemetry_status"] == "missing").astype(int)
    df["coverage_ratio"] = np.where(
        df["is_missing"] == 1,
        np.nan,
        np.where(
            df["eligible_units"].fillna(0) > 0,
            (df["active_units"] / df["eligible_units"]).clip(0, 1),
            np.nan,
        ),
    )
    df["policy_per_unit"] = np.where(
        (df["active_units"].fillna(0) > 0) & df["policy_events"].notna(),
        df["policy_events"] / df["active_units"].clip(lower=1),
        np.nan,
    )

    # Coverage trend: OLS slope vs elapsed calendar weeks (Fix #5).
    # Using position indices treats gaps as equally-spaced; using actual days
    # correctly reflects the calendar distance between observations.
    def _slope_calendar(grp):
        s = grp.dropna(subset=["coverage_ratio"]).sort_values("week_start_dt")
        if len(s) < 3:
            return np.nan
        x = (s["week_start_dt"] - s["week_start_dt"].iloc[0]).dt.days.values.astype(float) / 7.0
        y = s["coverage_ratio"].values.astype(float)
        return float(np.polyfit(x, y, 1)[0])

    cov_trend = (
        df.groupby("renewal_lineage_id")
        .apply(_slope_calendar)
        .reset_index(name="coverage_trend_90d")
    )

    agg_signals = df.groupby("renewal_lineage_id").agg(
        avg_coverage_90d          = ("coverage_ratio",       "mean"),
        policy_events_per_unit_90d= ("policy_per_unit",      "mean"),
        remediation_velocity_90d  = ("remediation_events",   "mean"),
        automation_adoption_90d   = ("automation_events",    lambda x: (x.fillna(0) > 0).mean()),
        scan_success_rate_90d     = ("scan_success_rate",    "mean"),
        missing_telemetry_pct_90d = ("is_missing",           "mean"),
    ).reset_index()

    # Start from completeness — it contains EVERY lineage with at least one due
    # period, including lineages whose rows are all overdue (no received data).
    # Left-joining from agg_signals would silently drop those lineages.
    agg = completeness.copy()
    agg = agg.merge(agg_signals,  on="renewal_lineage_id", how="left")
    agg = agg.merge(cov_trend,    on="renewal_lineage_id", how="left")

    # ── 30-day coverage split: recent vs prior ───────────────────────────────
    # Supports the four behavioral cases from the change plan:
    #   healthy_low_touch, declining_adoption, unknown_measurement, persistent_under_adoption
    # Split the 90-day window into recent 30d and prior 30-60d.
    recent_cutoff = as_of_date - timedelta(days=30)
    prior_cutoff  = as_of_date - timedelta(days=60)

    df_recent = df[df["week_start_dt"] >= recent_cutoff]
    df_prior  = df[(df["week_start_dt"] >= prior_cutoff) & (df["week_start_dt"] < recent_cutoff)]

    cov_recent = (
        df_recent.groupby("renewal_lineage_id")["coverage_ratio"]
        .mean()
        .reset_index()
        .rename(columns={"coverage_ratio": "coverage_recent_30d"})
    )
    cov_prior = (
        df_prior.groupby("renewal_lineage_id")["coverage_ratio"]
        .mean()
        .reset_index()
        .rename(columns={"coverage_ratio": "coverage_prior_30d"})
    )
    agg = agg.merge(cov_recent, on="renewal_lineage_id", how="left")
    agg = agg.merge(cov_prior,  on="renewal_lineage_id", how="left")
    agg["coverage_delta_30d"] = agg["coverage_recent_30d"] - agg["coverage_prior_30d"]
    return agg


def build_interaction_features(as_of_date: pd.Timestamp, lookback_days: int = 180) -> pd.DataFrame:
    """
    Aggregate account relationship touchpoints per company for the lookback window.

    Point-in-time: only events with available_at <= as_of_date are used.
    No future scheduled or unlogged interactions are included.

    Returns one row per company_id with:
      executive_interactions_180d   — executive briefing count in last 180d
      marketing_engagements_180d    — marketing event count in last 180d
      days_since_last_interaction   — days since most recent event (any type)
    """
    ixn_path = DATA_CLEAN / "account_interactions_clean.csv"
    if not ixn_path.exists():
        return pd.DataFrame()

    window_start = as_of_date - timedelta(days=lookback_days)

    df = pd.read_csv(ixn_path, usecols=["company_id", "interaction_type", "event_at", "available_at"])
    df["event_at_dt"]     = pd.to_datetime(df["event_at"],     errors="coerce")
    df["available_at_dt"] = pd.to_datetime(df["available_at"], errors="coerce")

    # Point-in-time: only data logged by as_of_date, and event must not be in the future.
    # Both conditions are required: available_at prevents using unlogged events;
    # event_at prevents using events that haven't occurred yet at this scoring date.
    df = df[
        df["available_at_dt"].notna() &
        (df["available_at_dt"] <= as_of_date) &
        df["event_at_dt"].notna() &
        (df["event_at_dt"] <= as_of_date)
    ]
    # Within lookback window
    df_window = df[df["event_at_dt"] >= window_start]

    exec_counts = (
        df_window[df_window["interaction_type"] == "executive_briefing"]
        .groupby("company_id").size()
        .reset_index(name="executive_interactions_180d")
    )
    mkt_counts = (
        df_window[df_window["interaction_type"] == "marketing_event"]
        .groupby("company_id").size()
        .reset_index(name="marketing_engagements_180d")
    )
    last_ixn = (
        df.groupby("company_id")["event_at_dt"]
        .max()
        .reset_index(name="last_interaction_dt")
    )
    last_ixn["days_since_last_interaction"] = (
        (as_of_date - last_ixn["last_interaction_dt"]).dt.days
    )

    result = exec_counts.merge(mkt_counts, on="company_id", how="outer")
    result = result.merge(last_ixn[["company_id", "days_since_last_interaction"]], on="company_id", how="outer")
    result["executive_interactions_180d"] = result["executive_interactions_180d"].fillna(0).astype(int)
    result["marketing_engagements_180d"]  = result["marketing_engagements_180d"].fillna(0).astype(int)
    return result


# ── Dataset 2: renewal_unit_features ─────────────────────────────────────────

def build_renewal_unit_features_for(cfg: dict, as_of_date: pd.Timestamp,
                                     lineage: list) -> pd.DataFrame:
    """
    Build renewal_unit_features as-of a single scoring date.
    Point-in-time: features use only data available at as_of_date.
    Label: cancellation within label_window_days after term_end.
    """
    lookback_90  = as_of_date - timedelta(days=90)
    lookback_180 = as_of_date - timedelta(days=180)
    outcome_window_days = cfg["outcome_window_days"]

    items    = read_clean("order_items.csv")
    orders   = read_clean("orders.csv", usecols=["order_id", "company_id", "order_date", "order_type"])
    companies = read_clean("companies_clean.csv",
                            usecols=["company_id", "size_segment", "industry",
                                     "region", "employee_count", "channel"])
    products = read_clean("products.csv",
                           usecols=["product_id", "product_family", "offering_type",
                                    "product_launch_date"])
    tickets  = read_clean("support_tickets_clean.csv",
                           usecols=["company_id", "ticket_id", "priority",
                                    "created_at", "resolved_at", "resolution_hours"])
    events   = read_clean("contract_events_clean.csv")

    # Only recurring, positive-value items
    recur = items[
        (items["recurring_flag"] == True) & (items["line_net_amount"] > 0)
    ].merge(orders, on="order_id", how="left")
    recur = recur.merge(products, on="product_id", how="left")

    recur["term_end_dt"]   = pd.to_datetime(recur["term_end"],   errors="coerce")
    recur["term_start_dt"] = pd.to_datetime(recur["term_start"], errors="coerce")
    recur["order_date_dt"] = pd.to_datetime(recur["order_date"], errors="coerce")

    # Point-in-time filter: only orders placed at or before as_of_date
    recur = recur[recur["order_date_dt"] <= as_of_date]

    # Scoring eligibility: only contracts whose outcome is still unknown at as_of_date.
    # term_end must be in the future (>= as_of_date) and within the label window.
    # Contracts with term_end < as_of_date have known outcomes and must be excluded
    # from scoring rows to prevent label leakage into model inputs.
    renewal_window_end = as_of_date + timedelta(days=outcome_window_days)
    in_window = (
        recur["term_end_dt"].notna() &
        (recur["term_end_dt"] >= as_of_date) &
        (recur["term_end_dt"] <= renewal_window_end)
    )
    recur_window = recur[in_window].copy()

    # Days to renewal
    recur_window["days_to_renewal"] = (recur_window["term_end_dt"] - as_of_date).dt.days

    # ACV: use pre-computed net_acv from order_items if available
    if "net_acv" in recur_window.columns and recur_window["net_acv"].notna().any():
        recur_window["acv"] = recur_window["net_acv"]
    else:
        recur_window["term_days"] = (
            recur_window["term_end_dt"] - recur_window["term_start_dt"]
        ).dt.days
        recur_window["term_months"] = (recur_window["term_days"] / 30.44).round(1)
        recur_window["acv"] = np.where(
            recur_window["term_months"] > 0,
            (recur_window["line_net_amount"] / recur_window["term_months"] * 12).round(2),
            recur_window["line_net_amount"]
        )

    # Ticket signals (lookback_90 before as_of_date)
    # Point-in-time: only use resolution info known by as_of_date.
    # Tickets resolved AFTER as_of_date are still "open" at that snapshot.
    tickets["created_dt"]  = _parse_datetime_validated(tickets["created_at"],  "ticket.created_at")
    tickets["resolved_dt"] = _parse_datetime_validated(tickets["resolved_at"], "ticket.resolved_at")
    recent_tickets = tickets[
        (tickets["created_dt"] >= lookback_90) & (tickets["created_dt"] <= as_of_date)
    ].copy()
    # Resolution hours: null if resolved after cutoff (treat as still open)
    recent_tickets["resolved_by_cutoff"] = (
        recent_tickets["resolved_dt"].notna() &
        (recent_tickets["resolved_dt"] <= as_of_date)
    )
    recent_tickets["resolution_hrs_known"] = np.where(
        recent_tickets["resolved_by_cutoff"],
        recent_tickets["resolution_hours"],
        np.nan,   # unknown at this snapshot
    )
    # Age at cutoff for unresolved tickets (useful signal: how long has it been open?)
    recent_tickets["age_at_cutoff_hrs"] = np.where(
        ~recent_tickets["resolved_by_cutoff"],
        (as_of_date - recent_tickets["created_dt"]).dt.total_seconds() / 3600,
        np.nan,
    )
    ticket_summary = recent_tickets.groupby("company_id").agg(
        tickets_90d=("ticket_id", "count"),
        p1_tickets_90d=("priority", lambda x: (x == "P1").sum()),
        avg_resolution_hrs_90d=("resolution_hrs_known", "mean"),
        open_tickets_90d=("resolved_by_cutoff", lambda x: (~x).sum()),
        avg_open_age_hrs_90d=("age_at_cutoff_hrs", "mean"),
    ).reset_index()

    # Product breadth per company
    company_products = recur_window.groupby("company_id").agg(
        n_products_in_window=("product_id", "nunique"),
        n_families_in_window=("product_family", "nunique"),
        total_acv_in_window=("acv", "sum"),
    ).reset_index()

    features = recur_window.merge(companies, on="company_id", how="left")
    features = features.merge(ticket_summary, on="company_id", how="left")
    features = features.merge(company_products, on="company_id", how="left")

    # ── Product family features (contract-level, point-in-time) ─────────────
    # These three features capture breadth within the same product family and
    # whether sibling products are being disengaged. They are computed at the
    # (company_id, product_family) grain and then joined back to each contract.
    #
    # Mechanism: multi-product customers are stickier because switching cost
    # is higher and the vendor relationship is deeper. Conversely, if a company
    # is already cancelling other products in the same family, it signals a
    # broader disengagement from that platform — raising risk for remaining contracts.
    #
    # 1. n_active_products_in_family: distinct product_ids in same family with
    #    an active (non-cancelled) contract at as_of_date.
    # 2. family_adoption_completeness: fraction of that family's available products
    #    the company is using (0-1). Requires products catalog.
    # 3. family_sibling_cancellations_1yr: count of OTHER products in same family
    #    that had a cancellation order visible at as_of_date in the past year.
    #
    # Point-in-time: all joins use order_date <= as_of_date.

    _lookback_1yr = as_of_date - timedelta(days=365)

    # Active contracts per company × family: latest order per lineage (including
    # zero-value cancellation orders) is NOT a cancellation.
    # Must use the full recurring items (not just positive-value recur) so that
    # cancellation orders can mark a lineage as inactive.
    _all_recur = (
        items[items["recurring_flag"] == True]
        .merge(orders[["order_id", "company_id", "order_date"]], on="order_id", how="left")
        .merge(products[["product_id", "product_family"]], on="product_id", how="left")
    )
    _all_recur["_order_date_dt"] = pd.to_datetime(_all_recur["order_date"], errors="coerce")
    _all_recur = _all_recur[_all_recur["_order_date_dt"] <= as_of_date]
    _all_recur["term_end_dt2"] = pd.to_datetime(_all_recur["term_end"], errors="coerce")

    _recur_latest = (
        _all_recur[_all_recur["term_end_dt2"].notna()]
        .sort_values("_order_date_dt")
        .groupby("renewal_lineage_id")
        .last()
        .reset_index()
    )
    _active = _recur_latest[
        ~(
            (_recur_latest["term_years"].fillna(1) == 0) &
            (_recur_latest["net_acv"].fillna(1) == 0)
        ) &
        (_recur_latest["term_end_dt2"] >= as_of_date)  # must currently cover scoring date
    ]
    _active_by_co_fam = (
        _active.groupby(["company_id", "product_family"])["product_id"]
        .nunique()
        .reset_index(name="n_active_products_in_family")
    )

    # Products available per family from catalog — filter by product_launch_date so a
    # product that didn't exist at as_of_date doesn't inflate the denominator.
    _prods_available = products.copy()
    _launch_col = next((c for c in _prods_available.columns if c in ("product_launch_date", "launch_date")), None)
    if _launch_col:
        _prods_available["_launch_dt"] = pd.to_datetime(_prods_available[_launch_col], errors="coerce")
        _prods_available = _prods_available[
            _prods_available["_launch_dt"].isna() | (_prods_available["_launch_dt"] <= as_of_date)
        ]
    _fam_catalog_count = (
        _prods_available.groupby("product_family")["product_id"]
        .nunique()
        .reset_index(name="n_catalog_products_in_family")
    )
    _active_by_co_fam = _active_by_co_fam.merge(_fam_catalog_count, on="product_family", how="left")
    _active_by_co_fam["family_adoption_completeness"] = (
        _active_by_co_fam["n_active_products_in_family"] /
        _active_by_co_fam["n_catalog_products_in_family"].clip(lower=1)
    )

    # Cancellations of OTHER family products in past year (sibling churn signal)
    # NOTE: recur only contains positive-value orders; cancellation orders have
    # line_net_amount=0 and are filtered out.  Use the full items table instead.
    _items_for_cancel = (
        items[items["recurring_flag"] == True]
        .merge(orders[["order_id", "company_id", "order_date"]], on="order_id", how="left")
        .merge(products[["product_id", "product_family"]], on="product_id", how="left")
    )
    _items_for_cancel["_order_date_dt"] = pd.to_datetime(
        _items_for_cancel["order_date"], errors="coerce"
    )
    _cancels = _items_for_cancel[
        (_items_for_cancel["term_years"].fillna(1) == 0) &
        (_items_for_cancel["net_acv"].fillna(1) == 0) &
        (_items_for_cancel["_order_date_dt"] >= _lookback_1yr) &
        (_items_for_cancel["_order_date_dt"] <= as_of_date)
    ][["company_id", "product_family", "product_id", "renewal_lineage_id"]].copy()

    if len(_cancels) > 0:
        _sibling_cancels = (
            _cancels.groupby(["company_id", "product_family"])["product_id"]
            .nunique()
            .reset_index(name="family_sibling_cancellations_1yr")
        )
    else:
        _sibling_cancels = pd.DataFrame(
            columns=["company_id", "product_family", "family_sibling_cancellations_1yr"]
        )

    # Merge family features: join on (company_id, product_family) so each
    # contract sees its own family's breadth and sibling cancellation count.
    _fam_features = _active_by_co_fam[
        ["company_id", "product_family", "n_active_products_in_family", "family_adoption_completeness"]
    ]
    if len(_sibling_cancels) > 0:
        _fam_features = _fam_features.merge(_sibling_cancels, on=["company_id", "product_family"], how="left")
    else:
        _fam_features["family_sibling_cancellations_1yr"] = 0

    features = features.merge(_fam_features, on=["company_id", "product_family"], how="left")
    # Defaults: single-product companies have 1 active product, no sibling cancels
    features["n_active_products_in_family"] = features["n_active_products_in_family"].fillna(1).astype(int)
    features["family_adoption_completeness"] = features["family_adoption_completeness"].fillna(
        1.0 / features["product_family"].map(
            _prods_available.groupby("product_family")["product_id"].nunique()
        ).clip(lower=1)
    )
    features["family_sibling_cancellations_1yr"] = features["family_sibling_cancellations_1yr"].fillna(0).astype(int)

    # A5: Exclude self: if the target product_id itself was cancelled in the lookback window
    # for this company/family, subtract 1 so only true siblings are counted.
    _self_cancel_set = set(zip(_cancels["company_id"], _cancels["product_id"]))
    features["_self_in_cancel"] = [
        1 if (r.company_id, r.product_id) in _self_cancel_set else 0
        for r in features[["company_id", "product_id"]].itertuples(index=False)
    ]
    features["family_sibling_cancellations_1yr"] = (
        features["family_sibling_cancellations_1yr"] - features["_self_in_cancel"]
    ).clip(lower=0).astype(int)
    features = features.drop(columns=["_self_in_cancel"])

    # ── Deployment engagement signals (90d lookback per lineage) ─────────────
    # deployment_week_features is written by a prior step; may not exist on the
    # first run.  Left-join: missing lineages get NaN (expected for new products).
    eng = build_engagement_features(as_of_date, lookback_days=90)
    if len(eng) > 0 and "renewal_lineage_id" in features.columns:
        features = features.merge(eng, on="renewal_lineage_id", how="left")

    # Relationship interaction features (point-in-time, 180d lookback)
    ixn_feats = build_interaction_features(as_of_date, lookback_days=180)
    if len(ixn_feats) > 0:
        features = features.merge(ixn_feats, on="company_id", how="left")
        features["executive_interactions_180d"] = features["executive_interactions_180d"].fillna(0).astype(int)
        features["marketing_engagements_180d"]  = features["marketing_engagements_180d"].fillna(0).astype(int)

    # ── New forward-experience features (WP4) ────────────────────────────────
    # All 10 features are point-in-time: only data with available_at <= as_of_date.

    # Build deployment_id → renewal_lineage_id map for telemetry join
    ent_dep_path = DATA_CLEAN / "entitlement_deployment_map_clean.csv"
    _dep_lineage_map: dict = {}
    if ent_dep_path.exists():
        _ent = pd.read_csv(ent_dep_path, usecols=["contract_id", "deployment_id"])
        _dep_lineage_map = _ent.set_index("contract_id")["deployment_id"].to_dict()

    # --- Features 1-4: from telemetry new columns (scope/deployed/workflow) ---
    telem_path_new = DATA_CLEAN / "telemetry_weekly_clean.csv"
    _telem_feats = pd.DataFrame()
    if telem_path_new.exists():
        _telem_needed = ["deployment_id", "week_start", "available_at",
                         "scope_targets", "deployed_targets",
                         "productive_reach", "productive_opportunities",
                         "workflow_attempts", "workflow_successes"]
        _telem_cols_avail = pd.read_csv(telem_path_new, nrows=0).columns.tolist()
        _telem_use = [c for c in _telem_needed if c in _telem_cols_avail]

        if len(_telem_use) > 3:  # must have at least deployment_id, week_start, available_at
            _lookback_30  = as_of_date - timedelta(days=30)
            _lookback_60  = as_of_date - timedelta(days=60)
            _t_chunks = []
            for _chunk in pd.read_csv(telem_path_new, chunksize=200_000, usecols=_telem_use):
                _chunk["week_start_dt"]   = pd.to_datetime(_chunk["week_start"],   errors="coerce")
                _chunk["available_at_dt"] = pd.to_datetime(_chunk["available_at"], errors="coerce")
                _in_30 = (
                    (_chunk["week_start_dt"] >= _lookback_30) &
                    (_chunk["week_start_dt"] <= as_of_date) &
                    (
                        _chunk["available_at_dt"].isna() |
                        (_chunk["available_at_dt"] <= as_of_date)
                    )
                )
                _c = _chunk[_in_30].copy()
                if len(_c) > 0:
                    _t_chunks.append(_c)
            _prior_chunks = []
            for _chunk in pd.read_csv(telem_path_new, chunksize=200_000, usecols=_telem_use):
                _chunk["week_start_dt"]   = pd.to_datetime(_chunk["week_start"],   errors="coerce")
                _chunk["available_at_dt"] = pd.to_datetime(_chunk["available_at"], errors="coerce")
                _in_prior = (
                    (_chunk["week_start_dt"] >= _lookback_60) &
                    (_chunk["week_start_dt"] < _lookback_30) &
                    (
                        _chunk["available_at_dt"].isna() |
                        (_chunk["available_at_dt"] <= as_of_date)
                    )
                )
                _c = _chunk[_in_prior].copy()
                if len(_c) > 0:
                    _prior_chunks.append(_c)

            # Add renewal_lineage_id via deployment→entitlement map
            _deps_for_lin = pd.read_csv(DATA_CLEAN / "deployments_clean.csv",
                                         usecols=["deployment_id", "renewal_lineage_id"])
            _lin_map_t = _deps_for_lin.set_index("deployment_id")["renewal_lineage_id"].to_dict()

            _dep_to_co = {}
            if ent_dep_path.exists():
                _ent2 = pd.read_csv(ent_dep_path, usecols=["contract_id", "deployment_id"])
                # Map deployment_id → company_id via contract_id → recur
                _ctr_co_map = recur.drop_duplicates("contract_id").set_index("contract_id")["company_id"].to_dict()
                for _, _r in _ent2.iterrows():
                    _co = _ctr_co_map.get(str(_r["contract_id"]))
                    if _co:
                        _dep_to_co[_r["deployment_id"]] = _co

            def _agg_telem_feats(chunks, suffix):
                if not chunks:
                    return pd.DataFrame()
                _df = pd.concat(chunks, ignore_index=True)
                _df["renewal_lineage_id"] = _df["deployment_id"].map(_lin_map_t)
                _df = _df.dropna(subset=["renewal_lineage_id"])
                _sc = "scope_targets"   if "scope_targets"   in _df.columns else None
                _dt = "deployed_targets" if "deployed_targets" in _df.columns else None
                _pr = "productive_reach" if "productive_reach" in _df.columns else None
                _wa = "workflow_attempts" if "workflow_attempts" in _df.columns else None
                _ws = "workflow_successes" if "workflow_successes" in _df.columns else None
                _agg = {}
                if _sc and _dt:
                    _agg[f"sum_scope{suffix}"]    = (_sc, "sum")
                    _agg[f"sum_deployed{suffix}"] = (_dt, "sum")
                if _pr:
                    _agg[f"sum_pr{suffix}"] = (_pr, "sum")
                if _wa and _ws:
                    _agg[f"sum_wa{suffix}"] = (_wa, "sum")
                    _agg[f"sum_ws{suffix}"] = (_ws, "sum")
                if not _agg:
                    return pd.DataFrame()
                return _df.groupby("renewal_lineage_id").agg(**_agg).reset_index()

            _agg_30   = _agg_telem_feats(_t_chunks,     "_30")
            _agg_prior = _agg_telem_feats(_prior_chunks, "_prior")

            if len(_agg_30) > 0:
                _telem_feats = _agg_30.copy()
                # Feature 1: deployment_reach_30d
                if "sum_scope_30" in _telem_feats.columns and "sum_deployed_30" in _telem_feats.columns:
                    _telem_feats["deployment_reach_30d"] = np.where(
                        _telem_feats["sum_scope_30"] > 0,
                        (_telem_feats["sum_deployed_30"] / _telem_feats["sum_scope_30"]).clip(0, 1),
                        np.nan,
                    )
                # Feature 4: workflow_success_30d
                if "sum_wa_30" in _telem_feats.columns and "sum_ws_30" in _telem_feats.columns:
                    _telem_feats["workflow_success_30d"] = np.where(
                        _telem_feats["sum_wa_30"] > 0,
                        (_telem_feats["sum_ws_30"] / _telem_feats["sum_wa_30"]).clip(0, 1),
                        np.nan,
                    )
                # Feature 3: productive_use_delta_30d (current vs prior period)
                # Normalize by workflow_attempts (opportunities), not by scope_targets,
                # so the delta measures effectiveness change rather than workload change.
                if len(_agg_prior) > 0 and "sum_pr_30" in _telem_feats.columns:
                    _prior_cols = ["renewal_lineage_id", "sum_pr_prior"]
                    if "sum_wa_prior" in _agg_prior.columns:
                        _prior_cols.append("sum_wa_prior")
                    _telem_feats = _telem_feats.merge(
                        _agg_prior[[c for c in _prior_cols if c in _agg_prior.columns]],
                        on="renewal_lineage_id", how="left"
                    )
                    # Normalize by workflow attempts to measure success-rate change
                    def _norm_by_attempts(df, pr_col, wa_col):
                        if pr_col in df.columns and wa_col in df.columns:
                            return np.where(df[wa_col] > 0, df[pr_col] / df[wa_col].clip(lower=1), np.nan)
                        return np.nan
                    _curr_ratio = _norm_by_attempts(_telem_feats, "sum_pr_30", "sum_wa_30") \
                        if "sum_wa_30" in _telem_feats.columns else np.nan
                    _prior_ratio = _norm_by_attempts(_telem_feats, "sum_pr_prior", "sum_wa_prior") \
                        if "sum_wa_prior" in _telem_feats.columns else np.nan
                    _telem_feats["productive_use_delta_30d"] = (
                        np.array(_curr_ratio, dtype=float) - np.array(_prior_ratio, dtype=float)
                        if not (np.isscalar(_curr_ratio) and np.isnan(_curr_ratio))
                        else np.nan
                    )
                else:
                    _telem_feats["productive_use_delta_30d"] = np.nan

                # Feature 2: onboarding_delay_days — observable proxy from telemetry
                # Approx: if deployment_reach_30d < 0.4 and tenure < 180d → days past 90-day milestone
                if "deployment_reach_30d" in _telem_feats.columns:
                    _telem_feats["onboarding_delay_days_proxy"] = np.nan  # default
                    # Will be set per-contract after tenure join below

                # Drop working columns
                _drop = [c for c in _telem_feats.columns if c.startswith("sum_")]
                _telem_feats = _telem_feats.drop(columns=_drop, errors="ignore")

    # Join telem features onto features by renewal_lineage_id
    if len(_telem_feats) > 0 and "renewal_lineage_id" in features.columns:
        features = features.merge(_telem_feats, on="renewal_lineage_id", how="left")

    # --- Features 5-6: from support_tickets extended columns ---
    _tkt_path = DATA_CLEAN / "support_tickets_clean.csv"
    _tkt_ext_feats = pd.DataFrame()
    if _tkt_path.exists():
        _tkt_cols_avail = pd.read_csv(_tkt_path, nrows=0).columns.tolist()
        _tkt_ext_needed = ["company_id", "ticket_id", "created_at", "resolved_at",
                           "customer_impact", "impact_hours_open", "sla_breached",
                           "resolution_hours", "priority",
                           "contract_id"]
        _tkt_ext_use = [c for c in _tkt_ext_needed if c in _tkt_cols_avail]
        if "company_id" in _tkt_ext_use and "ticket_id" in _tkt_ext_use:
            _lookback_90 = as_of_date - timedelta(days=90)
            _tkt_ext = pd.read_csv(_tkt_path, usecols=_tkt_ext_use)
            _tkt_ext["created_dt"]  = _parse_datetime_validated(_tkt_ext["created_at"],  "tkt.created_at")
            _tkt_ext["resolved_dt"] = _parse_datetime_validated(_tkt_ext["resolved_at"], "tkt.resolved_at") \
                if "resolved_at" in _tkt_ext.columns else pd.NaT
            # Point-in-time filter: tickets created in last 90 days, available by as_of_date
            _tkt_90 = _tkt_ext[
                (_tkt_ext["created_dt"] >= _lookback_90) &
                (_tkt_ext["created_dt"] <= as_of_date)
            ].copy()

            if "customer_impact" in _tkt_90.columns and len(_tkt_90) > 0:
                _tkt_90["customer_impact"] = _tkt_90["customer_impact"].fillna(False).astype(bool)

                # A1: Compute elapsed hours at scoring date (not stored impact_hours_open which extends to obs_end)
                _tkt_90["is_unresolved"] = (
                    _tkt_90["customer_impact"] &
                    (
                        _tkt_90["resolved_dt"].isna() |
                        (_tkt_90["resolved_dt"] > as_of_date)
                    )
                )
                # Compute elapsed hours at as_of_date for unresolved; actual resolution hours for resolved
                def _elapsed_hrs(row):
                    created = row["created_dt"]
                    resolved = row["resolved_dt"]
                    if pd.isna(created):
                        return 0.0
                    if pd.isna(resolved) or resolved > as_of_date:
                        # Unresolved: elapsed from created to as_of_date, clamped to 90d lookback
                        hrs = (as_of_date - created).total_seconds() / 3600
                        return max(0.0, min(hrs, 90 * 24))
                    else:
                        # Resolved before as_of_date: actual resolution duration
                        return max(0.0, (resolved - created).total_seconds() / 3600)
                _tkt_90["_elapsed_hrs"] = _tkt_90.apply(_elapsed_hrs, axis=1)

                # A2: Compute SLA breach status at scoring date (not stored sla_breached which uses future resolution)
                _sla_targets = {"P1": 4.0, "P2": 8.0, "P3": 24.0, "P4": 72.0}
                def _sla_breached_at_scoring(row):
                    pri = str(row.get("priority", "P3")) if "priority" in _tkt_90.columns else "P3"
                    target = _sla_targets.get(pri, 24.0)
                    created = row["created_dt"]
                    resolved = row["resolved_dt"]
                    if pd.isna(created):
                        return False
                    if not pd.isna(resolved) and resolved <= as_of_date:
                        # Resolved: use actual resolution hours
                        res_hrs = (resolved - created).total_seconds() / 3600
                        return res_hrs > target
                    else:
                        # Unresolved: check if already past SLA target
                        elapsed = (as_of_date - created).total_seconds() / 3600
                        return elapsed > target
                _tkt_90["_sla_breached_at_scoring"] = _tkt_90.apply(_sla_breached_at_scoring, axis=1)

                # P1/P2 customer-impacting tickets that breached SLA at scoring date
                _tkt_90["_is_p1p2"] = _tkt_90.get("priority", pd.Series("P3", index=_tkt_90.index)).isin(["P1", "P2"]) \
                    if "priority" in _tkt_90.columns else False

                _ci_agg = _tkt_90.groupby("company_id").agg(
                    unresolved_customer_impact_hours_90d=(
                        "_elapsed_hrs",
                        lambda x: x[_tkt_90.loc[x.index, "is_unresolved"]].sum()
                    ),
                    repeat_sla_breaches_90d=(
                        "_sla_breached_at_scoring",
                        lambda x: (
                            x & _tkt_90.loc[x.index, "customer_impact"] &
                            _tkt_90.loc[x.index, "_is_p1p2"]
                        ).sum()
                    ),
                ).reset_index()
                _tkt_ext_feats = _ci_agg

    if len(_tkt_ext_feats) > 0 and "company_id" in features.columns:
        features = features.merge(_tkt_ext_feats, on="company_id", how="left")
        features["unresolved_customer_impact_hours_90d"] = \
            features["unresolved_customer_impact_hours_90d"].fillna(0.0)
        features["repeat_sla_breaches_90d"] = \
            features["repeat_sla_breaches_90d"].fillna(0).astype(int)
    else:
        features["unresolved_customer_impact_hours_90d"] = np.nan
        features["repeat_sla_breaches_90d"] = np.nan

    # --- Feature 7: quoted_unit_price_change_pct from renewal_quotes ---
    _quotes_path = DATA_CLEAN / "renewal_quotes_clean.csv"
    _quote_feats = pd.DataFrame()
    if _quotes_path.exists():
        _q = pd.read_csv(_quotes_path,
                          usecols=["quote_id", "contract_id", "issued_at", "available_at",
                                   "unit_price_change_pct"])
        _q["available_at_dt"] = pd.to_datetime(_q["available_at"], errors="coerce")
        # Point-in-time: only quotes available at as_of_date
        _q_pit = _q[
            _q["available_at_dt"].notna() &
            (_q["available_at_dt"] <= as_of_date)
        ].copy()
        if len(_q_pit) > 0:
            # Most recent quote per contract
            _q_pit = _q_pit.sort_values("available_at_dt", ascending=False)
            _q_latest = _q_pit.drop_duplicates(subset=["contract_id"], keep="first")
            _quote_feats = _q_latest[["contract_id", "unit_price_change_pct"]].rename(
                columns={"unit_price_change_pct": "quoted_unit_price_change_pct"}
            )

    if len(_quote_feats) > 0 and "contract_id" in features.columns:
        features = features.merge(_quote_feats, on="contract_id", how="left")
    else:
        features["quoted_unit_price_change_pct"] = np.nan

    # --- Features 8-10: from account_interactions extended columns ---
    _ixn_path = DATA_CLEAN / "account_interactions_clean.csv"
    _ixn_ext_feats = pd.DataFrame()
    if _ixn_path.exists():
        _ixn_cols_avail = pd.read_csv(_ixn_path, nrows=0).columns.tolist()
        _ixn_ext_needed = ["company_id", "interaction_id", "interaction_type",
                            "event_at", "available_at",
                            "is_owner_handoff", "new_owner_id",
                            "agreed_action_id", "agreed_action_due_date",
                            "agreed_action_completed_at"]
        _ixn_ext_use = [c for c in _ixn_ext_needed if c in _ixn_cols_avail]
        if "company_id" in _ixn_ext_use:
            _lookback_180 = as_of_date - timedelta(days=180)
            _ixn_ext = pd.read_csv(_ixn_path, usecols=_ixn_ext_use)
            _ixn_ext["event_at_dt"]     = pd.to_datetime(_ixn_ext["event_at"],     errors="coerce")
            _ixn_ext["available_at_dt"] = pd.to_datetime(_ixn_ext["available_at"], errors="coerce")

            # Point-in-time filter
            _ixn_pit = _ixn_ext[
                _ixn_ext["available_at_dt"].notna() &
                (_ixn_ext["available_at_dt"] <= as_of_date)
            ].copy()

            # Feature 8: dated_budget_concern (last 180 days)
            _ixn_180 = _ixn_pit[_ixn_pit["event_at_dt"] >= _lookback_180]
            if "interaction_type" in _ixn_180.columns:
                _budget_concern = (
                    _ixn_180[_ixn_180["interaction_type"] == "budget_concern"]
                    .groupby("company_id").size()
                    .reset_index(name="_bc_count")
                )
                _budget_concern["dated_budget_concern"] = 1

            # Feature 9: days_without_confirmed_owner (B7: tracks vendor relationship owner gaps —
            # CSM/AE/TAM coverage — not customer champion.  A gap means no vendor owner is confirmed.)
            # Logic:
            #   - If an owner-departure handoff occurred (is_owner_handoff=True, new_owner_id=null)
            #     and no replacement handoff followed → days since departure
            #   - If a replacement handoff (is_owner_handoff=True, new_owner_id not null) occurred
            #     after last departure → 0 (owner confirmed)
            #   - If no handoff records exist but account has cs_meeting interactions → 0
            #     (owner continuously present, no observed departures)
            #   - If no interactions at all → NaN (unknown)
            _owner_feats = pd.DataFrame()
            if "is_owner_handoff" in _ixn_pit.columns:
                # Accounts with any interaction — owner assumed present if no departures seen
                _accs_with_ixn = _ixn_pit["company_id"].unique()
                _owner_base = pd.DataFrame({"company_id": _accs_with_ixn,
                                            "days_without_confirmed_owner": 0})

                # Find accounts where the LAST handoff event was a departure (new_owner_id is null)
                _handoffs_all = _ixn_pit[
                    _ixn_pit["is_owner_handoff"].fillna(False).astype(bool)
                ].copy()
                if len(_handoffs_all) > 0:
                    _handoffs_all["has_replacement"] = _handoffs_all["new_owner_id"].notna()
                    # Get the last handoff per account and check if it was a departure
                    _last_h = (
                        _handoffs_all.sort_values("available_at_dt")
                        .groupby("company_id").last()
                        .reset_index()
                    )
                    _departures = _last_h[~_last_h["has_replacement"]][["company_id", "available_at_dt"]]
                    if len(_departures) > 0:
                        _departures = _departures.copy()
                        _departures["days_without_confirmed_owner"] = (
                            (as_of_date - _departures["available_at_dt"]).dt.days
                        ).clip(lower=0)
                        # Merge departures into the base (override 0 with actual gap)
                        _owner_base = _owner_base.merge(
                            _departures[["company_id", "days_without_confirmed_owner"]],
                            on="company_id", how="left", suffixes=("_base", "_dep")
                        )
                        _owner_base["days_without_confirmed_owner"] = _owner_base[
                            "days_without_confirmed_owner_dep"
                        ].fillna(_owner_base["days_without_confirmed_owner_base"])
                        _owner_base = _owner_base[["company_id", "days_without_confirmed_owner"]]

                _owner_feats = _owner_base

            # Feature 10: overdue_agreed_actions
            _action_feats = pd.DataFrame()
            if "agreed_action_id" in _ixn_pit.columns:
                _actions = _ixn_pit[_ixn_pit["agreed_action_id"].notna()].copy()
                if len(_actions) > 0:
                    _actions["due_dt"]       = pd.to_datetime(_actions["agreed_action_due_date"], errors="coerce")
                    _actions["completed_dt"] = pd.to_datetime(_actions["agreed_action_completed_at"], errors="coerce")
                    _overdue = _actions[
                        _actions["due_dt"].notna() &
                        (_actions["due_dt"] < as_of_date) &
                        (
                            _actions["completed_dt"].isna() |
                            (_actions["completed_dt"] > as_of_date)
                        )
                    ]
                    _action_feats = (
                        _overdue.groupby("company_id").size()
                        .reset_index(name="overdue_agreed_actions")
                    )

            # Merge interaction extension features
            _ixn_ext_feats_list = []
            # Budget concern
            if "interaction_type" in _ixn_180.columns and "_budget_concern" in dir():
                _ixn_ext_feats_list.append(
                    _budget_concern[["company_id", "dated_budget_concern"]]
                )
            if len(_owner_feats) > 0:
                _ixn_ext_feats_list.append(_owner_feats)
            if len(_action_feats) > 0:
                _ixn_ext_feats_list.append(_action_feats)

            if _ixn_ext_feats_list:
                _ixn_ext_merged = _ixn_ext_feats_list[0]
                for _f in _ixn_ext_feats_list[1:]:
                    _ixn_ext_merged = _ixn_ext_merged.merge(_f, on="company_id", how="outer")
                _ixn_ext_feats = _ixn_ext_merged

    if len(_ixn_ext_feats) > 0 and "company_id" in features.columns:
        features = features.merge(_ixn_ext_feats, on="company_id", how="left")

    # Fill missing values for new features
    for _col, _fill in [
        ("dated_budget_concern", 0),
        ("days_without_confirmed_owner", np.nan),   # NaN = no interaction data; 0 = owner present
        ("overdue_agreed_actions", 0),
    ]:
        if _col not in features.columns:
            features[_col] = _fill if _fill != np.nan else np.nan
        else:
            if _fill != np.nan:
                features[_col] = features[_col].fillna(_fill)

    # Feature 2 (onboarding_delay_days): computed after tenure join below
    features["onboarding_delay_days"] = np.nan  # placeholder; filled after tenure_days is available

    # ── Label join (FIX Bug #1) ───────────────────────────────────────────────
    # Cancellation events use expiring_contract_id to link back to the expiring
    # contract. We CANNOT join on the cancellation's own contract_id.
    labels = build_churn_labels(events)
    features = features.merge(labels[["contract_id", "churned"]], on="contract_id", how="left")

    # label_matured: re-derive from the EXPIRING contract's term_end.
    obs_end          = pd.Timestamp(cfg["observation_end"])
    label_window_days = cfg["label_window_days"]
    features["term_end_dt2"] = pd.to_datetime(features["term_end"], errors="coerce")
    features["label_matured"] = (
        features["term_end_dt2"] + pd.Timedelta(days=label_window_days) <= obs_end
    )
    features = features.drop(columns=["term_end_dt2"], errors="ignore")

    # Outcome semantics (Issue 6):
    #   label_matured=True  → outcome is known: churned=1 or churned=0
    #   label_matured=False → outcome is UNKNOWN at this snapshot: churned=null
    # Using fillna(0) for immature rows would create false negatives in training.
    # Models must filter to label_matured=True for supervised training.
    features["churned"] = np.where(
        features["label_matured"],
        features["churned"].fillna(0).astype(float),  # 0 = retained, 1 = churned
        np.nan,                                        # unknown outcome
    )

    # Fill ticket NaNs
    for col in ["tickets_90d", "p1_tickets_90d", "open_tickets_90d"]:
        if col in features.columns:
            features[col] = features[col].fillna(0).astype(int)

    # Tenure: days from first order to as_of_date
    first_orders = recur.groupby("company_id")["order_date_dt"].min().reset_index()
    first_orders["tenure_days"] = (as_of_date - first_orders["order_date_dt"]).dt.days
    features = features.merge(first_orders[["company_id", "tenure_days"]], on="company_id", how="left")

    # Feature 2 (onboarding_delay_days): based on planned_first_value_date milestone from deployments.
    # B6: Read milestone columns from deployments_clean.csv and compute delay at as_of_date.
    _deps_milestone_path = DATA_CLEAN / "deployments_clean.csv"
    if _deps_milestone_path.exists():
        _deps_cols_avail = pd.read_csv(_deps_milestone_path, nrows=0).columns.tolist()
        _milestone_cols = ["renewal_lineage_id", "planned_first_value_date", "first_value_achieved_date"]
        _milestone_use = [c for c in _milestone_cols if c in _deps_cols_avail]
        if "renewal_lineage_id" in _milestone_use and "planned_first_value_date" in _milestone_use:
            _deps_ms = pd.read_csv(_deps_milestone_path, usecols=_milestone_use)
            _deps_ms["_planned_fv"] = pd.to_datetime(_deps_ms["planned_first_value_date"], errors="coerce")
            if "first_value_achieved_date" in _deps_ms.columns:
                _deps_ms["_achieved_fv"] = pd.to_datetime(_deps_ms["first_value_achieved_date"], errors="coerce")
            else:
                _deps_ms["_achieved_fv"] = pd.NaT

            def _compute_onboard_delay(row):
                planned = row["_planned_fv"]
                achieved = row["_achieved_fv"]
                if pd.isna(planned):
                    return np.nan
                # Treat a future achievement date as not yet achieved at scoring date:
                # a completion recorded after as_of_date is not visible yet.
                _achieved_known = (
                    not pd.isna(achieved) and achieved <= as_of_date
                )
                if not _achieved_known and planned < as_of_date:
                    # Still unfinished (or completed in the future) and past planned date
                    return float((as_of_date - planned).days)
                elif _achieved_known and achieved > planned:
                    # Achieved late but before scoring date
                    return float((achieved - planned).days)
                else:
                    return 0.0

            _deps_ms["_onboard_delay"] = _deps_ms.apply(_compute_onboard_delay, axis=1)
            # Take max delay per lineage (if multiple deployments)
            _lin_delay = _deps_ms.groupby("renewal_lineage_id")["_onboard_delay"].max().reset_index()
            _lin_delay = _lin_delay.rename(columns={"_onboard_delay": "onboarding_delay_days"})
            if "renewal_lineage_id" in features.columns:
                features = features.drop(columns=["onboarding_delay_days"], errors="ignore")
                features = features.merge(_lin_delay, on="renewal_lineage_id", how="left")
        else:
            # Fallback: tenure-based proxy if milestone columns absent
            if "deployment_reach_30d" in features.columns and "tenure_days" in features.columns:
                _reach_low = features["deployment_reach_30d"].fillna(1.0) < 0.4
                _tenure_early = features["tenure_days"].fillna(999) < 180
                _overdue = (_reach_low & _tenure_early)
                features["onboarding_delay_days"] = np.where(
                    features["deployment_reach_30d"].isna(),
                    np.nan,
                    np.where(_overdue, np.maximum(0, features["tenure_days"].fillna(0) - 90), 0)
                )
    else:
        # Fallback: tenure-based proxy if deployments_clean.csv absent
        if "deployment_reach_30d" in features.columns and "tenure_days" in features.columns:
            _reach_low = features["deployment_reach_30d"].fillna(1.0) < 0.4
            _tenure_early = features["tenure_days"].fillna(999) < 180
            _overdue = (_reach_low & _tenure_early)
            features["onboarding_delay_days"] = np.where(
                features["deployment_reach_30d"].isna(),
                np.nan,
                np.where(_overdue, np.maximum(0, features["tenure_days"].fillna(0) - 90), 0)
            )

    # Median prior renewal delay: typical gap between predecessor expiry and booking date.
    # Point-in-time: include completed renewals whose order_date <= as_of_date.
    # (Do NOT filter on term_end < as_of_date — that excludes renewals booked but not
    # yet expired, which are fully known at this snapshot and reduce the sample.)
    #
    # Correct formula: delay = renewal_order_date − predecessor_contract.term_end
    # The predecessor's term_end is NOT the new contract's term_start (they can differ).
    # Look up each row's predecessor_contract_id in the full recur term_end map.
    _ren_hist = recur[
        recur["order_type"].isin(["renewal", "renewal_expansion", "renewal_contraction"]) &
        recur["prior_net_acv"].notna() &
        recur["predecessor_contract_id"].notna() &
        (recur["order_date_dt"] <= as_of_date)
    ].copy()
    if len(_ren_hist) > 0:
        # Build predecessor term_end lookup from all items seen so far
        pred_term_end = (
            recur.drop_duplicates("contract_id")
            .set_index("contract_id")["term_end_dt"]
            .to_dict()
        )
        _ren_hist["pred_term_end"] = _ren_hist["predecessor_contract_id"].map(pred_term_end)
        _ren_hist = _ren_hist.dropna(subset=["pred_term_end"])
        if len(_ren_hist) > 0:
            _ren_hist["renewal_delay_days"] = (
                _ren_hist["order_date_dt"] - _ren_hist["pred_term_end"]
            ).dt.days.clip(-30, 365)
            renewal_delay = (
                _ren_hist.groupby("company_id")["renewal_delay_days"]
                .median()
                .reset_index(name="median_prior_renewal_delay_days")
            )
            features = features.merge(renewal_delay, on="company_id", how="left")
        else:
            features["median_prior_renewal_delay_days"] = np.nan
    else:
        features["median_prior_renewal_delay_days"] = np.nan

    # Scoring date column for multi-date stacks
    features["as_of_date"] = as_of_date.date().isoformat()

    # Adoption case classification (Part 2 routing input).
    # Applied row-wise using point-in-time signals; saved in the output so Part 2
    # can consume it directly without re-deriving the thresholds.
    # NaN-safe: any missing completeness or coverage → "unknown_measurement".
    import math as _math
    def _classify_row(r):
        comp  = r.get("obs_completeness_90d")
        cov   = r.get("coverage_recent_30d")
        delta = r.get("coverage_delta_30d")
        p1    = int(r.get("p1_tickets_90d", 0) or 0)
        # NaN guard
        for v in (comp, cov, delta):
            if v is None or (isinstance(v, float) and _math.isnan(v)):
                return "unknown_measurement"
        return classify_adoption_case(
            obs_completeness_90d=float(comp),
            coverage_recent_30d=float(cov),
            coverage_delta_30d=float(delta),
            p1_tickets_90d=p1,
        )
    features["adoption_case"] = features.apply(_classify_row, axis=1)

    out_cols = [
        "as_of_date", "contract_id", "renewal_lineage_id", "company_id", "product_id",
        "product_family", "offering_type", "term_start", "term_end",
        "days_to_renewal", "acv", "size_segment", "industry", "region",
        "employee_count", "channel", "tenure_days",
        "n_products_in_window", "n_families_in_window", "total_acv_in_window",
        # Ticket signals — resolution only if known at this snapshot
        "tickets_90d", "p1_tickets_90d", "avg_resolution_hrs_90d",
        "open_tickets_90d", "avg_open_age_hrs_90d",
        # Deployment engagement signals (90d lookback, point-in-time)
        "avg_coverage_90d", "coverage_trend_90d",
        "obs_completeness_90d",           # fraction of weeks with data (renamed from policy_adoption)
        "policy_events_per_unit_90d",     # true policy activity signal
        "remediation_velocity_90d", "automation_adoption_90d",
        "scan_success_rate_90d", "missing_telemetry_pct_90d",
        # 30-day coverage split — supports four behavioral cases
        "coverage_recent_30d", "coverage_prior_30d", "coverage_delta_30d",
        # Adoption routing (Part 2 input): classification based on coverage completeness
        # and trend signals. Saves the routing decision at feature-build time so Part 2
        # uses the same thresholds as the validate.py unit tests.
        "adoption_case",
        # Relationship features (point-in-time 180d)
        "executive_interactions_180d", "marketing_engagements_180d",
        "days_since_last_interaction",
        # Renewal velocity
        "median_prior_renewal_delay_days",
        # History integrity — rows with incomplete_history=True have a broken
        # renewal chain and should be excluded from analyses requiring complete
        # contract history (NRR computation, multi-period churn models).
        # predecessor_contract_id_orig preserves the original reference for audit.
        "incomplete_history", "predecessor_contract_id_orig",
        # ── Forward experience features (WP4) ──────────────────────────────────
        # 1. Deployment reach ratio over last 30d telemetry weeks
        "deployment_reach_30d",
        # 2. Onboarding delay proxy (observable from telemetry, not hidden state)
        "onboarding_delay_days",
        # 3. Change in productive use ratio: current 30d vs prior 30d
        "productive_use_delta_30d",
        # 4. Workflow success rate over last 30d
        "workflow_success_30d",
        # 5. Unresolved customer impact hours (open P1/P2 tickets, last 90d)
        "unresolved_customer_impact_hours_90d",
        # 6. Count of SLA-breached customer-impact tickets in last 90d
        "repeat_sla_breaches_90d",
        # 7. Most recent renewal quote price change % (point-in-time)
        "quoted_unit_price_change_pct",
        # 8. Budget concern interaction flag (last 180d)
        "dated_budget_concern",
        # 9. Days without a confirmed owner (since last handoff with new_owner_id)
        "days_without_confirmed_owner",
        # 10. Count of overdue (past due_date, not completed) agreed success plan actions
        "overdue_agreed_actions",
        # ── Product family breadth features ────────────────────────────────────
        # 11. Number of distinct active products in same family (higher = stickier)
        "n_active_products_in_family",
        # 12. Fraction of catalog products in this family being used (0–1)
        "family_adoption_completeness",
        # 13. Count of OTHER products in same family cancelled in last 1yr
        "family_sibling_cancellations_1yr",
        "churned", "label_matured",
    ]
    out_cols = [c for c in out_cols if c in features.columns]
    return features[out_cols]


def build_renewal_unit_features(cfg: dict, lineage: list) -> pd.DataFrame:
    """Build renewal features for the primary scoring date + all historical dates."""
    print("[integrate] Building renewal_unit_features (all scoring dates)...")

    primary_date = pd.Timestamp(cfg["scoring_date"])
    historical_dates = [pd.Timestamp(d) for d in cfg.get("historical_scoring_dates", [])]
    all_dates = historical_dates + [primary_date]

    all_frames = []
    for as_of in all_dates:
        print(f"  → as_of_date={as_of.date()}", end="")
        df = build_renewal_unit_features_for(cfg, as_of, lineage)
        all_frames.append(df)
        print(f"  {len(df):,} rows")

    features = pd.concat(all_frames, ignore_index=True)

    # Primary scoring date is the current renewal_unit_features
    primary_features = features[features["as_of_date"] == cfg["scoring_date"]]
    labeled = features[features["label_matured"] == True]

    features.to_csv(DATA_CLEAN / "renewal_unit_features.csv", index=False)
    labeled.to_csv(DATA_CLEAN / "renewal_unit_features_labeled.csv", index=False)
    primary_features.to_csv(DATA_CLEAN / "renewal_unit_features_scoring.csv", index=False)

    # Report the incomplete-history population so analysts know what to exclude.
    # These rows stay in the outputs (exclusion is a downstream decision), but
    # the exclusion impact must be visible before any model training.
    # IMPORTANT: incomplete_history is a data-cleaning artifact (broken predecessor
    # FK), NOT a customer-behavior signal — do not use it as a model feature.
    _ih_col = "incomplete_history"
    if _ih_col in features.columns:
        n_ih_all      = int((features[_ih_col] == True).sum())
        n_ih_labeled  = int((labeled[_ih_col] == True).sum()) if _ih_col in labeled.columns else 0
        n_ih_scoring  = int((primary_features[_ih_col] == True).sum()) if _ih_col in primary_features.columns else 0
        complete_pct  = round(100 * (1 - n_ih_all / max(len(features), 1)), 1)
        print(f"  incomplete_history: {n_ih_all} rows across all dates "
              f"({n_ih_labeled} labeled, {n_ih_scoring} scoring) — "
              f"{complete_pct}% of rows have complete history")
        print(f"  → analyses requiring full renewal chain: filter incomplete_history == False")
    else:
        n_ih_all = n_ih_labeled = n_ih_scoring = 0

    lineage.append({
        "dataset": "renewal_unit_features",
        "source_tables": ["order_items", "orders", "companies", "products",
                           "support_tickets", "contract_events",
                           "telemetry_weekly", "renewal_quotes", "account_interactions"],
        "scoring_dates": [d.date().isoformat() for d in all_dates],
        "label_join": "expiring_contract_id (Bug#1 fix)",
        "rows_total": len(features),
        "labeled_rows": len(labeled),
        "incomplete_history": {
            "all_dates": n_ih_all,
            "labeled": n_ih_labeled,
            "scoring": n_ih_scoring,
            "note": (
                "Rows where a predecessor contract was quarantined during cleaning. "
                "Filter incomplete_history == False for analyses requiring complete "
                "renewal chain. Do NOT use as a model feature — this is a cleaning "
                "artifact, not a behavioral signal."
            ),
        },
        "note": "One row per recurring contract × scoring date; label via expiring_contract_id",
    })

    print(f"  renewal_unit_features: {len(features):,} total rows ({len(labeled):,} labeled)")
    return features


# ── GRR / NRR helpers ─────────────────────────────────────────────────────────

def compute_grr_nrr(events: pd.DataFrame, as_of_date: pd.Timestamp,
                    lookback_months: int = 12) -> pd.DataFrame:
    """
    Compute event-based Gross Revenue Retention (event_GRR) and Net Revenue
    Retention (event_NRR) per company over the trailing lookback window.

    These are *observed-event estimates*, not full ARR reconciliations:
      - Opening ARR = contracts whose term was active at window_start
        (half-open: term_start <= ws AND term_end > ws)
      - Movements captured = explicit cancellation / renewal events fired in window
      - Opening ARR is retained unless a recorded loss event reduces it.
        Silent expirations (contracts that expired without any event) are not
        detected, so retention may be overstated.
      These metrics are not reconciled account-level GRR/NRR.

    GRR = (starting_ARR - churned_ARR - contracted_ARR) / starting_ARR
    NRR = (starting_ARR - churned_ARR - contracted_ARR + expanded_ARR) / starting_ARR

    Column names retain the grr/nrr aliases for downstream compatibility.
    """
    window_start = as_of_date - pd.DateOffset(months=lookback_months)

    # ── Opening ARR: active contracts at window_start ─────────────────────────
    # Half-open interval convention: a contract is active at window_start iff
    #   term_start <= window_start < term_end
    # A contract whose term_end == window_start had already expired; exclude it.
    ev_snap = events.copy()
    ev_snap["term_start_dt"] = pd.to_datetime(ev_snap["term_start"], errors="coerce")
    ev_snap["term_end_dt"]   = pd.to_datetime(ev_snap["term_end"],   errors="coerce")
    opening_cohort = ev_snap[
        (ev_snap["event_type"].isin(["initial_purchase", "renewal", "renewal_expansion",
                                      "renewal_contraction"])) &
        (ev_snap["net_acv"] > 0) &
        (ev_snap["term_start_dt"] <= window_start) &
        (ev_snap["term_end_dt"]   >  window_start)   # half-open: exclude expired-at-boundary
    ]
    starting = opening_cohort.groupby("company_id")["net_acv"].sum().rename("starting_arr")
    opening_company_ids = set(starting.index)   # only these companies have valid GRR/NRR

    ev = events.copy()
    ev["event_date_dt"] = pd.to_datetime(ev["event_date"], errors="coerce")
    # Restrict movements to opening cohort — new logos are excluded from GRR/NRR
    ev_window = ev[
        (ev["event_date_dt"] >= window_start) &
        (ev["event_date_dt"] <= as_of_date) &
        ev["company_id"].isin(opening_company_ids)
    ]

    cancels = ev_window[ev_window["event_type"] == "cancellation"].groupby("company_id")[
        "prior_net_acv"
    ].sum().rename("churned_arr")

    # FIX: compute dollar movements from ALL renewal events using raw ACV delta,
    # not from event_type labels (which carry a 2% dead-band from classify_renewal).
    # A $100→$101.50 renewal (1.5% change) gets event_type="renewal" but IS an
    # expansion — previously invisible to NRR.  Use prior_net_acv vs net_acv directly.
    all_renewals = ev_window[ev_window["event_type"].isin(
        ["renewal", "renewal_expansion", "renewal_contraction"]
    )].copy()
    if len(all_renewals) > 0:
        all_renewals["_delta"] = all_renewals["net_acv"] - all_renewals["prior_net_acv"].fillna(all_renewals["net_acv"])
        contractions_raw = all_renewals[all_renewals["_delta"] < 0].copy()
        expansions_raw   = all_renewals[all_renewals["_delta"] > 0].copy()
        contractions_raw["contracted_arr"] = (-contractions_raw["_delta"]).clip(lower=0)
        expansions_raw["expanded_arr"]     = expansions_raw["_delta"].clip(lower=0)
        contraction_agg = contractions_raw.groupby("company_id")["contracted_arr"].sum()
        expansion_agg   = expansions_raw.groupby("company_id")["expanded_arr"].sum()
    else:
        contraction_agg = pd.Series(dtype=float, name="contracted_arr")
        expansion_agg   = pd.Series(dtype=float, name="expanded_arr")

    # New-logo ARR in the window (reported separately, not in GRR/NRR)
    new_logos = ev[
        (ev["event_date_dt"] >= window_start) &
        (ev["event_date_dt"] <= as_of_date) &
        (ev["event_type"] == "initial_purchase") &
        ~ev["company_id"].isin(opening_company_ids)
    ]
    new_logo_arr = new_logos.groupby("company_id")["net_acv"].sum().rename("new_logo_arr")

    # Build retention table — only for opening cohort
    ret = starting.to_frame()
    ret = ret.join(cancels,         how="left")
    ret = ret.join(contraction_agg, how="left")
    ret = ret.join(expansion_agg,   how="left")
    ret = ret.join(new_logo_arr,    how="left")
    ret = ret.fillna(0).reset_index()

    # GRR/NRR only defined for companies with positive opening ARR
    s = ret["starting_arr"]
    ret["grr"] = np.where(
        s > 0,
        ((s - ret.get("churned_arr", 0) - ret.get("contracted_arr", 0)) / s).clip(0, 1).round(4),
        np.nan,
    )
    ret["nrr"] = np.where(
        s > 0,
        ((s - ret.get("churned_arr", 0) - ret.get("contracted_arr", 0)
          + ret.get("expanded_arr", 0)) / s).clip(0).round(4),
        np.nan,
    )

    return ret[["company_id", "starting_arr", "churned_arr", "grr", "nrr"]]


# ── Dataset 3: company_summary ────────────────────────────────────────────────

def build_company_summary(cfg: dict, lineage: list) -> pd.DataFrame:
    print("[integrate] Building company_summary...")

    scoring_date = pd.Timestamp(cfg["scoring_date"])
    orders   = read_clean("orders.csv")
    items    = read_clean("order_items.csv")
    companies = read_clean("companies_clean.csv")
    products = read_clean("products.csv",
                           usecols=["product_id", "product_family", "recurring_flag"])
    events   = read_clean("contract_events_clean.csv")
    tickets  = read_clean("support_tickets_clean.csv")
    deps     = read_clean("deployments_clean.csv",
                           usecols=["deployment_id", "company_id", "deployment_state", "product_id"])

    orders["order_date_dt"] = pd.to_datetime(orders["order_date"], errors="coerce")

    # Eligible completed orders
    eligible = orders[orders["in_eligible_cohort"] == True]

    # Revenue signals
    recur_items = items[items["recurring_flag"] == True].merge(
        orders[["order_id", "company_id", "order_date_dt"]], on="order_id", how="left"
    ).merge(products, on="product_id", how="left")

    # Active recurring (term_end > scoring_date)
    recur_items["term_end_dt"] = pd.to_datetime(recur_items["term_end"], errors="coerce")
    active_recur = recur_items[
        recur_items["term_end_dt"].notna() &
        (recur_items["term_end_dt"] > scoring_date) &
        (recur_items["line_net_amount"] > 0)
    ].copy()

    # ACV per active line (use net_acv if available)
    if "net_acv" in active_recur.columns and active_recur["net_acv"].notna().any():
        active_recur["acv"] = active_recur["net_acv"]
    else:
        active_recur["term_days"] = (
            active_recur["term_end_dt"] -
            pd.to_datetime(active_recur["term_start"], errors="coerce")
        ).dt.days
        active_recur["term_months"] = (active_recur["term_days"] / 30.44).round(1)
        active_recur["acv"] = np.where(
            active_recur["term_months"] > 0,
            active_recur["line_net_amount"] / active_recur["term_months"] * 12,
            active_recur["line_net_amount"]
        )

    # Aggregate to company
    co_recur = active_recur.groupby("company_id").agg(
        active_recurring_lines=("order_item_id", "count"),
        total_active_acv=("acv", "sum"),
        n_products_active=("product_id", "nunique"),
        n_families_active=("product_family", "nunique"),
    ).reset_index()

    # Order history signals
    co_orders = eligible.groupby("company_id").agg(
        total_orders=("order_id", "count"),
        total_spend=("order_net_amount", "sum"),
        avg_order_value=("order_net_amount", "mean"),
        first_order_date=("order_date_dt", "min"),
        last_order_date=("order_date_dt", "max"),
    ).reset_index()
    co_orders["tenure_days"] = (scoring_date - co_orders["first_order_date"]).dt.days

    # Support signals
    ticket_co = tickets.groupby("company_id").agg(
        total_tickets=("ticket_id", "count"),
        p1_tickets=("priority", lambda x: (x == "P1").sum()),
        avg_resolution_hrs=("resolution_hours", "mean"),
    ).reset_index()

    # Deployment signals
    dep_co = deps.groupby("company_id").agg(
        total_deployments=("deployment_id", "count"),
        active_deployments=("deployment_state", lambda x: (x == "active_paid").sum()),
        abandoned_deployments=("deployment_state", lambda x: (x == "abandoned").sum()),
    ).reset_index()

    # Churn signals: use prior_net_acv from cancellation events for lost value
    cancel_events = events[events["event_type"] == "cancellation"]
    co_churn = cancel_events.groupby("company_id").agg(
        cancellations=("event_id", "count"),
        churned_acv=("prior_net_acv", "sum"),   # FIX: prior_net_acv is the lost value
    ).reset_index()

    # Expansion events
    exp_events = events[events["event_type"] == "renewal_expansion"]
    co_exp = exp_events.groupby("company_id").agg(
        expansions=("event_id", "count"),
    ).reset_index()

    # GRR / NRR (trailing 12 months) — opening ARR from active contracts at window start
    ret_df = compute_grr_nrr(events, scoring_date, lookback_months=12)

    # Assemble
    summary = companies.merge(co_recur, on="company_id", how="left")
    summary = summary.merge(co_orders, on="company_id", how="left")
    summary = summary.merge(ticket_co, on="company_id", how="left")
    summary = summary.merge(dep_co, on="company_id", how="left")
    summary = summary.merge(co_churn, on="company_id", how="left")
    summary = summary.merge(co_exp, on="company_id", how="left")
    summary = summary.merge(ret_df, on="company_id", how="left")

    # Fill zeros for companies with no activity
    zero_cols = ["active_recurring_lines", "total_active_acv", "n_products_active",
                  "n_families_active", "total_orders", "total_spend", "total_tickets",
                  "p1_tickets", "total_deployments", "active_deployments",
                  "abandoned_deployments", "cancellations", "churned_acv", "expansions",
                  "starting_arr", "churned_arr"]
    for col in zero_cols:
        if col in summary.columns:
            summary[col] = summary[col].fillna(0)
    # GRR/NRR: preserve null for companies with zero opening ARR (undefined ratio).
    # Filling with 1.0 would falsely report 100% retention for accounts that had
    # no recurring revenue at window_start — these are excluded from cohort analysis.

    # Company health flags (no leakage — only historical events)
    summary["has_active_recurring"]   = (summary["active_recurring_lines"] > 0).astype(int)
    summary["has_abandonment"]        = (summary["abandoned_deployments"] > 0).astype(int)
    summary["has_cancellation"]       = (summary["cancellations"] > 0).astype(int)
    summary["partial_loss_flag"]      = (
        (summary["cancellations"] > 0) & (summary["active_recurring_lines"] > 0)
    ).astype(int)

    # Reformat dates
    for col in ["first_order_date", "last_order_date"]:
        if col in summary.columns:
            summary[col] = summary[col].dt.date.astype(str).where(
                summary[col].notna(), other=None
            )

    summary.to_csv(DATA_CLEAN / "company_summary.csv", index=False)
    lineage.append({
        "dataset": "company_summary",
        "source_tables": ["companies", "orders", "order_items", "products",
                           "support_tickets", "deployments", "contract_events"],
        "as_of_date": cfg["scoring_date"],
        "rows": len(summary),
        "note": "One row per company; GRR/NRR computed trailing 12m; "
                "recurring value NOT double-counted across teams",
    })

    print(f"  company_summary: {len(summary):,} rows")
    return summary


# ── Dataset 4: deployment_week_features ───────────────────────────────────────

def build_deployment_week_features(cfg: dict, lineage: list):
    """Build deployment × week features.
    FIX (Bug #4): Filter on available_at <= as_of_date (not week_start).
    """
    print("[integrate] Building deployment_week_features...")

    scoring_date  = pd.Timestamp(cfg["scoring_date"])
    lookback_180  = scoring_date - timedelta(days=180)

    deps     = read_clean("deployments_clean.csv")
    dep_team = pd.read_csv(DATA_CLEAN / "deployment_team_map_clean.csv")
    dep_team_map = dep_team.groupby("deployment_id")["team_id"].apply(list).to_dict()

    telem_path = DATA_CLEAN / "telemetry_weekly_clean.csv"
    out_path   = DATA_CLEAN / "deployment_week_features.csv"

    chunksize = 500_000
    first     = True
    total_out = 0

    for chunk in pd.read_csv(telem_path, chunksize=chunksize):
        chunk["week_start_dt"]   = pd.to_datetime(chunk["week_start"],   errors="coerce")
        chunk["available_at_dt"] = pd.to_datetime(chunk["available_at"], errors="coerce")

        # Point-in-time filter: week must be within the lookback window.
        # Rows with null available_at are MISSING observations — keep them so the
        # feature dataset can distinguish unavailable, missing, zero-use, and late.
        # Only exclude rows whose data arrived provably after the scoring date.
        chunk = chunk[
            (chunk["week_start_dt"] >= lookback_180) &
            (chunk["available_at_dt"].isna() | (chunk["available_at_dt"] <= scoring_date))
        ]

        if len(chunk) == 0:
            continue

        # Merge deployment metadata — keep telemetry's own eligible_units as the
        # authoritative denominator for this week's coverage calculation.
        # The deployment record may reflect a later renewal's capacity; substituting
        # it would inflate/deflate historical coverage ratios.
        chunk = chunk.merge(
            deps[["deployment_id", "company_id", "product_id",
                   "product_family", "deployment_state"]],
            on="deployment_id", how="left"
        )

        # Coverage ratio: active / eligible_units (from telemetry row, not deployment)
        chunk["coverage_ratio"] = np.where(
            chunk["telemetry_status"] == "missing",
            np.nan,
            np.where(
                chunk["eligible_units"].fillna(0) > 0,
                (chunk["active_units"] / chunk["eligible_units"]).clip(0, 1),
                np.nan
            )
        )
        chunk["is_missing_telemetry"] = (chunk["telemetry_status"] == "missing").astype(int)
        chunk["is_zero_usage"] = (
            (chunk["active_units"] == 0) & (~chunk["is_missing_telemetry"].astype(bool))
        ).astype(int)
        chunk["is_late"] = chunk["is_late"].astype(int)

        chunk = chunk.drop(columns=["week_start_dt", "available_at_dt"], errors="ignore")
        chunk.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        first = False
        total_out += len(chunk)
        print(f"  [dep_week_features] {total_out:,} rows written...", end="\r")

    print()
    lineage.append({
        "dataset": "deployment_week_features",
        "source_tables": ["telemetry_weekly", "deployments", "deployment_team_map"],
        "as_of_date": cfg["scoring_date"],
        "point_in_time_filter": "available_at <= scoring_date (Bug #4 fix)",
        "lookback_days": 180,
        "rows": total_out,
        "note": "One row per deployment × week; missing telemetry preserved as null, not zero",
    })

    print(f"  deployment_week_features: {total_out:,} rows")


# ── Main ──────────────────────────────────────────────────────────────────────

def _acquire_stage_lock(directory: Path) -> object:
    """
    Acquire an exclusive lock (fcntl.LOCK_EX) on the pipeline lock file.
    Using LOCK_EX (not LOCK_SH) ensures that only one pipeline stage runs at a
    time: generation, cleaning, and integration are all mutually exclusive.
    Without this, a cleaner could overwrite data/cleaned/ while integration reads
    those same files.  Fails immediately (LOCK_NB) if any other stage is active.
    """
    lock_path = directory / "._generating.lock"
    lock_fh = open(lock_path, "a")  # 'a' creates file if absent; does not truncate
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_fh.close()
        raise RuntimeError(
            "[stage-lock] Cannot acquire exclusive lock — another pipeline stage is active.\n"
            "Wait for it to finish, then re-run integrate.py."
        )
    return lock_fh


def _verify_pipeline_coherence():
    """
    Hard gate: both generation_manifest.json and quality_report.json must exist
    and carry the same run_id.  Missing metadata is an error, not a skip — a
    pipeline without traceable lineage cannot be trusted for modeling.

    Also validates that the generation manifest still has file_hashes (confirming
    generate_data.py was run with the version that writes them).
    """
    manifest_path = OUTPUTS / "generation_manifest.json"
    report_path   = OUTPUTS / "quality_report.json"

    # Both files are mandatory
    missing = [str(p) for p in [manifest_path, report_path] if not p.exists()]
    if missing:
        raise RuntimeError(
            "[coherence] Required metadata files not found:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nRun generate_data.py then ingest_clean.py before integrate.py."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(report_path) as f:
        report = json.load(f)

    gen_id = manifest.get("run_id")
    cln_id = report.get("run_id")

    errors = []
    if not gen_id:
        errors.append("  generation_manifest.json missing run_id — re-run generate_data.py")
    if not cln_id:
        errors.append("  quality_report.json missing run_id — re-run ingest_clean.py")
    if gen_id and cln_id and gen_id != cln_id:
        errors.append(f"  run_id mismatch: generation={gen_id}  cleaning={cln_id}")

    gen_cos = manifest.get("company_count")
    cln_cos = report.get("core_table_integrity", {}).get("companies", {}).get("clean_rows")
    if gen_cos and cln_cos and gen_cos != cln_cos:
        errors.append(f"  company_count mismatch: generation={gen_cos:,}  cleaning={cln_cos:,}")

    if not manifest.get("file_hashes"):
        errors.append("  generation_manifest.json has no file_hashes — re-run generate_data.py")

    if errors:
        raise RuntimeError(
            "[coherence] Pipeline metadata is inconsistent.\n"
            "Run generate_data.py then ingest_clean.py with no intervening runs.\n"
            + "\n".join(errors)
        )

    print(f"  [coherence] pipeline metadata verified "
          f"(run_id={gen_id}  companies={gen_cos:,}  "
          f"hashed_files={len(manifest.get('file_hashes', {}))})")


def main():
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Acquire shared lock BEFORE coherence check — holds throughout all reads
    # so a new generation run cannot overwrite cleaned files mid-pipeline.
    _stage_lock_fh = _acquire_stage_lock(DATA_RAW)
    _verify_pipeline_coherence()

    cfg = load_config()
    t0  = time.time()
    lineage = []

    fact_df    = build_order_item_fact(lineage)
    # Deployment week features must be built before renewal features so that
    # engagement signals (coverage, remediation, automation) can be joined in.
    build_deployment_week_features(cfg, lineage)
    renewal_df = build_renewal_unit_features(cfg, lineage)
    summary_df = build_company_summary(cfg, lineage)

    elapsed = round(time.time() - t0, 1)

    lineage_report = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "scoring_date":  cfg["scoring_date"],
        "runtime_seconds": elapsed,
        "joins": lineage,
    }
    with open(OUTPUTS / "integration_lineage.json", "w") as f:
        json.dump(lineage_report, f, indent=2, default=_json_default)

    # Labeled churn check
    labeled = renewal_df[renewal_df["label_matured"] == True]
    n_labeled_churn = int((labeled["churned"] == 1).sum())
    print(f"\n  Churn label check: {n_labeled_churn:,} labeled churns in {len(labeled):,} labeled rows "
          f"({100*n_labeled_churn/max(len(labeled),1):.1f}%)")
    if n_labeled_churn == 0:
        print("  [WARN] Zero labeled churn events — verify expiring_contract_id linking")

    print(f"\n✓ Integration complete in {elapsed}s")
    print(f"  order_item_fact:          {len(fact_df):,} rows")
    print(f"  renewal_unit_features:    {len(renewal_df):,} rows")
    print(f"  company_summary:          {len(summary_df):,} rows")
    print(f"  Lineage report: outputs/integration_lineage.json")


if __name__ == "__main__":
    main()

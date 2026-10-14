"""
ingest_clean.py
===============
Ingests the four core raw CSVs (users, orders, order_items, products) and
supplementary tables, validates them, cleans them, and writes:

  data/cleaned/{users,orders,order_items,products}.csv   (four required outputs)
  data/cleaned/companies_clean.csv
  data/cleaned/teams_clean.csv
  data/cleaned/deployments_clean.csv
  data/cleaned/deployment_team_map_clean.csv
  data/cleaned/telemetry_weekly_clean.csv   (streamed, not loaded all at once)
  data/cleaned/support_tickets_clean.csv
  data/cleaned/contract_events_clean.csv
  outputs/quality_report.json               (Step 5 quality evidence)

Run:
    python3 src/ingest_clean.py

All raw files are preserved unchanged.
"""

import fcntl
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "cleaned"
OUTPUTS = ROOT / "outputs"
EVAL_DIR = ROOT / "eval"


def load_config() -> dict:
    with open(ROOT / "config" / "simulation.yaml") as f:
        return yaml.safe_load(f)


# ── Quality report accumulator ────────────────────────────────────────────────

class QualityReport:
    def __init__(self):
        self.tables = {}
        self.commercial = {}
        self.telemetry = {}
        self.cleaning_accuracy = {}

    def record_table(self, name: str, raw_rows: int, clean_rows: int,
                     dupes_removed: int, fk_unresolved: int,
                     quarantined: dict):
        self.tables[name] = {
            "raw_rows": raw_rows,
            "clean_rows": clean_rows,
            "dupes_removed": dupes_removed,
            "fk_unresolved": fk_unresolved,
            "quarantined_by_reason": quarantined,
        }

    def to_dict(self) -> dict:
        return {
            "run_timestamp": datetime.utcnow().isoformat(),
            "core_table_integrity": self.tables,
            "commercial_consistency": self.commercial,
            "telemetry_consistency": self.telemetry,
            "cleaning_accuracy": self.cleaning_accuracy,
        }


qr = QualityReport()


# ── Generic helpers ───────────────────────────────────────────────────────────

def safe_parse_date(series: pd.Series, col_name: str, quarantine: dict) -> pd.Series:
    """Parse dates; quarantine unparseable values; return datetime series."""
    parsed = pd.to_datetime(series, errors="coerce")
    bad_mask = parsed.isna() & series.notna() & (series != "")
    n_bad = bad_mask.sum()
    if n_bad > 0:
        quarantine[f"invalid_date_{col_name}"] = int(n_bad)
    return parsed


def normalize_id(series: pd.Series) -> pd.Series:
    """Uppercase and strip product/entity IDs for consistent matching."""
    return series.astype(str).str.strip().str.upper()


def check_pk_unique(df: pd.DataFrame, pk_col: str, table_name: str) -> int:
    n_dupes = df.duplicated(subset=[pk_col]).sum()
    if n_dupes > 0:
        print(f"  [WARN] {table_name}: {n_dupes} duplicate {pk_col} values")
    return int(n_dupes)


# ── Products ──────────────────────────────────────────────────────────────────

def clean_products(cfg: dict) -> pd.DataFrame:
    print("[products] Cleaning...")
    raw = pd.read_csv(DATA_RAW / "products.csv")
    raw_rows = len(raw)
    quarantine = {}

    # Normalize
    df = raw.copy()
    df["product_id"] = normalize_id(df["product_id"])
    df["product_family"] = df["product_family"].str.strip()
    df["offering_type"] = df["offering_type"].str.strip()

    # Validate required fields
    required = ["product_id", "product_name", "product_family", "offering_type",
                "billing_unit", "recurring_flag", "annual_catalog_price"]
    for col in required:
        n_null = df[col].isna().sum()
        if n_null > 0:
            quarantine[f"null_{col}"] = int(n_null)
            df = df[df[col].notna()]

    # Price must be non-negative (support tiers use pct, so 0 is valid for some)
    invalid_price = df["annual_catalog_price"] < 0
    if invalid_price.sum() > 0:
        quarantine["invalid_price"] = int(invalid_price.sum())
        df = df[~invalid_price]

    dupes = check_pk_unique(df, "product_id", "products")
    df = df.drop_duplicates(subset=["product_id"])

    qr.record_table("products", raw_rows, len(df), dupes, 0, quarantine)
    df.to_csv(DATA_CLEAN / "products.csv", index=False)
    print(f"  raw={raw_rows:,}  clean={len(df):,}  dupes_removed={dupes}")
    return df


# ── Companies ─────────────────────────────────────────────────────────────────

def clean_companies() -> pd.DataFrame:
    print("[companies] Cleaning...")
    raw = pd.read_csv(DATA_RAW / "companies.csv")
    raw_rows = len(raw)
    quarantine = {}

    df = raw.copy()
    df["company_id"] = df["company_id"].str.strip()

    null_mask = df["company_id"].isna()
    if null_mask.sum() > 0:
        quarantine["null_company_id"] = int(null_mask.sum())
        df = df[~null_mask]

    dupes = check_pk_unique(df, "company_id", "companies")
    df = df.drop_duplicates(subset=["company_id"])

    qr.record_table("companies", raw_rows, len(df), dupes, 0, quarantine)
    df.to_csv(DATA_CLEAN / "companies_clean.csv", index=False)
    print(f"  raw={raw_rows:,}  clean={len(df):,}")
    return df


# ── Teams ─────────────────────────────────────────────────────────────────────

def clean_teams(companies_df: pd.DataFrame) -> pd.DataFrame:
    print("[teams] Cleaning...")
    raw = pd.read_csv(DATA_RAW / "teams.csv")
    raw_rows = len(raw)
    quarantine = {}

    df = raw.copy()
    df["team_id"] = df["team_id"].str.strip()

    # FK check: company_id must exist
    valid_cos = set(companies_df["company_id"])
    fk_bad = ~df["company_id"].isin(valid_cos)
    n_fk = fk_bad.sum()
    if n_fk > 0:
        quarantine["unresolved_company_fk"] = int(n_fk)
        df = df[~fk_bad]

    dupes = check_pk_unique(df, "team_id", "teams")
    df = df.drop_duplicates(subset=["team_id"])

    qr.record_table("teams", raw_rows, len(df), dupes, n_fk, quarantine)
    df.to_csv(DATA_CLEAN / "teams_clean.csv", index=False)
    print(f"  raw={raw_rows:,}  clean={len(df):,}  fk_bad={n_fk}")
    return df


# ── Users ─────────────────────────────────────────────────────────────────────

def clean_users(companies_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    print("[users] Cleaning...")
    raw = pd.read_csv(DATA_RAW / "users.csv")
    raw_rows = len(raw)
    quarantine = {}

    df = raw.copy()
    df["user_id"] = df["user_id"].str.strip()

    # FK: company_id
    valid_cos = set(companies_df["company_id"])
    fk_co = ~df["company_id"].isin(valid_cos)
    n_fk_co = fk_co.sum()
    if n_fk_co > 0:
        quarantine["unresolved_company_fk"] = int(n_fk_co)

    # Parse dates
    for col in ["created_at", "active_from", "active_to"]:
        df[col] = safe_parse_date(df[col], col, quarantine)

    # active_to must be after active_from if present
    invalid_active = df["active_to"].notna() & (df["active_to"] <= df["active_from"])
    if invalid_active.sum() > 0:
        quarantine["active_to_before_active_from"] = int(invalid_active.sum())
        df.loc[invalid_active, "active_to"] = None

    # Remove truly broken rows (null user_id or unresolvable company)
    bad_mask = df["user_id"].isna() | fk_co
    df = df[~bad_mask]

    dupes = check_pk_unique(df, "user_id", "users")
    df = df.drop_duplicates(subset=["user_id"])

    # Reformat dates back to ISO strings
    for col in ["created_at", "active_from", "active_to"]:
        df[col] = df[col].dt.date.astype(str).replace("NaT", None)
        df[col] = df[col].where(df[col] != "None", other=None)

    qr.record_table("users", raw_rows, len(df), dupes, n_fk_co, quarantine)
    df.to_csv(DATA_CLEAN / "users.csv", index=False)
    print(f"  raw={raw_rows:,}  clean={len(df):,}  dupes={dupes}  fk_bad={n_fk_co}")
    return df


# ── Orders ────────────────────────────────────────────────────────────────────

VALID_ORDER_TYPES = {"new_logo", "renewal", "renewal_expansion", "renewal_contraction",
                      "cancellation", "expansion", "trial", "refund"}
VALID_ORDER_STATUSES = {"completed", "pending", "cancelled", "refunded", "partial"}

def dedup_orders_by_pk(df: pd.DataFrame) -> tuple:
    """
    Deduplicate an orders DataFrame on order_id, preferring rows with a parseable
    order_date over rows with an unparseable one.

    Returns:
        (deduped_df, duplicate_rows, n_removed, removed_order_ids)

    Design:
      - Primary identity: order_id is the stable transaction key.  Two rows with the
        same order_id are duplicates regardless of content.
      - Quality-aware sort: rows with parseable order_date sort before unparseable rows
        so the valid-date copy is always kept when one exists.  Without this, a
        bad-date duplicate appearing first in the file would eliminate a recoverable order.
      - Secondary content-hash dedup is intentionally avoided: it would risk dropping
        valid same-amount multi-product orders from the same customer.
    """
    df = df.copy()
    df["_dp"] = pd.to_datetime(df["order_date"], errors="coerce").notna().astype(int)
    df_sorted = df.sort_values(["order_id", "_dp"], ascending=[True, False]).drop(columns=["_dp"])
    dup_mask = df_sorted.duplicated(subset=["order_id"], keep="first")
    removed_ids: set = set(df_sorted.loc[dup_mask, "order_id"].astype(str))
    dup_rows = df_sorted[dup_mask].copy()
    deduped = df_sorted[~dup_mask].copy()
    return deduped, dup_rows, int(dup_mask.sum()), removed_ids


def clean_orders(users_df: pd.DataFrame, companies_df: pd.DataFrame) -> pd.DataFrame:
    print("[orders] Cleaning...")
    raw = pd.read_csv(DATA_RAW / "orders.csv")
    raw_rows = len(raw)
    quarantine = {}
    quarantine_rows = []  # accumulate rows for quarantine export

    df = raw.copy()
    df["order_id"] = df["order_id"].str.strip()

    # ── Dedup via transaction identity ───────────────────────────────────────
    df, dup_rows, n_dup_removed, dedup_removed_order_ids = dedup_orders_by_pk(df)
    dup_rows["quarantine_reason"] = "duplicate_order_id"
    quarantine_rows.append(dup_rows)
    quarantine["duplicate_records_removed"] = n_dup_removed
    # dedup_removed_order_ids: used by evaluate_cleaning_accuracy so that removals
    # by invalid_date or FK cleaning of the same order_id are not double-counted.

    # Parse dates
    df["order_date_raw"] = df["order_date"].copy()  # preserve raw
    df["order_date"] = safe_parse_date(df["order_date"], "order_date", quarantine)

    # Quarantine invalid dates
    invalid_date_mask = df["order_date"].isna() & df["order_date_raw"].notna()
    n_invalid_date = int(invalid_date_mask.sum())
    quarantine["invalid_date_quarantined"] = n_invalid_date
    if n_invalid_date > 0:
        bad_date_rows = df[invalid_date_mask].copy()
        bad_date_rows["quarantine_reason"] = "invalid_date"
        quarantine_rows.append(bad_date_rows)
    df = df[~invalid_date_mask]
    df = df.drop(columns=["order_date_raw"])

    # FK: user_id — orders with unknown buyers are invalid and must be quarantined
    valid_users = set(users_df["user_id"])
    fk_user = ~df["user_id"].isin(valid_users)
    n_fk_user = int(fk_user.sum())
    if n_fk_user > 0:
        quarantine["unresolved_user_fk"] = n_fk_user
        bad_user_rows = df[fk_user].copy()
        bad_user_rows["quarantine_reason"] = "unknown_user_fk"
        quarantine_rows.append(bad_user_rows)
        df = df[~fk_user]

    # FK: company_id (buyer company must exist)
    valid_cos = set(companies_df["company_id"])
    fk_co = ~df["company_id"].isin(valid_cos)
    n_fk_co = int(fk_co.sum())
    if n_fk_co > 0:
        quarantine["unresolved_company_fk"] = n_fk_co

    # Validate order_type
    unknown_type = ~df["order_type"].isin(VALID_ORDER_TYPES)
    if unknown_type.sum() > 0:
        quarantine["unknown_order_type"] = int(unknown_type.sum())

    # Negative amounts are invalid (refunds use order_type=refund, not negative amount)
    neg_amount = df["order_net_amount"] < 0
    if neg_amount.sum() > 0:
        quarantine["negative_order_amount"] = int(neg_amount.sum())
        neg_rows = df[neg_amount].copy()
        neg_rows["quarantine_reason"] = "negative_order_amount"
        quarantine_rows.append(neg_rows)
        df = df[~neg_amount]

    # Remove FK-broken rows
    bad_mask = df["order_id"].isna() | fk_co
    if bad_mask.sum() > 0:
        bad_fk_rows = df[bad_mask].copy()
        bad_fk_rows["quarantine_reason"] = "unresolved_fk"
        quarantine_rows.append(bad_fk_rows)
    df_clean = df[~bad_mask]

    # Only completed/partial orders in eligible sales cohort
    eligible_mask = df_clean["order_status"].isin({"completed", "partial"})
    n_ineligible = int((~eligible_mask).sum())
    if n_ineligible > 0:
        quarantine["ineligible_status_excluded_from_sales_cohort"] = n_ineligible

    df_clean["order_date"] = df_clean["order_date"].dt.date.astype(str)

    dupes = check_pk_unique(df_clean, "order_id", "orders")
    df_clean = df_clean.drop_duplicates(subset=["order_id"])

    total_fk = n_fk_user + n_fk_co
    qr.record_table("orders", raw_rows, len(df_clean), n_dup_removed, total_fk, quarantine)

    # Export quarantine CSV (all rejected/flagged rows)
    non_empty = [r for r in quarantine_rows if len(r) > 0]
    if non_empty:
        all_quarantine = pd.concat(non_empty, ignore_index=True)
        all_quarantine.to_csv(DATA_CLEAN / "quarantine_orders.csv", index=False)
        print(f"  [quarantine] {len(all_quarantine):,} rows written to quarantine_orders.csv")

    # Export both full and eligible-cohort flags
    # Eligible = completed/partial status AND positive net amount (excludes zero-dollar cancellations)
    df_clean["in_eligible_cohort"] = (
        df_clean["order_status"].isin({"completed", "partial"}) &
        (df_clean["order_net_amount"] > 0)
    )
    df_clean.to_csv(DATA_CLEAN / "orders.csv", index=False)
    print(f"  raw={raw_rows:,}  clean={len(df_clean):,}  dupes_removed={n_dup_removed}  fk_bad={total_fk}")
    # Return the clean DataFrame AND the set of order_ids removed by the dedup
    # step specifically.  The caller passes this to evaluate_cleaning_accuracy so
    # that removals from other cleaning stages (invalid_date, FK) are not counted
    # as duplicate detections.
    return df_clean, dedup_removed_order_ids


# ── Order items ───────────────────────────────────────────────────────────────

def clean_order_items(orders_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    print("[order_items] Cleaning...")
    raw = pd.read_csv(DATA_RAW / "order_items.csv")
    raw_rows = len(raw)
    quarantine = {}

    df = raw.copy()
    df["order_item_id"] = df["order_item_id"].str.strip()

    # Normalize product_id casing (injected dirty data: lowercased)
    df["product_id_raw"] = df["product_id"].copy()
    df["product_id"] = normalize_id(df["product_id"])

    n_case_fixed = (df["product_id"] != df["product_id_raw"].str.upper().str.strip()).sum()
    if n_case_fixed > 0:
        quarantine["product_id_case_normalized"] = int(n_case_fixed)
    df = df.drop(columns=["product_id_raw"])

    # FK: order_id must exist in clean orders
    valid_orders = set(orders_df["order_id"])
    fk_order = ~df["order_id"].isin(valid_orders)
    n_fk_order = fk_order.sum()
    if n_fk_order > 0:
        quarantine["unresolved_order_fk"] = int(n_fk_order)

    # FK: product_id
    valid_prods = set(products_df["product_id"])
    fk_prod = ~df["product_id"].isin(valid_prods)
    n_fk_prod = fk_prod.sum()
    if n_fk_prod > 0:
        quarantine["unresolved_product_fk"] = int(n_fk_prod)

    # Parse term dates
    for col in ["term_start", "term_end"]:
        df[col] = safe_parse_date(df[col], col, quarantine)

    # Validate: term_end > term_start (for recurring)
    recur_mask = df["recurring_flag"] == True
    invalid_term = recur_mask & df["term_end"].notna() & (df["term_end"] <= df["term_start"])
    if invalid_term.sum() > 0:
        quarantine["invalid_term_dates"] = int(invalid_term.sum())

    # Validate: line_net_amount = qty * full_term_unit_price - discount (within $1 tolerance)
    # TCV formula: gross_tcv = qty * full_term_unit_price; line_net = gross_tcv - discount
    df["_expected_net"] = (df["quantity"] * df["full_term_unit_price"] - df["line_discount_amount"]).round(2)
    recon_err = (df["line_net_amount"] - df["_expected_net"]).abs() > 1.0
    n_recon_err = int(recon_err.sum())
    if n_recon_err > 0:
        quarantine["line_reconciliation_error"] = n_recon_err
        print(f"  [FATAL] {n_recon_err} line reconciliation errors (TCV formula mismatch)")
        sys.exit(f"[FATAL] {n_recon_err} unexplained line reconciliation errors — "
                 "investigate compute_line_financials() before proceeding")
    df = df.drop(columns=["_expected_net"])

    # Missing team mapping — already NULL from dirty injection; flag it
    n_missing_team = df["owner_team_id"].isna().sum()
    quarantine["missing_team_mapping"] = int(n_missing_team)
    # Keep rows but flag (team mapping may be legitimately absent for some order types)

    # Remove FK-broken rows (order must exist; product must exist)
    bad_mask = fk_order | fk_prod
    df_clean = df[~bad_mask]

    # Reformat dates
    for col in ["term_start", "term_end"]:
        df_clean[col] = df_clean[col].dt.date.astype(str).replace("NaT", None)
        df_clean[col] = df_clean[col].where(df_clean[col] != "None", other=None)

    dupes = check_pk_unique(df_clean, "order_item_id", "order_items")
    df_clean = df_clean.drop_duplicates(subset=["order_item_id"]).copy()

    # Predecessor FK: predecessor_contract_id must reference a surviving contract.
    # Quarantined items had valid contract_ids that didn't survive cleaning.
    # Policy: null out broken predecessor references and mark those lineages as
    # incomplete_history=True so renewal-continuity analyses can exclude them.
    valid_contract_ids = set(df_clean["contract_id"].dropna())
    has_pred = df_clean["predecessor_contract_id"].notna()
    broken_pred = has_pred & ~df_clean["predecessor_contract_id"].isin(valid_contract_ids)
    n_broken_pred = int(broken_pred.sum())
    df_clean["incomplete_history"] = False
    # Audit column: captures the original value before it is nulled.
    # Always present so downstream schemas are stable regardless of run.
    df_clean["predecessor_contract_id_orig"] = None
    if n_broken_pred > 0:
        broken_lineages = set(df_clean.loc[broken_pred, "renewal_lineage_id"].dropna())
        # Flag every item in affected lineages, not just the broken link
        df_clean.loc[
            df_clean["renewal_lineage_id"].isin(broken_lineages), "incomplete_history"
        ] = True
        # Preserve the original value for the affected rows before nulling
        df_clean.loc[broken_pred, "predecessor_contract_id_orig"] = (
            df_clean.loc[broken_pred, "predecessor_contract_id"]
        )
        # Null the dangling predecessor reference so downstream joins don't silently fail
        df_clean.loc[broken_pred, "predecessor_contract_id"] = None
        quarantine["broken_predecessor_fk_nulled"] = n_broken_pred
        quarantine["incomplete_history_lineages"] = len(broken_lineages)
        print(f"  [predecessor FK] {n_broken_pred} broken refs nulled across "
              f"{len(broken_lineages)} lineages → incomplete_history=True")

    total_fk = n_fk_order + n_fk_prod
    qr.record_table("order_items", raw_rows, len(df_clean), dupes, total_fk, quarantine)

    # Commercial reconciliation
    # Order header amount vs sum of lines
    line_totals = df_clean.groupby("order_id")["line_net_amount"].sum().reset_index()
    line_totals.columns = ["order_id", "lines_total"]
    merged = orders_df[["order_id", "order_net_amount"]].merge(line_totals, on="order_id", how="left")
    merged["recon_diff"] = (merged["order_net_amount"] - merged["lines_total"].fillna(0)).abs()
    n_header_mismatch = (merged["recon_diff"] > 1.0).sum()
    qr.commercial["order_header_line_mismatch"] = int(n_header_mismatch)
    qr.commercial["orders_with_no_lines"] = int(merged["lines_total"].isna().sum())

    # Overlapping entitlement terms (same company + product + overlapping dates)
    # Simplified check: count renewals with term overlap
    recur_clean = df_clean[df_clean["recurring_flag"] == True].copy()
    qr.commercial["recurring_line_count"] = int(len(recur_clean))
    qr.commercial["unmapped_renewals"] = int(recur_clean["renewal_lineage_id"].isna().sum())

    df_clean.to_csv(DATA_CLEAN / "order_items.csv", index=False)
    print(f"  raw={raw_rows:,}  clean={len(df_clean):,}  dupes={dupes}  fk_bad={total_fk}")
    print(f"  case_normalized={n_case_fixed}  recon_errors={n_recon_err}  missing_team={n_missing_team}")
    return df_clean


# ── Deployments ───────────────────────────────────────────────────────────────

def clean_deployments(companies_df: pd.DataFrame) -> pd.DataFrame:
    print("[deployments] Cleaning...")
    raw = pd.read_csv(DATA_RAW / "deployments.csv")
    raw_rows = len(raw)
    quarantine = {}

    df = raw.copy()
    valid_cos = set(companies_df["company_id"])
    fk_co = ~df["company_id"].isin(valid_cos)
    n_fk = fk_co.sum()
    if n_fk > 0:
        quarantine["unresolved_company_fk"] = int(n_fk)

    for col in ["deployment_start", "deployment_end"]:
        df[col] = safe_parse_date(df[col], col, quarantine)

    df_clean = df[~fk_co].drop_duplicates(subset=["deployment_id"]).copy()
    for col in ["deployment_start", "deployment_end"]:
        df_clean[col] = df_clean[col].dt.date.astype(str).replace("NaT", None)
        df_clean[col] = df_clean[col].where(df_clean[col] != "None", other=None)

    qr.record_table("deployments", raw_rows, len(df_clean), 0, n_fk, quarantine)
    df_clean.to_csv(DATA_CLEAN / "deployments_clean.csv", index=False)
    print(f"  raw={raw_rows:,}  clean={len(df_clean):,}")
    return df_clean


def clean_deployment_team_map(deps_df: pd.DataFrame) -> pd.DataFrame:
    print("[deployment_team_map] Cleaning...")
    raw = pd.read_csv(DATA_RAW / "deployment_team_map.csv")
    valid_deps = set(deps_df["deployment_id"])
    df = raw[raw["deployment_id"].isin(valid_deps)].copy()
    df.to_csv(DATA_CLEAN / "deployment_team_map_clean.csv", index=False)
    print(f"  rows={len(df):,}")
    return df


# ── Telemetry (streamed) ──────────────────────────────────────────────────────

def clean_telemetry(deps_df: pd.DataFrame):
    print("[telemetry] Cleaning (chunked)...")
    raw_path = DATA_RAW / "telemetry_weekly.csv"
    clean_path = DATA_CLEAN / "telemetry_weekly_clean.csv"
    valid_deps = set(deps_df["deployment_id"])

    chunksize = 500_000
    total_raw = 0
    total_clean = 0
    missing_count = 0
    zero_count = 0
    late_count = 0
    first = True

    for chunk in pd.read_csv(raw_path, chunksize=chunksize):
        total_raw += len(chunk)

        # FK filter
        df = chunk[chunk["deployment_id"].isin(valid_deps)].copy()

        # Classify telemetry status
        # missing = telemetry_status == 'missing' (null active_units)
        # zero = active_units == 0 but status == 'received'
        missing_mask = df["telemetry_status"] == "missing"
        zero_mask = (df["active_units"] == 0) & (df["telemetry_status"] == "received")
        missing_count += missing_mask.sum()
        zero_count += zero_mask.sum()

        # Flag late arrivals (available_at > week_start + 7 days)
        df["week_start_dt"] = pd.to_datetime(df["week_start"], errors="coerce")
        df["available_at_dt"] = pd.to_datetime(df["available_at"], errors="coerce")
        df["is_late"] = (df["available_at_dt"] - df["week_start_dt"]).dt.days > 7
        late_count += df["is_late"].sum()
        df = df.drop(columns=["week_start_dt", "available_at_dt"])

        total_clean += len(df)
        df.to_csv(clean_path, mode="w" if first else "a", header=first, index=False)
        first = False
        print(f"  [telemetry] {total_clean:,} rows cleaned...", end="\r")

    print()
    qr.telemetry["raw_rows"] = total_raw
    qr.telemetry["clean_rows"] = total_clean
    qr.telemetry["missing_telemetry_rows"] = int(missing_count)
    qr.telemetry["true_zero_usage_rows"] = int(zero_count)
    qr.telemetry["late_arrival_rows"] = int(late_count)
    qr.telemetry["missing_rate_pct"] = round(100 * missing_count / max(total_raw, 1), 2)
    print(f"  raw={total_raw:,}  clean={total_clean:,}  missing={missing_count:,}  late={late_count:,}")


# ── Support tickets ───────────────────────────────────────────────────────────

def clean_support_tickets(companies_df: pd.DataFrame, users_df: pd.DataFrame,
                           items_df: pd.DataFrame = None) -> pd.DataFrame:
    print("[support_tickets] Cleaning...")
    raw = pd.read_csv(DATA_RAW / "support_tickets.csv")
    raw_rows = len(raw)
    quarantine = {}

    df = raw.copy()
    valid_cos = set(companies_df["company_id"])
    valid_users = set(users_df["user_id"])
    fk_co = ~df["company_id"].isin(valid_cos)
    fk_user = ~df["user_id"].isin(valid_users)
    n_fk = fk_co.sum() + fk_user.sum()
    if n_fk > 0:
        quarantine["unresolved_fk"] = int(n_fk)

    df_clean = df[~fk_co & ~fk_user].drop_duplicates(subset=["ticket_id"]).copy()

    # Handle new extended columns — fill with safe defaults if absent (backward compat)
    if "deployment_id" not in df_clean.columns:
        df_clean["deployment_id"] = None
    if "contract_id" not in df_clean.columns:
        df_clean["contract_id"] = None
    if "customer_impact" not in df_clean.columns:
        df_clean["customer_impact"] = False
    else:
        df_clean["customer_impact"] = df_clean["customer_impact"].fillna(False).astype(bool)
    if "impact_hours_open" not in df_clean.columns:
        df_clean["impact_hours_open"] = 0.0
    else:
        df_clean["impact_hours_open"] = df_clean["impact_hours_open"].fillna(0.0)
    if "sla_target_hours" not in df_clean.columns:
        df_clean["sla_target_hours"] = None
    if "sla_breached" not in df_clean.columns:
        df_clean["sla_breached"] = False
    else:
        df_clean["sla_breached"] = df_clean["sla_breached"].fillna(False).astype(bool)
    # impact_start_at and impact_end_at: keep as-is (string timestamps or None)
    if "impact_start_at" not in df_clean.columns:
        df_clean["impact_start_at"] = None
    if "impact_end_at" not in df_clean.columns:
        df_clean["impact_end_at"] = None

    # Contract-link validation: check each populated contract_id against cleaned
    # order items.  Do NOT drop the ticket — preserve it for company-level analysis.
    # Flag unresolved references with a boolean column so contract-specific
    # attribution can exclude them without losing the ticket entirely.
    #
    # Distinction:
    #   contract_id IS NULL        → account-level / billing / relationship ticket (fine)
    #   contract_id IS NOT NULL
    #     AND resolves in items_df → product-specific ticket; attribution is valid
    #     AND missing from items_df → contract was removed during cleaning;
    #                                 NULL out the contract_id and set flag=True
    if "contract_id" in df_clean.columns and items_df is not None and len(items_df) > 0:
        _valid_cids = set(items_df["contract_id"].dropna().astype(str))
        _has_cid    = df_clean["contract_id"].notna()
        _bad_cid    = _has_cid & ~df_clean["contract_id"].astype(str).isin(_valid_cids)
        n_unresolved = int(_bad_cid.sum())
        if n_unresolved > 0:
            df_clean.loc[_bad_cid, "contract_id"] = None
            if n_unresolved > 0:
                quarantine["unresolved_contract_links"] = n_unresolved
    else:
        n_unresolved = 0

    df_clean.to_csv(DATA_CLEAN / "support_tickets_clean.csv", index=False)
    qr.record_table("support_tickets", raw_rows, len(df_clean), 0, n_fk, quarantine)
    print(f"  raw={raw_rows:,}  clean={len(df_clean):,}  unresolved_contract_links={n_unresolved}")
    return df_clean


# ── Contract events ───────────────────────────────────────────────────────────

def clean_entitlement_map(items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the entitlement_deployment_map by dropping rows whose contract_id
    was quarantined during order_items cleaning.

    build_entitlement_renewal_map() validates against raw items_df, but
    ingest_clean.py may quarantine some of those items (FK violations, invalid
    dates).  Rows in the map that reference quarantined contracts would break FK
    integrity for any downstream consumer.  We re-validate against the *cleaned*
    items and quarantine affected rows.
    """
    raw_path = DATA_RAW / "entitlement_deployment_map.csv"
    if not raw_path.exists():
        return pd.DataFrame()

    print("[entitlement_deployment_map] Cleaning...")
    raw = pd.read_csv(raw_path)
    raw_rows = len(raw)

    valid_contract_ids = set(items_df["contract_id"].dropna())
    fk_bad = ~raw["contract_id"].isin(valid_contract_ids)
    n_fk_bad = int(fk_bad.sum())

    df_clean = raw[~fk_bad].copy()
    df_clean.to_csv(DATA_CLEAN / "entitlement_deployment_map_clean.csv", index=False)

    quarantine = {}
    if n_fk_bad > 0:
        quarantine["quarantined_contract_fk"] = n_fk_bad
    qr.record_table("entitlement_deployment_map", raw_rows, len(df_clean), 0, n_fk_bad, quarantine)
    print(f"  raw={raw_rows:,}  clean={len(df_clean):,}  fk_bad={n_fk_bad}")
    return df_clean


def _apply_interaction_cleaning(raw: pd.DataFrame, companies_df: pd.DataFrame
                                ) -> tuple:
    """
    Pure-logic helper: apply all interaction cleaning rules to *raw* DataFrame
    and return (clean_df, quarantine_dict).

    Separated from I/O so validate.py can unit-test each rule without touching
    the production CSV files.

    Cleaning steps (in order):
      1. Validate interaction_id non-null; quarantine null-ID rows.
      2. Parse event_at as datetime; quarantine unparseable or null event_at.
      3. Parse available_at as datetime; quarantine rows with null available_at.
      4. Validate timestamp consistency: quarantine rows where available_at < event_at.
      5. Validate company_id FK against companies_df; quarantine orphans.
      6. Validate interaction_type against known set; quarantine unknown types.
      7. Deduplicate on interaction_id (keep first after all other filters).
    """
    VALID_TYPES = {"executive_briefing", "marketing_event", "cs_meeting", "sales_meeting",
                   "budget_concern"}
    quarantine: dict = {}
    df = raw.copy()

    # Handle extended columns — fill with safe defaults if absent (backward compat)
    if "owner_role" not in df.columns:
        df["owner_role"] = None
    if "is_owner_handoff" not in df.columns:
        df["is_owner_handoff"] = False
    else:
        df["is_owner_handoff"] = df["is_owner_handoff"].fillna(False).astype(bool)
    if "new_owner_id" not in df.columns:
        df["new_owner_id"] = None
    if "agreed_action_id" not in df.columns:
        df["agreed_action_id"] = None
    if "agreed_action_due_date" not in df.columns:
        df["agreed_action_due_date"] = None
    if "agreed_action_completed_at" not in df.columns:
        df["agreed_action_completed_at"] = None

    # 1. Non-null interaction_id
    n_null_id = int(df["interaction_id"].isna().sum())
    if n_null_id > 0:
        quarantine["null_interaction_id"] = n_null_id
    df = df[df["interaction_id"].notna()]

    # 2. Parse event_at; quarantine unparseable
    df["event_at"] = safe_parse_date(df["event_at"], "event_at", quarantine)
    before = len(df)
    df = df.dropna(subset=["event_at"])
    quarantine["missing_event_at"] = before - len(df)

    # 3. Parse available_at; quarantine null (unknown log time → cannot safely
    #    include or exclude from any point-in-time scoring window)
    df["available_at"] = safe_parse_date(df["available_at"], "available_at", quarantine)
    n_null_avail = int(df["available_at"].isna().sum())
    if n_null_avail > 0:
        quarantine["null_available_at"] = n_null_avail
    df = df[df["available_at"].notna()]

    # 4. Timestamp consistency: available_at must not precede event_at
    bad_ts = df["available_at"] < df["event_at"]
    n_bad_ts = int(bad_ts.sum())
    if n_bad_ts > 0:
        quarantine["available_before_event"] = n_bad_ts
    df = df[~bad_ts]

    # 5. FK: company_id must be in cleaned companies
    valid_cos = set(companies_df["company_id"].astype(str))
    bad_co = ~df["company_id"].astype(str).isin(valid_cos)
    n_bad_co = int(bad_co.sum())
    if n_bad_co > 0:
        quarantine["unknown_company_id"] = n_bad_co
    df = df[~bad_co]

    # 6. Validate interaction_type
    bad_type = ~df["interaction_type"].isin(VALID_TYPES)
    n_bad_type = int(bad_type.sum())
    if n_bad_type > 0:
        quarantine["invalid_interaction_type"] = n_bad_type
    df = df[~bad_type]

    # 7. Deduplicate on interaction_id
    dupes = int(df.duplicated(subset=["interaction_id"]).sum())
    df = df.drop_duplicates(subset=["interaction_id"], keep="first")

    quarantine["_dupes"] = dupes
    return df, quarantine


def clean_account_interactions(companies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean account_interactions.csv.  Delegates all rule logic to
    _apply_interaction_cleaning(); this function handles I/O only.
    """
    raw = pd.read_csv(DATA_RAW / "account_interactions.csv")
    raw_rows = len(raw)

    clean_df, quarantine = _apply_interaction_cleaning(raw, companies_df)

    dupes  = quarantine.pop("_dupes", 0)
    n_bad_co    = quarantine.get("unknown_company_id", 0)
    n_bad_type  = quarantine.get("invalid_interaction_type", 0)
    n_null_avail = quarantine.get("null_available_at", 0)

    clean_df.to_csv(DATA_CLEAN / "account_interactions_clean.csv", index=False)
    qr.record_table("account_interactions", raw_rows, len(clean_df),
                    dupes, n_bad_co, quarantine)
    print(f"  account_interactions: {raw_rows:,} raw → {len(clean_df):,} clean "
          f"({dupes} dupes, {n_bad_co} bad_company, {n_bad_type} invalid types, "
          f"{n_null_avail} null_available_at)")
    return clean_df


def clean_contract_events(companies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean contract_events.csv.

    Validates:
      1. company_id FK against cleaned companies (orphan events quarantined).
      2. contract_id references against cleaned order_items (events referencing
         unknown contracts are quarantined and counted separately for cancellations,
         which feed Part 1 retention calculations).
      3. Deduplication on event_id.

    Note: contract_id FK check requires order_items.csv to be already written.
    If it doesn't exist yet, the check is skipped and a warning is printed.
    """
    print("[contract_events] Cleaning...")
    raw = pd.read_csv(DATA_RAW / "contract_events.csv")
    raw_rows = len(raw)
    quarantine = {}

    df = raw.copy()

    # 1. Company FK
    valid_cos = set(companies_df["company_id"])
    fk_co = ~df["company_id"].isin(valid_cos)
    n_fk_co = int(fk_co.sum())
    if n_fk_co > 0:
        quarantine["invalid_company_id"] = n_fk_co
    df = df[~fk_co]

    # 2. Contract FK — reconcile against cleaned order_items
    #
    # Cancellation events must be treated differently from other event types:
    #   - cancellation.contract_id is the zero-value cancellation transaction, which
    #     may legitimately have no matching order item in clean order_items.
    #   - cancellation.expiring_contract_id is the contract being cancelled — this
    #     is the FK that matters for label building (build_churn_labels uses it).
    #
    # If we quarantine a cancellation because its contract_id fails the FK check, the
    # label builder never sees the cancellation event → affected rows get churned=0
    # (false negative).  Fix: for cancellations, validate expiring_contract_id.
    items_path = DATA_CLEAN / "order_items.csv"
    if items_path.exists():
        clean_items = pd.read_csv(items_path, usecols=["contract_id"])
        valid_cids = set(clean_items["contract_id"].dropna().astype(str))
        # contract_id column may have mixed types; coerce to str for comparison
        df["_contract_id_str"] = df["contract_id"].astype(str)
        cancel_mask = df["event_type"] == "cancellation"

        # For cancellations: expiring_contract_id must reference a known contract
        df["_expiring_cid_str"] = df["expiring_contract_id"].astype(str)
        bad_cancel_fk = (
            cancel_mask &
            df["_expiring_cid_str"].notna() &
            ~df["_expiring_cid_str"].isin(valid_cids)
        )

        # For all other event types: contract_id must reference a known contract
        fk_cid = df["_contract_id_str"].notna() & ~df["_contract_id_str"].isin(valid_cids)
        bad_other_fk = ~cancel_mask & fk_cid

        bad_combined = bad_cancel_fk | bad_other_fk
        n_fk_cid = int(bad_other_fk.sum())
        n_bad_cancel = int(bad_cancel_fk.sum())
        if n_fk_cid > 0:
            quarantine["unresolved_contract_id"] = n_fk_cid
        if n_bad_cancel > 0:
            quarantine["unresolved_cancellation_expiry"] = n_bad_cancel
            print(f"  [WARN] {n_bad_cancel} cancellation events reference unknown expiring_contract_id "
                  f"— quarantined")
        df = df[~bad_combined].drop(columns=["_contract_id_str", "_expiring_cid_str"])
    else:
        print("  [WARN] order_items.csv not found; skipping contract FK check")

    # 3. Deduplicate on event_id
    df_clean = df.drop_duplicates(subset=["event_id"])
    df_clean.to_csv(DATA_CLEAN / "contract_events_clean.csv", index=False)
    n_fk_total = (n_fk_co
                  + quarantine.get("unresolved_contract_id", 0)
                  + quarantine.get("unresolved_cancellation_expiry", 0))
    qr.record_table("contract_events", raw_rows, len(df_clean), 0, n_fk_total, quarantine)
    print(f"  raw={raw_rows:,}  clean={len(df_clean):,}  "
          f"fk_company={n_fk_co}  "
          f"fk_contract(non-cancel)={quarantine.get('unresolved_contract_id', 0)}  "
          f"fk_cancel_expiry={quarantine.get('unresolved_cancellation_expiry', 0)}")
    return df_clean


# ── Renewal quotes ───────────────────────────────────────────────────────────

def clean_renewal_quotes(items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean renewal_quotes.csv.

    Validates:
      1. quote_id uniqueness.
      2. contract_id FK against cleaned order_items.
      3. issued_at <= available_at (quarantine rows where available_at < issued_at).
      4. Dates must be parseable.

    Returns cleaned DataFrame saved to data/cleaned/renewal_quotes_clean.csv.
    """
    raw_path = DATA_RAW / "renewal_quotes.csv"
    if not raw_path.exists():
        print("[renewal_quotes] File not found — skipping")
        return pd.DataFrame()

    print("[renewal_quotes] Cleaning...")
    raw = pd.read_csv(raw_path)
    raw_rows = len(raw)
    quarantine = {}
    quarantine_rows = []

    df = raw.copy()

    # 1. Parse dates
    df["issued_at_dt"]    = safe_parse_date(df["issued_at"],    "issued_at",    quarantine)
    df["available_at_dt"] = safe_parse_date(df["available_at"], "available_at", quarantine)

    # Quarantine rows with unparseable dates
    bad_dates = df["issued_at_dt"].isna() | df["available_at_dt"].isna()
    n_bad_dates = int(bad_dates.sum())
    if n_bad_dates > 0:
        quarantine["unparseable_dates"] = n_bad_dates
        bad_date_rows = df[bad_dates].copy()
        bad_date_rows["quarantine_reason"] = "unparseable_dates"
        quarantine_rows.append(bad_date_rows)
    df = df[~bad_dates]

    # 2. available_at must be >= issued_at
    bad_ts = df["available_at_dt"] < df["issued_at_dt"]
    n_bad_ts = int(bad_ts.sum())
    if n_bad_ts > 0:
        quarantine["available_before_issued"] = n_bad_ts
        bad_ts_rows = df[bad_ts].copy()
        bad_ts_rows["quarantine_reason"] = "available_before_issued"
        quarantine_rows.append(bad_ts_rows)
    df = df[~bad_ts]

    # 3. contract_id FK against cleaned order_items
    valid_contract_ids = set(items_df["contract_id"].dropna().astype(str))
    fk_bad = ~df["contract_id"].astype(str).isin(valid_contract_ids)
    n_fk_bad = int(fk_bad.sum())
    if n_fk_bad > 0:
        quarantine["unresolved_contract_fk"] = n_fk_bad
        fk_rows = df[fk_bad].copy()
        fk_rows["quarantine_reason"] = "unresolved_contract_fk"
        quarantine_rows.append(fk_rows)
    df = df[~fk_bad]

    # 4. Deduplicate on quote_id
    dupes = check_pk_unique(df, "quote_id", "renewal_quotes")
    df = df.drop_duplicates(subset=["quote_id"])

    # Drop working datetime columns
    df = df.drop(columns=["issued_at_dt", "available_at_dt"], errors="ignore")

    # Export quarantine CSV
    non_empty = [r for r in quarantine_rows if len(r) > 0]
    if non_empty:
        all_quarantine = pd.concat(non_empty, ignore_index=True)
        all_quarantine.to_csv(DATA_CLEAN / "quarantine_renewal_quotes.csv", index=False)
        print(f"  [quarantine] {len(all_quarantine):,} rows written to quarantine_renewal_quotes.csv")

    df.to_csv(DATA_CLEAN / "renewal_quotes_clean.csv", index=False)
    qr.record_table("renewal_quotes", raw_rows, len(df), dupes, n_fk_bad, quarantine)
    print(f"  raw={raw_rows:,}  clean={len(df):,}  dupes={dupes}  fk_bad={n_fk_bad}  "
          f"bad_ts={n_bad_ts}  bad_dates={n_bad_dates}")
    return df


# ── Cleaning accuracy vs injected anomalies ───────────────────────────────────

def evaluate_cleaning_accuracy(orders_raw: pd.DataFrame, orders_clean: pd.DataFrame,
                               items_clean: pd.DataFrame,
                               dedup_removed_order_ids: set = None):
    """Compare cleaning results against injected anomaly ground truth.

    For duplicate_order: uses dedup_removed_order_ids (the set of order_ids
    removed by the deduplication step specifically) rather than raw_count -
    clean_count.  The net-difference approach overstates detection when another
    cleaning stage (e.g. invalid_date) also removes a copy of the same order_id.

    dedup_removed_order_ids: set of order_id strings removed at the dedup step.
    If None, falls back to the (less accurate) net-difference approach.
    """
    gt_path = EVAL_DIR / "anomaly_ground_truth.csv"
    if not gt_path.exists():
        print("  [WARN] anomaly_ground_truth.csv not found; skipping accuracy check")
        return

    gt = pd.read_csv(gt_path)
    gt_by_type = gt.groupby("anomaly_type")
    clean_order_ids = set(orders_clean["order_id"].astype(str))
    raw_order_ids   = set(orders_raw["order_id"].astype(str))

    results = {}
    for atype, group in gt_by_type:
        injected_ids = set(group["record_id"].astype(str))
        n_injected = len(injected_ids)

        if atype == "duplicate_order":
            # Duplicates are injected as exact-copy rows with the same order_id.
            # GT record_id = the canonical order_id that was duplicated.
            # Each injected id has exactly 1 extra copy → max detectable = n_injected.
            #
            # Use dedup_removed_order_ids if available: it records exactly which
            # order_ids had a row removed at the dedup step, before any other cleaning
            # stage runs.  This avoids double-counting with invalid_date or FK removals.
            if dedup_removed_order_ids is not None:
                detected = len(injected_ids & dedup_removed_order_ids)
            else:
                # Fallback: net difference, capped at n_injected
                raw_counts   = orders_raw[orders_raw["order_id"].isin(injected_ids)].shape[0]
                clean_counts = orders_clean[orders_clean["order_id"].isin(injected_ids)].shape[0]
                detected = min(raw_counts - clean_counts, n_injected)
            # False positives: unique order_ids removed not explained by any known anomaly
            other_anomaly_ids = set(gt[gt["anomaly_type"] != "duplicate_order"]["record_id"].astype(str))
            raw_id_counts  = orders_raw["order_id"].value_counts()
            unique_raw_ids = set(raw_id_counts[raw_id_counts == 1].index)
            unexplained_removed = (unique_raw_ids - set(orders_clean["order_id"])) - other_anomaly_ids
            false_pos = int(len(unexplained_removed))
        elif atype == "product_id_case":
            # Items should have normalized product_id (uppercase)
            matching = items_clean[items_clean["order_item_id"].isin(injected_ids)]
            detected = int((matching["product_id"] == matching["product_id"].str.upper()).sum())
            false_pos = int((matching["product_id"] != matching["product_id"].str.upper()).sum())
        elif atype == "missing_team_mapping":
            # We flag but keep; detection = flagged as null (ground truth rows should be null)
            matching = items_clean[items_clean["order_item_id"].isin(injected_ids)]
            detected = int(matching["owner_team_id"].isna().sum())
            false_pos = 0
        elif atype == "invalid_date":
            # Rows with invalid dates removed from clean output
            surviving = injected_ids & clean_order_ids
            detected = n_injected - len(surviving)
            false_pos = 0
        else:
            detected = 0
            false_pos = 0

        results[atype] = {
            "injected": n_injected,
            "detected": detected,
            "missed": max(0, n_injected - detected),
            "false_positives": false_pos,
            "detection_rate_pct": round(100 * detected / max(n_injected, 1), 1),
        }

    qr.cleaning_accuracy = results
    print("  [cleaning accuracy]")
    for atype, r in results.items():
        print(f"    {atype}: {r['detected']}/{r['injected']} detected "
              f"({r['detection_rate_pct']}%) | FP={r['false_positives']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# Must match generate_data._EXPECTED_HASH_FILES exactly
_EXPECTED_HASH_FILES: frozenset = frozenset({
    "products.csv",
    "parent_orgs.csv",
    "companies.csv",
    "teams.csv",
    "users.csv",
    "orders.csv",
    "order_items.csv",
    "deployments.csv",
    "telemetry_weekly.csv",
    "support_tickets.csv",
    "contract_events.csv",
    "entitlement_deployment_map.csv",
    "deployment_team_map.csv",
    "account_interactions.csv",
    "renewal_quotes.csv",
})


def _acquire_stage_lock(directory: Path) -> object:
    """
    Acquire an exclusive lock (fcntl.LOCK_EX) on the pipeline lock file.
    Using LOCK_EX (not LOCK_SH) ensures that only one pipeline stage runs at a
    time: generation, cleaning, and integration are all mutually exclusive.
    Without this, two simultaneous cleaners could both write to data/cleaned/,
    or a cleaner could overwrite cleaned CSVs while integration reads them.
    Fails immediately (LOCK_NB) if any other stage is active.
    """
    lock_path = directory / "._generating.lock"
    lock_fh = open(lock_path, "a")  # 'a' creates file if absent; does not truncate
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_fh.close()
        raise RuntimeError(
            "[stage-lock] Cannot acquire exclusive lock — another pipeline stage is active.\n"
            "Wait for it to finish, then re-run ingest_clean.py."
        )
    return lock_fh


def _verify_raw_coherence():
    """
    Hard gate: manifest must exist, run_id must be present, and every file
    whose hash was stored in the manifest must still match on disk.

    Count-only checks can miss same-size overwrites (e.g. a new full-scale
    run that happens to produce the same row count for one table).  SHA-256
    hashes are the only reliable check.

    The write lock prevents concurrent generation runs from writing to the
    same directory, but the hash check catches any contamination that slipped
    through before the lock was introduced.
    """
    manifest_path = OUTPUTS / "generation_manifest.json"

    # Metadata is mandatory — no silent skips
    if not manifest_path.exists():
        raise RuntimeError(
            "[coherence] generation_manifest.json not found.\n"
            "Run generate_data.py before ingest_clean.py."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    run_id = manifest.get("run_id")
    if not run_id:
        raise RuntimeError(
            "[coherence] generation_manifest.json is missing run_id.\n"
            "This manifest was written by an older version of generate_data.py.\n"
            "Re-run generate_data.py to produce a fresh manifest with a run_id."
        )

    file_hashes = manifest.get("file_hashes", {})
    if not file_hashes:
        raise RuntimeError(
            "[coherence] generation_manifest.json has no file_hashes.\n"
            "Re-run generate_data.py to produce a manifest with per-file SHA-256 hashes."
        )

    # Require exact key match — extra or missing entries both signal a
    # generate_data.py version mismatch (e.g. deployment_team_map.csv absent
    # from an older manifest).
    manifest_keys = frozenset(file_hashes)
    extra   = manifest_keys - _EXPECTED_HASH_FILES
    missing = _EXPECTED_HASH_FILES - manifest_keys
    inventory_errors = []
    if missing:
        inventory_errors.append(
            "  Missing from manifest hashes: " + ", ".join(sorted(missing))
        )
    if extra:
        inventory_errors.append(
            "  Unexpected entries in manifest hashes: " + ", ".join(sorted(extra))
        )
    if inventory_errors:
        raise RuntimeError(
            "[coherence] Manifest file_hashes inventory does not match expected set.\n"
            "Re-run generate_data.py with the current version:\n"
            + "\n".join(inventory_errors)
        )

    # Verify every hashed file — missing file is also a failure
    errors = []
    for fname, expected_hash in file_hashes.items():
        fpath = DATA_RAW / fname
        if not fpath.exists():
            errors.append(f"  {fname}: expected on disk but not found")
            continue
        actual_hash = _sha256_file(fpath)
        if actual_hash != expected_hash:
            errors.append(
                f"  {fname}: hash mismatch\n"
                f"    manifest={expected_hash}\n"
                f"    on-disk ={actual_hash}"
            )

    if errors:
        raise RuntimeError(
            "[coherence] Raw file hashes do not match generation manifest.\n"
            "A concurrent or background run overwrote files after generation.\n"
            "Re-run generate_data.py to produce a coherent dataset:\n"
            + "\n".join(errors)
        )

    print(f"  [coherence] {len(file_hashes)} file hashes verified "
          f"(run_id={run_id}  dev_mode={manifest.get('dev_mode')}  "
          f"companies={manifest.get('company_count'):,})")
    return run_id


def main():
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    t0 = time.time()

    # Acquire shared lock BEFORE hash verification — holds throughout all reads
    # so generate_data.py cannot overwrite files between the hash check and the
    # actual pd.read_csv calls that follow.
    _stage_lock_fh = _acquire_stage_lock(DATA_RAW)
    run_id = _verify_raw_coherence()

    products_df  = clean_products(cfg)
    companies_df = clean_companies()
    teams_df     = clean_teams(companies_df)
    users_df     = clean_users(companies_df, teams_df)
    orders_raw   = pd.read_csv(DATA_RAW / "orders.csv")  # preserve raw for accuracy eval
    orders_df, dedup_removed_order_ids = clean_orders(users_df, companies_df)
    items_df     = clean_order_items(orders_df, products_df)
    deps_df      = clean_deployments(companies_df)
    _            = clean_deployment_team_map(deps_df)
    clean_telemetry(deps_df)
    _            = clean_entitlement_map(items_df)
    _            = clean_support_tickets(companies_df, users_df, items_df)
    _            = clean_contract_events(companies_df)
    _            = clean_account_interactions(companies_df)
    _            = clean_renewal_quotes(items_df)

    print("\n[cleaning accuracy] Evaluating against injected anomalies...")
    evaluate_cleaning_accuracy(orders_raw, orders_df, items_df, dedup_removed_order_ids)

    elapsed = round(time.time() - t0, 1)
    report = qr.to_dict()
    report["runtime_seconds"] = elapsed
    report["run_id"] = run_id or "unknown"

    def _json_default(obj):
        if hasattr(obj, "item"):  # numpy scalars
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(OUTPUTS / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2, default=_json_default)

    print(f"\n✓ Cleaning complete in {elapsed}s")
    print(f"  Quality report: outputs/quality_report.json")

    # Print summary
    print("\n  ── Core table integrity ──")
    for tname, info in report["core_table_integrity"].items():
        print(f"  {tname:<20} raw={info['raw_rows']:>8,}  clean={info['clean_rows']:>8,}  "
              f"dupes={info['dupes_removed']:>5}  fk_bad={info['fk_unresolved']:>5}")

    print("\n  ── Commercial consistency ──")
    for k, v in report["commercial_consistency"].items():
        print(f"  {k}: {v:,}")

    print("\n  ── Telemetry consistency ──")
    for k, v in report["telemetry_consistency"].items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()

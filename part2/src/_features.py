"""
_features.py — Shared feature specification for Part 2 model pipeline.

All of train.py, evaluate.py, and prioritize.py import from here.
Do NOT copy feature lists or preprocessing logic into callers.

Run ID this file is paired with: 82927c54-85b2-4dc1-aee9-dba8a32815ca

Three declared feature experiments (§2 of implementation plan):
  EXPERIENCE_LED   — health/service/commercial/relationship inputs plus minimal
                     applicability controls; no account value or company-size variables.
  SIZE_VALUE       — size, employee count, ACV, renewal timing and minimal product context;
                     tests how much prediction comes from structural differences.
  FULL_CONTEXT     — experience-led plus size/value, portfolio breadth and relationship context.

Prohibited predictors (never in any model input):
  See PROHIBITED_PREDICTORS list below — includes simulator eval states, true cancellation
  probabilities, target column, maturity flags and any field that only becomes known
  after the label window closes.

Missing-value policy per block:
  deployment_reach_30d   NaN → 0.0  (inapplicable product; add _missing indicator)
  workflow_success_30d   NaN → 0.5  (unknown — no workflow attempts; add _missing indicator)
  avg_coverage_90d       NaN → 0.5  (unknown measurement; neutral, not worst-case)
  days_since_last_interaction  NaN → train-fitted median (add _missing indicator)
  avg_resolution_hrs_90d NaN → 0.0  (no resolved tickets; add _missing indicator)
  median_prior_renewal_delay_days NaN → 0.0 (first renewal; add _missing indicator)
  scan_success_rate_90d  NaN → 1.0  (not applicable for non-Prisma; zero penalty)
  All count/binary fields  NaN → 0.0

NOTE: missing_telemetry_pct_90d is EXCLUDED from all feature sets. The main scenario
uses reliable reporting (rate ≈ 0%). Including it would add a constant zero predictor.
Drop it in ColumnTransformer; do not include in any FEATURES list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression  # noqa: F401 – imported by callers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Audit / identity columns (never predictors, always kept for grouping/audit)
# ---------------------------------------------------------------------------
AUDIT_COLS = [
    "contract_id",
    "renewal_lineage_id",
    "company_id",
    "parent_org_id",
    "as_of_date",
    "term_start",
    "term_end",
    "label_available_at",
    "churned",
    "label_matured",
    "eligibility_reason",
    "incomplete_history",
    "split",
    "product_id",
    "offering_type",
    "size_segment",   # also used as categorical predictor; kept here for audit grouping
]

TARGET = "churned"

# ---------------------------------------------------------------------------
# Prohibited predictors — never include in X
# ---------------------------------------------------------------------------
PROHIBITED_PREDICTORS = [
    # Directly managed case-study scenario: sales route is outside the analysis.
    "channel", "sales_channel",
    # Simulator eval states / true probabilities
    "churn_prob",
    "true_churn_probability",
    "deployment_archetype",
    "experience_state",
    # Target and maturity fields
    "churned",
    "label_matured",
    "label_available_at",
    # Future-known commercial outcomes (not available at scoring time)
    "accepted_price",
    "renewal_accepted",
    "successor_contract_id",
    # Constant reporting defect predictor (main scenario: reliable telemetry ≈ 0%)
    "missing_telemetry_pct_90d",
]

# ---------------------------------------------------------------------------
# ── Experiment 1: EXPERIENCE-LED ─────────────────────────────────────────────
# Health/service/commercial/relationship inputs + applicability controls.
# Deliberately excludes account value and company-size variables.
# ---------------------------------------------------------------------------

EXPERIENCE_NUM = [
    # Deployment block
    "deployment_reach_30d",        # fraction of entitled scope actively deployed
    "onboarding_delay_days",       # days milestone is overdue (0 if on-time / achieved)
    # Value block
    "workflow_success_30d",        # workflow success rate in prior 30 days
    "productive_use_delta_30d",    # 30-day change in workflow success rate
    # Service block
    "unresolved_customer_impact_hours_90d",  # accumulated unresolved impact hours
    "repeat_sla_breaches_90d",     # count of repeat SLA breaches in 90 days
    "p1_tickets_90d",              # P1/critical tickets opened
    "open_tickets_90d",            # open tickets at snapshot
    "avg_open_age_hrs_90d",        # mean age of open tickets
    "avg_resolution_hrs_90d",      # mean time to resolve (NaN if no resolutions)
    # Telemetry/coverage
    "avg_coverage_90d",            # mean fraction of entitled units in active use
    "coverage_trend_90d",          # linear slope of weekly coverage
    "obs_completeness_90d",        # fraction of expected observations present
    # Commercial block
    "quoted_unit_price_change_pct",  # pct change in unit price vs prior quote
    "dated_budget_concern",        # 1 if budget concern event recorded
    # Relationship block
    "days_without_confirmed_owner",  # days without a confirmed vendor-side owner
    "overdue_agreed_actions",      # count of overdue agreed follow-up actions
    # Applicability controls (not predictors of mechanism; context for interpretation)
    "days_to_renewal",
    "tenure_days",
    # Missingness indicators (add_indicator_cols must be called before ColumnTransformer)
    "avg_resolution_hrs_90d_missing",
    "deployment_reach_30d_missing",
    "workflow_success_30d_missing",
    "days_since_last_interaction_missing",  # needed for full context; zero-cost in experience
    "median_prior_renewal_delay_missing",
]

EXPERIENCE_CAT = [
    "product_family",
]

# ---------------------------------------------------------------------------
# ── Experiment 2: SIZE/VALUE COMPARATOR ──────────────────────────────────────
# Tests how much prediction comes from structural commercial differences.
# ---------------------------------------------------------------------------

SIZE_VALUE_NUM = [
    "employee_count",
    "acv",
    "total_acv_in_window",
    "days_to_renewal",
    "tenure_days",
    "n_products_in_window",
    "n_families_in_window",
]

SIZE_VALUE_CAT = [
    "size_segment",
    "industry",
    "region",
    "product_family",
]

# ---------------------------------------------------------------------------
# ── Experiment 3: FULL CONTEXT ────────────────────────────────────────────────
# Experience-led + size/value + portfolio breadth + relationship context.
# Check n_active_products_in_family vs family_adoption_completeness for
# redundancy before including both (plan §2 note).
# ---------------------------------------------------------------------------

FULL_CONTEXT_NUM = EXPERIENCE_NUM + [
    # Size/value (not in experience-led)
    "employee_count",
    "acv",
    "total_acv_in_window",
    "n_products_in_window",
    "n_families_in_window",
    # Breadth context
    # NOTE: n_active_products_in_family and family_adoption_completeness may be
    # near-redundant (plan §2). Inspect correlation before final model; include
    # both here for the redundancy check to be performed during selection.
    "n_active_products_in_family",
    "family_sibling_cancellations_1yr",
    "family_adoption_completeness",   # = active_in_family / available_in_catalog; see feature_contract.md
    # Relationship context (full set)
    "executive_interactions_180d",
    "marketing_engagements_180d",
    "days_since_last_interaction",
    "median_prior_renewal_delay_days",
    # Additional support context
    "tickets_90d",
    # Additional operational context
    "policy_events_per_unit_90d",
    "remediation_velocity_90d",
    "automation_adoption_90d",
    "scan_success_rate_90d",          # NaN → 1.0 for non-Prisma (not applicable)
]

FULL_CONTEXT_CAT = [
    "product_family",
    "size_segment",
    "industry",
    "region",
]

# De-duplicate (EXPERIENCE_NUM items listed again in FULL_CONTEXT_NUM for readability)
FULL_CONTEXT_NUM = list(dict.fromkeys(FULL_CONTEXT_NUM))


# ---------------------------------------------------------------------------
# Indicator columns (add before passing to ColumnTransformer)
# ---------------------------------------------------------------------------

# Maps source column → indicator column name.
# A _missing indicator = 1 means the source value was NaN (inapplicable or unmeasured).
INDICATOR_COLS: dict[str, str] = {
    "avg_resolution_hrs_90d":         "avg_resolution_hrs_90d_missing",
    "deployment_reach_30d":           "deployment_reach_30d_missing",
    "workflow_success_30d":           "workflow_success_30d_missing",
    "days_since_last_interaction":    "days_since_last_interaction_missing",
    "median_prior_renewal_delay_days": "median_prior_renewal_delay_missing",
}

# NaN fill values applied AFTER the indicator is captured
NULL_FILLS: dict[str, float] = {
    "avg_resolution_hrs_90d":          0.0,   # no resolved tickets → 0 resolution hrs
    "deployment_reach_30d":            0.0,   # inapplicable → no gap penalty via indicator
    "workflow_success_30d":            0.5,   # unknown → neutral (preserve distinction from 0)
    "productive_use_delta_30d":        0.0,   # no change (neutral)
    "avg_coverage_90d":                0.5,   # unknown measurement → neutral
    "coverage_trend_90d":              0.0,   # no trend signal
    "obs_completeness_90d":            0.5,   # neutral
    "p1_tickets_90d":                  0.0,
    "open_tickets_90d":                0.0,
    "avg_open_age_hrs_90d":            0.0,
    "tickets_90d":                     0.0,
    "repeat_sla_breaches_90d":         0.0,
    "unresolved_customer_impact_hours_90d": 0.0,
    "quoted_unit_price_change_pct":    0.0,   # no quote change
    "dated_budget_concern":            0.0,
    "days_without_confirmed_owner":    0.0,
    "overdue_agreed_actions":          0.0,
    "executive_interactions_180d":     0.0,
    "marketing_engagements_180d":      0.0,
    "scan_success_rate_90d":           1.0,   # not applicable for non-Prisma → no penalty
    "policy_events_per_unit_90d":      0.0,
    "remediation_velocity_90d":        0.0,
    "automation_adoption_90d":         0.0,
    "family_sibling_cancellations_1yr": 0.0,
    "family_adoption_completeness":    0.0,
    "n_active_products_in_family":     0.0,
    "onboarding_delay_days":           0.0,
    # days_since_last_interaction: train-fitted median (load from imputation_values.json)
    # median_prior_renewal_delay_days: handled via indicator + 0.0 fill
    "median_prior_renewal_delay_days": 0.0,
}


def add_indicator_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary missingness indicators for source columns listed in INDICATOR_COLS,
    then apply NULL_FILLS in-place.

    Must be called before passing X to a ColumnTransformer.
    Operates on a copy; does NOT modify the caller's DataFrame.
    """
    df = df.copy()
    for src_col, ind_col in INDICATOR_COLS.items():
        if src_col in df.columns:
            df[ind_col] = df[src_col].isna().astype(float)
    for col, fill in NULL_FILLS.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill)
    return df


def apply_imputation_values(
    df: pd.DataFrame, imputation_values: dict
) -> pd.DataFrame:
    """
    Apply train-fitted imputation values loaded from imputation_values.json.
    Call AFTER add_indicator_cols (indicator captures NaN before fill).
    """
    df = df.copy()
    median_dsi = imputation_values.get("days_since_last_interaction_median")
    if median_dsi is not None and "days_since_last_interaction" in df.columns:
        df["days_since_last_interaction"] = df["days_since_last_interaction"].fillna(
            float(median_dsi)
        )
    return df


# ---------------------------------------------------------------------------
# Preprocessing pipeline builders
# ---------------------------------------------------------------------------

def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    model_type: Literal["lr", "hgb"] = "lr",
) -> ColumnTransformer:
    """
    Build an unfitted ColumnTransformer for the given feature lists.

    model_type='lr'  → numeric pipeline includes StandardScaler (required for LR).
    model_type='hgb' → numeric pipeline skips scaler (tree-based; scale-invariant).

    The transformer is NOT fitted here. Fit it inside a Pipeline on training data only.
    Save the fitted Pipeline with joblib; load it for evaluation and scoring.
    """
    if model_type == "lr":
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
    else:  # hgb
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,    numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------

def assert_no_prohibited(df: pd.DataFrame, context: str = "") -> None:
    """Raise ValueError if any prohibited predictor column is present in df."""
    found = [c for c in PROHIBITED_PREDICTORS if c in df.columns]
    if found:
        raise ValueError(
            f"Prohibited predictor(s) found in feature matrix{' (' + context + ')' if context else ''}: "
            f"{found}. Remove before passing to model."
        )


def check_feature_availability(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    """Return list of declared features missing from df. Caller decides how to handle."""
    declared = set(numeric_features) | set(categorical_features)
    missing = [c for c in declared if c not in df.columns]
    return missing

"""
_expansion_decomp.py — Pure business-logic functions for expansion ACV decomposition.

Importable by both analyze_growth.py (production) and test_regression_fixtures.py.
No side-effects at import time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RECONCILIATION_TOLERANCE = 0.02   # max acceptable |residual| per row in dollars

# Columns required for a row to be "eligible" for decomposition
_REQUIRED_COLS = [
    "quantity", "unit_price", "term_years", "line_discount_amount",
    "prior_quantity_actual", "prior_unit_price_actual", "prior_term_years",
    "prior_discount_actual", "acv_change",
]


def _validate_inputs(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask of rows eligible for decomposition.

    A row is INELIGIBLE (mask=False) if any required column is null,
    if term_years or prior_term_years is ≤ 0, or if acv_change is not finite.
    Ineligible rows receive acv_component_residual = NaN and are routed to the
    "unresolved" category by classify_expansion_driver.  They do NOT trigger the
    reconciliation assertion.
    """
    eligible = pd.Series(True, index=df.index)

    for col in _REQUIRED_COLS:
        if col in df.columns:
            eligible &= df[col].notna()
        else:
            eligible &= False  # missing column → entire frame ineligible

    # Positive term lengths required for safe division
    if "term_years" in df.columns:
        eligible &= pd.to_numeric(df["term_years"], errors="coerce").fillna(0) > 0
    if "prior_term_years" in df.columns:
        eligible &= pd.to_numeric(df["prior_term_years"], errors="coerce").fillna(0) > 0

    # Finite acv_change
    if "acv_change" in df.columns:
        eligible &= np.isfinite(pd.to_numeric(df["acv_change"], errors="coerce").fillna(np.nan))

    return eligible


def decompose_expansion_acv(df_evt: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose ACV changes into volume / price / discount / residual components
    using annualized prices.

    Both current and predecessor unit_price and line_discount_amount are divided
    by their respective term_years before decomposing.  This prevents term-length
    changes from being misclassified as price movements.

    Parameters
    ----------
    df_evt
        DataFrame of expansion (or contraction) events.  Must include:
          quantity, unit_price, term_years              (current contract)
          prior_quantity_actual, prior_unit_price_actual,
          prior_term_years, prior_discount_actual        (from predecessor join)
          line_discount_amount                           (current contract)
          acv_change                                     (net_acv - prior_net_acv)

    Returns
    -------
    df_evt with added columns:
        eligible_for_decomposition  (bool),
        annual_unit_price, prior_annual_unit_price,
        annual_discount, prior_annual_discount,
        qty_change, annual_price_change,
        acv_component_volume, acv_component_price,
        acv_component_discount, acv_component_residual.

    Ineligible rows (null inputs, invalid term_years, non-finite acv_change)
    receive NaN for all component columns and are classified as
    "unknown_no_predecessor_data" by classify_expansion_driver.

    Raises
    ------
    AssertionError
        If any ELIGIBLE row's component sum deviates from acv_change by more than
        RECONCILIATION_TOLERANCE dollars.

    Design notes
    ------------
    net_acv = (quantity × unit_price − line_discount_amount) / term_years
    Therefore the "annual unit price" is unit_price / term_years.

    Decomposition (all values in $/year):
        vol      = qty_change × prior_annual_unit_price
        price    = annual_price_change × current_qty
        discount = −(current_annual_discount − prior_annual_discount)
        residual = acv_change − vol − price − discount   (should be ≈ 0 for eligible rows)
    """
    df = df_evt.copy()

    # Pre-validate: identify eligible rows
    eligible = _validate_inputs(df)
    df["eligible_for_decomposition"] = eligible

    n_ineligible = (~eligible).sum()
    if n_ineligible > 0:
        # Initialize component columns to NaN for ineligible rows; filled below for eligible
        for col in [
            "annual_unit_price", "prior_annual_unit_price",
            "annual_discount", "prior_annual_discount",
            "qty_change", "annual_price_change",
            "acv_component_volume", "acv_component_price",
            "acv_component_discount", "acv_component_residual",
        ]:
            df[col] = np.nan

    if eligible.any():
        e = df[eligible].copy()

        # Coerce numerics to be safe
        for col in _REQUIRED_COLS + ["term_years", "prior_term_years"]:
            if col in e.columns:
                e[col] = pd.to_numeric(e[col], errors="coerce")

        # Annualize both sides
        e["annual_unit_price"]       = e["unit_price"] / e["term_years"]
        e["prior_annual_unit_price"] = e["prior_unit_price_actual"] / e["prior_term_years"]
        e["annual_discount"]         = e["line_discount_amount"] / e["term_years"]
        e["prior_annual_discount"]   = e["prior_discount_actual"] / e["prior_term_years"]

        # Changes
        e["qty_change"]          = e["quantity"] - e["prior_quantity_actual"]
        e["annual_price_change"] = e["annual_unit_price"] - e["prior_annual_unit_price"]

        # Components
        e["acv_component_volume"]   = e["qty_change"] * e["prior_annual_unit_price"]
        e["acv_component_price"]    = e["annual_price_change"] * e["quantity"]
        e["acv_component_discount"] = -(e["annual_discount"] - e["prior_annual_discount"])
        e["acv_component_residual"] = (
            e["acv_change"]
            - e["acv_component_volume"]
            - e["acv_component_price"]
            - e["acv_component_discount"]
        )

        # Reconciliation check: only eligible rows with finite residual
        finite_resid = e["acv_component_residual"].dropna()
        if len(finite_resid) > 0:
            max_resid = finite_resid.abs().max()
            if max_resid > RECONCILIATION_TOLERANCE:
                raise AssertionError(
                    f"Decomposition residual {max_resid:.4f} exceeds tolerance "
                    f"{RECONCILIATION_TOLERANCE}. Check annualization logic."
                )

        # Write eligible results back into df
        df.loc[eligible, e.columns] = e.values

    return df


def classify_expansion_driver(row: pd.Series) -> str:
    """
    Choose the dominant driver from the annualized decomposition components.

    Returns one of: quantity_expansion, price_expansion, discount_reduction,
                    other_residual, unknown_no_predecessor_data, unresolved_ineligible.

    A row with eligible_for_decomposition == False (null predecessor data, invalid
    term_years, or non-finite acv_change) is classified as "unresolved_ineligible"
    so it is never silently treated as zero-movement.
    """
    # Ineligible rows: explicit unresolved category
    if row.get("eligible_for_decomposition") is False:
        return "unresolved_ineligible"

    if pd.isna(row.get("prior_quantity_actual")):
        return "unknown_no_predecessor_data"

    # All component columns must be finite for classification
    for col in ("qty_change", "annual_price_change", "acv_component_volume",
                "acv_component_price", "acv_component_discount"):
        if pd.isna(row.get(col)):
            return "unknown_no_predecessor_data"

    qty_grew      = row["qty_change"] > 0
    price_grew    = row["annual_price_change"] > 0
    disc_improved = row["acv_component_discount"] > 0

    components = {
        "quantity_expansion": abs(row["acv_component_volume"])   if qty_grew      else 0,
        "price_expansion":    abs(row["acv_component_price"])    if price_grew    else 0,
        "discount_reduction": abs(row["acv_component_discount"]) if disc_improved else 0,
    }
    if max(components.values()) == 0:
        return "other_residual"
    return max(components, key=components.get)

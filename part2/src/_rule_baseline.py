"""
_rule_baseline.py — Canonical rule-baseline scoring function.

Shared by train.py, evaluate.py, and prioritize.py.
Import this module; do NOT copy the function.

Design rationale
----------------
The previous formula included a 0.2 × missing_telemetry_pct_90d term.
That term is removed because the main simulation scenario uses reliable
reporting (missing_telemetry_pct ≈ 0% in every row). Keeping it would
add a near-constant zero predictor. See §4 of the implementation plan.

The rule is rebuilt from four observable experience blocks that directly
correspond to the simulator's declared experience_weights mechanism:

  deployment_gap    (0.35): incomplete deployment reach increases risk
  value_decline     (0.30): low or declining workflow success increases risk
  service_pain      (0.25): unresolved customer impact and SLA breaches
  commercial_concern(0.10): dated budget concern or price increase

Weights mirror config/simulation.yaml::experience_weights.
This is disclosed as a design choice, not proof of the correct ranking.

NaN handling policy
-------------------
deployment_reach_30d          NaN → 0.0  (inapplicable/unmeasured → no deployment gap scored)
workflow_success_30d          NaN → 0.5  (unknown → neutral; distinguish from confirmed zero)
unresolved_impact_hours_90d   NaN → 0.0  (no confirmed unresolved impact → no penalty)
repeat_sla_breaches_90d       NaN → 0.0  (no confirmed breaches → no penalty)
dated_budget_concern          NaN → 0.0  (no recorded concern → no penalty)

Rule formula (frozen; do not modify without updating analysis_contract.md)
---------------------------------------------------------------------------
deployment_gap     = clip(1 − deployment_reach_30d, 0, 1)
value_decline      = clip(1 − workflow_success_30d, 0, 1)
impact_norm        = clip(unresolved_customer_impact_hours_90d / 720.0, 0, 1)
  (720 h = 30 days × 24 h; reasonable ceiling for severe-impact normalization)
sla_norm           = clip(repeat_sla_breaches_90d / 3.0, 0, 1)
service_pain       = 0.6 × impact_norm + 0.4 × sla_norm
budget_concern     = clip(dated_budget_concern, 0, 1)
commercial_concern = budget_concern

score = 0.35 × deployment_gap
      + 0.30 × value_decline
      + 0.25 × service_pain
      + 0.10 × commercial_concern

Result clipped to [0.0, 1.0].

This is an ordinal risk score. It is NOT a calibrated probability estimate.
Label it a "priority index" in outreach dashboards unless a separate
sigmoid calibration is fitted on the dedicated calibration cohort.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Frozen rule configuration — update analysis_contract.md if any value changes.
RULE_CONFIG = {
    "weights": {
        "deployment_gap":      0.35,
        "value_decline":       0.30,
        "service_pain":        0.25,
        "commercial_concern":  0.10,
    },
    "service_pain_sub_weights": {
        "impact_norm": 0.60,
        "sla_norm":    0.40,
    },
    "normalization": {
        "impact_hours_ceiling":   720.0,   # 30 days × 24 h
        "sla_breaches_ceiling":   3.0,
    },
    "nan_fills": {
        "deployment_reach_30d":                    0.0,
        "workflow_success_30d":                    0.5,
        "unresolved_customer_impact_hours_90d":    0.0,
        "repeat_sla_breaches_90d":                 0.0,
        "dated_budget_concern":                    0.0,
    },
    "removed_term": (
        "missing_telemetry_pct_90d removed (main scenario: reliable reporting; "
        "term was constant near-zero and carried no predictive signal)."
    ),
}


def apply_rule_baseline(df_subset: pd.DataFrame) -> np.ndarray:
    """
    Canonical rule-baseline scoring. Call this; do not re-implement.

    Parameters
    ----------
    df_subset : DataFrame containing at minimum the five input columns listed
                in RULE_CONFIG['nan_fills']. Extra columns are ignored.

    Returns
    -------
    np.ndarray of float, same length as df_subset, values in [0.0, 1.0].
    """
    d = df_subset.copy()

    # ── Deployment block ─────────────────────────────────────────────────────
    # NaN = inapplicable or unmeasured → no deployment gap penalty
    reach = pd.to_numeric(d.get("deployment_reach_30d", pd.Series(np.nan, index=d.index)),
                          errors="coerce").fillna(0.0)
    deployment_gap = (1.0 - reach).clip(0.0, 1.0)

    # ── Value block ───────────────────────────────────────────────────────────
    # NaN = no workflow attempts (undefined ratio) → neutral 0.5, not worst-case
    ws = pd.to_numeric(d.get("workflow_success_30d", pd.Series(np.nan, index=d.index)),
                       errors="coerce").fillna(0.5)
    value_decline = (1.0 - ws).clip(0.0, 1.0)

    # ── Service block ─────────────────────────────────────────────────────────
    # NaN = no recorded unresolved impact → 0 (no penalty)
    hrs = pd.to_numeric(d.get("unresolved_customer_impact_hours_90d",
                               pd.Series(np.nan, index=d.index)),
                         errors="coerce").fillna(0.0)
    impact_norm = (hrs / RULE_CONFIG["normalization"]["impact_hours_ceiling"]).clip(0.0, 1.0)

    sla = pd.to_numeric(d.get("repeat_sla_breaches_90d", pd.Series(np.nan, index=d.index)),
                         errors="coerce").fillna(0.0)
    sla_norm = (sla / RULE_CONFIG["normalization"]["sla_breaches_ceiling"]).clip(0.0, 1.0)

    service_pain = (
        RULE_CONFIG["service_pain_sub_weights"]["impact_norm"] * impact_norm
        + RULE_CONFIG["service_pain_sub_weights"]["sla_norm"] * sla_norm
    )

    # ── Commercial block ──────────────────────────────────────────────────────
    # NaN = no recorded concern → 0 (no penalty)
    budget = pd.to_numeric(d.get("dated_budget_concern", pd.Series(np.nan, index=d.index)),
                            errors="coerce").fillna(0.0).clip(0.0, 1.0)
    commercial_concern = budget

    # ── Final score ───────────────────────────────────────────────────────────
    w = RULE_CONFIG["weights"]
    score = (
        w["deployment_gap"]      * deployment_gap
        + w["value_decline"]     * value_decline
        + w["service_pain"]      * service_pain
        + w["commercial_concern"] * commercial_concern
    )
    return score.clip(0.0, 1.0).values

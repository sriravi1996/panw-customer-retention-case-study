"""
_arr_bridge.py — Pure business-logic functions for ARR bridge computation.

Importable by both analyze_cohorts.py (production) and test_regression_fixtures.py.
No side-effects at import time: no file I/O, no prints, no global state.

Design
------
All components derive from a single OIF subscription slice and a fixed reporting
interval [period_start, period_end].  "Active at date D" means:
    term_start <= D  AND  term_end > D

Opening snapshot  : contracts active at period_start
Movement window   : order_date STRICTLY AFTER period_start (> ps, not >= ps)
                    Contracts renewed on period_start are captured in the opening
                    balance; their acv uplift must not also appear as a movement.
Closing snapshot  : contracts active at period_end

Opening contracts are classified at contract level (not account level):
  retained_still_active   — contract itself is still active at period_end (multi-year)
  retained_renewed        — contract expired but its direct successor is active at period_end
  partial_loss            — contract expired, company retained, no successor in closing
  account_churn           — contract expired, company absent from closing

New-cohort contracts (in closing but not linked to an opening contract) are split:
  new_business    — company has no subscription history before period_start
  reactivation    — company had historical contracts with term_end < period_start (gap)
  intra_expansion — existing-cohort company, brand-new subscription started in period
                    (not a renewal of anything in opening)

GRR = (opening − account_churn − partial_loss + contraction) / opening
NRR = (opening − account_churn − partial_loss + contraction + expansion) / opening

residual = closing − (opening − account_churn − partial_loss
                      + expansion + contraction
                      + new_business + reactivation + intra_expansion)

A non-zero residual after this classification reflects:
  - contracts with multi-hop lineage chains not captured by direct predecessor link
  - ACV changes in retained multi-year contracts (different net_acv despite same contract)
  - data model gaps (missing predecessor_contract_id, etc.)
The lineage_reconciliation table labels these explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PERIOD_START = pd.Timestamp("2024-10-01")
DEFAULT_PERIOD_END   = pd.Timestamp("2025-09-30")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _active_at(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Return contracts active at `date` (term_start <= date < term_end)."""
    return df[(df["term_start"] <= date) & (df["term_end"] > date)]


def _classify_contracts(
    oif_sub_seg: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Classify every opening contract and every unmatched closing contract.

    Returns
    -------
    opening_classified : DataFrame — one row per opening contract with columns:
        contract_id, company_id, opening_net_acv, fate, acv_change,
        successor_contract_id, note
    closing_new : DataFrame — closing contracts not linked to any opening contract,
        classified as new_business / reactivation / intra_expansion
    components : dict with pre-aggregated dollar amounts for bridge construction
    """
    df = oif_sub_seg

    opening = _active_at(df, period_start).copy()
    closing  = _active_at(df, period_end).copy()

    opening_cids = set(opening["contract_id"])
    closing_cids  = set(closing["contract_id"])
    opening_cos   = set(opening["company_id"])
    closing_cos   = set(closing["company_id"])

    # Successor map: predecessor_contract_id → successor contract_id
    # (only maps to contracts active at period_end)
    successor_map: dict[str, str] = (
        closing
        .dropna(subset=["predecessor_contract_id"])
        .set_index("predecessor_contract_id")["contract_id"]
        .to_dict()
    )

    # ── Lineage helpers for multi-hop and gap tracing ────────────────────────
    # full_successor_lookup: predecessor_contract_id → immediate successor anywhere
    # in the dataset.  When a predecessor has multiple successors, prefer ones active
    # at period_end (closing), then break ties by latest order_date.
    _all_succ = df.dropna(subset=["predecessor_contract_id"]).copy()
    if not _all_succ.empty:
        _all_succ["_in_closing"] = _all_succ["contract_id"].isin(closing_cids).astype(int)
        _all_succ = (
            _all_succ.sort_values(["_in_closing", "order_date"], ascending=[False, False])
            .drop_duplicates("predecessor_contract_id", keep="first")
        )
    full_successor_lookup: dict = (
        _all_succ.set_index("predecessor_contract_id")["contract_id"].to_dict()
        if not _all_succ.empty else {}
    )

    # predecessor_lookup: contract_id → predecessor_contract_id
    _has_pred = df.dropna(subset=["predecessor_contract_id"])
    predecessor_lookup: dict = (
        _has_pred.drop_duplicates("contract_id", keep="last")
        .set_index("contract_id")["predecessor_contract_id"]
        .to_dict()
    )

    # Per-contract metadata for lineage checks
    _dedup = df.drop_duplicates("contract_id", keep="last")
    _contract_term_end: dict = {
        cid: pd.Timestamp(te)
        for cid, te in zip(_dedup["contract_id"], _dedup["term_end"])
    }
    _contract_order_type: dict = _dedup.set_index("contract_id")["order_type"].to_dict()
    _closing_net_acv: dict = closing.set_index("contract_id")["net_acv"].to_dict()

    _RENEWAL_TYPES = frozenset({"renewal", "renewal_expansion", "renewal_contraction"})

    def _find_closing_successor(cid, _max=20):
        """Follow full_successor_lookup to find cid's eventual successor in closing."""
        cur, seen = cid, {cid}
        for _ in range(_max):
            nxt = full_successor_lookup.get(cur)
            if nxt is None or nxt in seen:
                return None
            seen.add(nxt)
            if nxt in closing_cids:
                return nxt
            cur = nxt
        return None

    def _trace_predecessor(cid, _max=20):
        """
        Walk predecessor_lookup backward from cid.
        Returns (lineage_type, ancestor_cid):
          "opening_ancestor"    — chain reaches a contract in opening
          "pre_period_ancestor" — chain reaches a contract with term_end < period_start
          "no_predecessor"      — dead-end (no predecessor in data)
          "unresolved"          — cycle or max-depth exceeded
        """
        cur, seen = cid, {cid}
        for _ in range(_max):
            pred = predecessor_lookup.get(cur)
            if pred is None:
                return "no_predecessor", cur
            if pred in seen:
                return "unresolved", cur
            seen.add(pred)
            if pred in opening_cids:
                return "opening_ancestor", pred
            te = _contract_term_end.get(pred)
            if te is not None and te < period_start:
                return "pre_period_ancestor", pred
            cur = pred
        return "unresolved", cur

    # ── Churn evidence lookups ───────────────────────────────────────────────
    # Used to distinguish confirmed cancellation from snapshot-only absence.
    # Priority: confirmed_cancellation > renewal_pending > subsequently_renewed > snapshot_gap_only
    #
    # FIX: confirmed_cancellation requires a cancellation order specifically for THIS contract
    # (via predecessor_contract_id), not just any cancellation anywhere in the company's history.
    # The old approach (company-level cancel_cos set) incorrectly tagged contracts as
    # "confirmed_cancellation" if the account had ANY cancellation order for any other product
    # at any time — inflating confirmed counts by ~17%.
    #
    # A cancellation order cancels a specific predecessor contract. We also apply a grace window:
    # the cancellation order_date must be within 90 days after the predecessor contract's term_end.
    _cancel_df = df[
        (df["order_type"] == "cancellation") &
        df["predecessor_contract_id"].notna()
    ].copy()
    _cancel_df["_order_ts"] = pd.to_datetime(_cancel_df["order_date"])
    # Map: predecessor_contract_id → earliest cancellation order_date for that contract
    _cancel_date_map: dict[str, pd.Timestamp] = (
        _cancel_df.groupby("predecessor_contract_id")["_order_ts"].min().to_dict()
    )

    # Renewals booked within the period but starting after period_end
    _ren_pending_mask = (
        df["order_type"].isin(_RENEWAL_TYPES)
        & (df["order_date"] > period_start)
        & (df["order_date"] <= period_end)
        & df["predecessor_contract_id"].notna()
        & (df["term_start"] > period_end)
    )
    renewal_pending_preds = set(df[_ren_pending_mask]["predecessor_contract_id"])

    # Any successor contract (renewal or new) starting after period_end
    _post_mask = (df["term_start"] > period_end) & df["predecessor_contract_id"].notna()
    post_period_preds = set(df[_post_mask]["predecessor_contract_id"])

    # Period movements: renewals ordered STRICTLY AFTER period_start
    ren_mask = (
        df["order_type"].isin(["renewal", "renewal_expansion", "renewal_contraction"])
        & (df["order_date"] > period_start)      # strict: excludes opening-day bookings
        & (df["order_date"] <= period_end)
        & df["predecessor_contract_id"].notna()
        & df["prior_net_acv"].notna()
    )
    period_ren = df[ren_mask].copy()
    period_ren["acv_change"] = period_ren["net_acv"] - period_ren["prior_net_acv"]

    # Build a lookup: predecessor_contract_id → acv_change in period
    ren_acv_map: dict[str, float] = (
        period_ren
        .set_index("predecessor_contract_id")["acv_change"]
        .to_dict()
    )

    # ── Classify opening contracts ────────────────────────────────────────────
    opening_rows = []
    for _, row in opening.iterrows():
        cid = row["contract_id"]
        co  = row["company_id"]
        acv = row["net_acv"]

        still_active   = cid in closing_cids
        has_successor  = cid in successor_map
        has_movement   = cid in ren_acv_map
        acv_change_val = ren_acv_map.get(cid, 0.0)
        successor_cid  = successor_map.get(cid)

        if still_active:
            # Contract itself is still running — no renewal needed
            if has_movement:
                # Edge case: still active AND a renewal row exists (very rare; log it)
                fate = "retained_still_active_with_movement"
                note = f"contract still active; renewal row found (unusual)"
            else:
                fate = "retained_still_active"
                note = "multi-year contract, no renewal in period"
        elif has_successor:
            # Contract expired; successor active at period_end
            if has_movement:
                if acv_change_val > 0:
                    fate = "retained_expanded"
                elif acv_change_val < 0:
                    fate = "retained_contracted"
                else:
                    fate = "retained_neutral"
                note = f"successor={successor_cid}"
            else:
                # Has a successor in closing but no renewal row in the period window
                # (renewal booked outside period, or before strict cutoff)
                fate = "retained_successor_no_movement"
                note = f"successor={successor_cid}; renewal outside movement window"
        else:
            # No direct successor in closing.  Check multi-hop: does the successor
            # chain eventually reach a closing contract via intermediate renewals?
            eventual_succ = _find_closing_successor(cid)
            if eventual_succ is not None:
                eventual_acv   = float(_closing_net_acv.get(eventual_succ, acv))
                acv_change_val = eventual_acv - float(acv)
                successor_cid  = eventual_succ
                if acv_change_val > 0:
                    fate = "retained_multi_hop_expansion"
                elif acv_change_val < 0:
                    fate = "retained_multi_hop_contraction"
                else:
                    fate = "retained_multi_hop_neutral"
                note = f"multi-hop renewal chain; eventual_successor={eventual_succ}"
            else:
                co_retained = co in closing_cos
                if co_retained:
                    fate = "partial_loss"
                    note = "company retained; this subscription has no successor"
                else:
                    fate = "account_churn"
                    note = "company absent from closing portfolio"

        # Churn evidence: only meaningful for loss fates.
        #
        # Contract-specific lookup: a cancellation is confirmed only when the cancellation
        # order explicitly names this contract (predecessor_contract_id == cid).
        #
        # Evidence is further split by timing relative to the reporting close (period_end):
        #   confirmed_in_period           — cancel_date <= period_end (within reporting window)
        #   confirmed_post_period_followup — period_end < cancel_date <= term_end + 90d
        #     (evidence collected in the 90-day follow-up window after reporting close;
        #      included for completeness but treated as supplementary, not reporting-period fact)
        #
        # Explicit lower bound: cancel_date must be >= period_start.
        # Cancellations booked before the reporting period cannot confirm a loss in this period.
        #
        # Note: this OIF-only lookup may omit evidence from supplementary cancellation event
        # ledgers where the transaction was quarantined but the expiring-contract event survived.
        # If such a ledger is available, it should be joined before this evidence lookup.
        if fate in ("account_churn", "partial_loss"):
            _term_end = _contract_term_end.get(cid, period_end)
            _grace_end = _term_end + pd.Timedelta(days=90)
            _cancel_date = _cancel_date_map.get(cid)
            if _cancel_date is not None and _cancel_date >= period_start and _cancel_date <= period_end:
                churn_evidence = "confirmed_in_period"
            elif _cancel_date is not None and _cancel_date > period_end and _cancel_date <= _grace_end:
                churn_evidence = "confirmed_post_period_followup"
            elif cid in renewal_pending_preds:
                churn_evidence = "renewal_pending"
            elif cid in post_period_preds:
                churn_evidence = "subsequently_renewed"
            else:
                churn_evidence = "snapshot_gap_only"
        else:
            churn_evidence = None

        opening_rows.append({
            "contract_id":          cid,
            "company_id":           co,
            "opening_net_acv":      acv,
            "fate":                 fate,
            "acv_change":           acv_change_val,
            "successor_contract_id": successor_cid,
            "churn_evidence":       churn_evidence,
            "note":                 note,
        })

    opening_classified = pd.DataFrame(opening_rows) if opening_rows else pd.DataFrame(
        columns=["contract_id", "company_id", "opening_net_acv", "fate",
                 "acv_change", "successor_contract_id", "churn_evidence", "note"]
    )

    # ── Classify new-cohort closing contracts ────────────────────────────────
    # Closing contracts not reachable from any opening contract
    linked_to_opening = set(
        row["successor_contract_id"]
        for row in opening_rows
        if row["successor_contract_id"] is not None
    ) | (closing_cids & opening_cids)   # still-active contracts are linked to themselves

    closing_unlinked = closing[~closing["contract_id"].isin(linked_to_opening)].copy()

    # Historical presence check: did this company have ANY contract with term_end < period_start?
    hist_cos = set(
        df[df["term_end"] < period_start]["company_id"].unique()
    )

    closing_rows = []
    for _, row in closing_unlinked.iterrows():
        cid = row["contract_id"]
        co  = row["company_id"]
        acv = row["net_acv"]

        if co in opening_cos:
            # Trace predecessor lineage to distinguish genuine new subscriptions from
            # renewals that had a coverage gap at period_start.
            lineage_type, ancestor = _trace_predecessor(cid)
            ot = _contract_order_type.get(cid, "")
            if lineage_type == "opening_ancestor":
                # Multi-hop should have been resolved on the opening side; flag
                fate = "intra_expansion_unresolved"
                note = f"unexpected opening ancestor={ancestor}; check multi-hop logic"
            elif lineage_type == "pre_period_ancestor":
                # Subscription had a gap at period_start; returned mid-period
                fate = "renewal_gap"
                note = f"subscription gap at opening; chain traces to expired={ancestor}"
            elif lineage_type == "no_predecessor":
                if ot in _RENEWAL_TYPES:
                    # Renewal order type but predecessor absent from dataset
                    fate = "intra_expansion_unresolved"
                    note = f"order_type={ot}; predecessor not in dataset"
                else:
                    fate = "verified_new_subscription"
                    note = "no predecessor chain; new product for existing account"
            else:
                fate = "intra_expansion_unresolved"
                note = f"lineage unresolvable for cid={cid}"
        elif co in hist_cos:
            # Company had prior contracts (lapsed between terms)
            fate = "reactivation"
            note = f"historical customer; absent at period_start, returned"
        else:
            fate = "new_business"
            note = "no prior subscription history"

        closing_rows.append({
            "contract_id":  cid,
            "company_id":   co,
            "closing_net_acv": acv,
            "fate":         fate,
            "note":         note,
        })

    closing_new = pd.DataFrame(closing_rows) if closing_rows else pd.DataFrame(
        columns=["contract_id", "company_id", "closing_net_acv", "fate", "note"]
    )

    # ── Aggregate components ─────────────────────────────────────────────────
    # Issue 1 fix: sum positive and negative acv_changes separately.
    # Never net them — expansion and contraction must be reported independently
    # so that GRR (which includes contraction but not expansion) is correct.
    _MOVEMENT_FATES = frozenset({
        "retained_expanded", "retained_contracted", "retained_neutral",
        "retained_still_active_with_movement",
        "retained_multi_hop_expansion", "retained_multi_hop_contraction", "retained_multi_hop_neutral",
    })
    _chg = opening_classified.loc[opening_classified["fate"].isin(_MOVEMENT_FATES), "acv_change"]
    expansion_arr   = float(_chg[_chg > 0].sum())
    contraction_arr = float(_chg[_chg < 0].sum())

    account_churn_arr = float(
        opening_classified.loc[opening_classified["fate"] == "account_churn", "opening_net_acv"].sum()
    )
    partial_loss_arr = float(
        opening_classified.loc[opening_classified["fate"] == "partial_loss", "opening_net_acv"].sum()
    )

    def _sum_closing(fates):
        if closing_new.empty:
            return 0.0
        return float(closing_new.loc[closing_new["fate"].isin(fates), "closing_net_acv"].sum())

    new_business_arr     = _sum_closing({"new_business"})
    reactivation_arr     = _sum_closing({"reactivation"})
    verified_new_arr     = _sum_closing({"verified_new_subscription"})
    renewal_gap_arr      = _sum_closing({"renewal_gap"})
    intra_unresolved_arr = _sum_closing({"intra_expansion_unresolved"})

    # Opening and closing totals
    opening_arr = float(opening["net_acv"].sum())
    closing_arr  = float(closing["net_acv"].sum())

    # Issue 2 fix: closing ARR from opening-account companies, for account-level NRR.
    # Naturally includes retained, expansion/contraction, multi-hop, verified new
    # subscriptions, and renewal gaps — all within companies that were in the opening
    # portfolio.  Excludes churned accounts (absent from closing).
    closing_arr_opening_accounts = float(
        closing.loc[closing["company_id"].isin(opening_cos), "net_acv"].sum()
    )

    components = {
        "opening_arr":                   opening_arr,
        "closing_arr":                   closing_arr,
        "account_churn_arr":             account_churn_arr,
        "partial_loss_arr":              partial_loss_arr,
        "expansion_arr":                 expansion_arr,
        "contraction_arr":               contraction_arr,
        "new_business_arr":              new_business_arr,
        "reactivation_arr":              reactivation_arr,
        "verified_new_subscription_arr": verified_new_arr,
        "renewal_gap_arr":               renewal_gap_arr,
        "intra_expansion_unresolved_arr": intra_unresolved_arr,
        "closing_arr_opening_accounts":  closing_arr_opening_accounts,
        "n_opening":                     len(opening),
        "n_closing":                     len(closing),
        "n_opening_cos":                 len(opening_cos),
        "n_closing_cos":                 len(closing_cos),
    }
    return opening_classified, closing_new, components


# ─────────────────────────────────────────────────────────────────────────────
# Public: bridge summary
# ─────────────────────────────────────────────────────────────────────────────

def build_period_bridge(
    oif_sub_seg: pd.DataFrame,
    label: str,
    period_start: pd.Timestamp = DEFAULT_PERIOD_START,
    period_end:   pd.Timestamp = DEFAULT_PERIOD_END,
) -> dict:
    """
    Compute a period-anchored ARR bridge from subscription OIF data.

    Components
    ----------
    account_churn        : ACV of opening contracts whose company is gone at period_end
    partial_loss         : ACV of opening contracts whose subscription expired while
                           their account is still present (partial product loss)
    expansion            : sum of positive acv_changes from renewals (NOT netted with contraction)
    contraction          : sum of negative acv_changes from renewals (≤ 0)
    new_business         : ACV from companies with no prior subscription history
    reactivation         : ACV from companies that lapsed and returned (gap > period)
    verified_new_subscription : new product for an existing opening-cohort account
    renewal_gap          : existing subscription that had a gap at period_start and returned
    intra_expansion_unresolved : unresolvable lineage within opening-account companies

    GRR = (opening − account_churn − partial_loss + contraction) / opening
        contraction is already ≤ 0, so this correctly reduces GRR
    NRR (account-level) = closing ARR from opening-account companies / opening
        includes retained + expansion/contraction + verified new + renewal gaps
    contract_nrr = (opening − losses + contraction + expansion) / opening
        renewal lineage only; excludes intra-account new subscriptions

    residual = closing − all_above_components (should be ≈ 0)

    Movement convention
    -------------------
    Renewals with order_date == period_start are already reflected in the opening
    snapshot; they are excluded from the movement window (strict >).
    """
    _, _, C = _classify_contracts(oif_sub_seg, period_start, period_end)

    opening_arr = C["opening_arr"]
    closing_arr  = C["closing_arr"]

    acct_churn    = C["account_churn_arr"]
    part_loss     = C["partial_loss_arr"]
    expansion     = C["expansion_arr"]
    contraction   = C["contraction_arr"]
    new_biz       = C["new_business_arr"]
    reactiv       = C["reactivation_arr"]
    verified_new  = C["verified_new_subscription_arr"]
    renewal_gap   = C["renewal_gap_arr"]
    intra_unres   = C["intra_expansion_unresolved_arr"]

    total_loss = acct_churn + part_loss   # both reduce gross retention

    residual = closing_arr - (
        opening_arr
        - acct_churn - part_loss
        + expansion + contraction
        + new_biz + reactiv
        + verified_new + renewal_gap + intra_unres
    )

    if opening_arr > 0:
        grr = (opening_arr - total_loss + contraction) / opening_arr
        # Account-level NRR: all closing revenue from opening-account companies.
        # Contraction and expansion are implicit in closing_arr_opening_accounts.
        nrr = C["closing_arr_opening_accounts"] / opening_arr
        # Contract-level NRR: direct renewal lineage only (no intra-account additions)
        contract_nrr = (opening_arr - total_loss + contraction + expansion) / opening_arr
    else:
        grr = nrr = contract_nrr = float("nan")

    return {
        "segment":                    label,
        "report_period":              f"{period_start.date()} → {period_end.date()}",
        "n_opening_accounts":         C["n_opening_cos"],
        "n_closing_accounts":         C["n_closing_cos"],
        "opening_arr":                round(opening_arr, 2),
        "account_churn_loss":         round(acct_churn, 2),      # positive = dollars lost
        "partial_loss":               round(part_loss, 2),        # positive = dollars lost
        "expansion":                  round(expansion, 2),
        "contraction":                round(contraction, 2),      # ≤ 0
        "new_business":               round(new_biz, 2),
        "reactivation":               round(reactiv, 2),
        "verified_new_subscription":  round(verified_new, 2),
        "renewal_gap":                round(renewal_gap, 2),
        "intra_expansion_unresolved": round(intra_unres, 2),
        "closing_arr":                round(closing_arr, 2),
        "residual":                   round(residual, 2),
        "grr":          round(grr, 4)          if not np.isnan(grr)          else np.nan,
        "nrr":          round(nrr, 4)          if not np.isnan(nrr)          else np.nan,
        "contract_nrr": round(contract_nrr, 4) if not np.isnan(contract_nrr) else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public: lineage-level reconciliation table
# ─────────────────────────────────────────────────────────────────────────────

def build_lineage_reconciliation(
    oif_sub_seg: pd.DataFrame,
    period_start: pd.Timestamp = DEFAULT_PERIOD_START,
    period_end:   pd.Timestamp = DEFAULT_PERIOD_END,
) -> pd.DataFrame:
    """
    Return a contract-level reconciliation table for the reporting period.

    Opening contracts: one row each, with fate, acv_change, and churn_evidence.
    Unlinked closing contracts: appended with fate and closing_net_acv.

    Fate vocabulary
    ---------------
    retained_still_active           multi-year, no renewal needed
    retained_still_active_with_movement  (unusual; flagged for review)
    retained_expanded               renewed with higher ACV
    retained_contracted             renewed with lower ACV
    retained_neutral                renewed at same ACV
    retained_successor_no_movement  successor exists but renewal outside window
    partial_loss                    subscription cancelled within retained account
    account_churn                   entire account gone
    new_business                    no prior history
    reactivation                    prior customer, lapsed, returned
    intra_expansion                 existing account, brand-new subscription

    churn_evidence (set for partial_loss and account_churn rows only)
    -----------------------------------------------------------------
    confirmed_cancellation  explicit cancellation order exists for this company
    renewal_pending         renewal booked within period but starts after period_end
    subsequently_renewed    successor contract starts after period_end (later renewal)
    snapshot_gap_only       no cancellation, no post-period renewal found; absence
                            inferred from snapshot convention only
    """
    opening_cls, closing_new, C = _classify_contracts(oif_sub_seg, period_start, period_end)

    # Opening side
    open_out = opening_cls[[
        "contract_id", "company_id", "opening_net_acv",
        "fate", "acv_change", "successor_contract_id", "churn_evidence", "note",
    ]].copy()
    open_out["closing_net_acv"] = np.nan

    # Closing new side
    if not closing_new.empty:
        close_out = closing_new.rename(columns={"closing_net_acv": "closing_net_acv"})[[
            "contract_id", "company_id", "closing_net_acv", "fate", "note",
        ]].copy()
        close_out["opening_net_acv"] = np.nan
        close_out["acv_change"]      = np.nan
        close_out["successor_contract_id"] = np.nan
    else:
        close_out = pd.DataFrame(columns=open_out.columns)

    combined = pd.concat([open_out, close_out], ignore_index=True)
    combined["period_start"] = period_start
    combined["period_end"]   = period_end
    return combined

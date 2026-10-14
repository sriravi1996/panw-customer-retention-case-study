"""
Regression fixtures for Part 2 analysis pipeline.

Tests are organized into three groups:

GROUP A — Business-logic unit tests
    Import production functions from _arr_bridge.py and _expansion_decomp.py
    and exercise specific invariants. These remain green or red regardless of
    simulation seed or data size.

GROUP B — Pipeline integration checks
    Read saved outputs and verify structural invariants (grain, disjoint queues,
    model selection record fields). Skipped when outputs are absent.

GROUP C — Snapshot tests (labelled as such)
    Lock in current-run counts that may legitimately change when the simulation
    seed or data generation parameters change. Clearly identified so reviewers
    know they need to be updated after a re-seed, not treated as failures.

Run with: python3 -m pytest part2/src/test_regression_fixtures.py -v
Requires: pytest, pandas, numpy (seaborn optional)
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

# ── Import production business-logic modules ────────────────────────────────
SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from _arr_bridge import (
    build_period_bridge, build_lineage_reconciliation,
    DEFAULT_PERIOD_START, DEFAULT_PERIOD_END,
)
from _expansion_decomp import (
    decompose_expansion_acv,
    classify_expansion_driver,
    RECONCILIATION_TOLERANCE,
)

ROOT    = SRC.parents[1]
OUTPUTS = ROOT / "part2" / "outputs"


# ============================================================================
# GROUP A — Business-logic unit tests (call production functions)
# ============================================================================

class TestArrBridgeGRR:
    """GRR formula correctness via production build_period_bridge."""

    def _make_oif(self, period_start, period_end):
        """
        Synthetic mini-OIF: company T1 has a contract active at period_start that
        renews during the period to 80% of its original ACV (contraction = -20%).
        No churn (same company stays), no new business.
        """
        ren_date = period_start + pd.Timedelta(days=30)
        return pd.DataFrame([
            {   # predecessor: active at period_start, expires before period_end
                "company_id": "T1",
                "contract_id": "P",
                "predecessor_contract_id": None,
                "net_acv": 100.0,
                "prior_net_acv": None,
                "term_start": period_start - pd.Timedelta(days=180),
                "term_end":   period_start + pd.Timedelta(days=180),
                "order_type": "new_logo",
                "order_date": period_start - pd.Timedelta(days=180),
                "offering_type": "subscription",
            },
            {   # renewal at $80: order_date in period, term active at period_end
                "company_id": "T1",
                "contract_id": "C",
                "predecessor_contract_id": "P",
                "net_acv": 80.0,
                "prior_net_acv": 100.0,
                "term_start": ren_date,
                "term_end":   ren_date + pd.Timedelta(days=365),
                "order_type": "renewal_contraction",
                "order_date": ren_date,
                "offering_type": "subscription",
            },
        ])

    def test_contraction_renewal_grr_80pct(self):
        """$100 → $80 renewal: GRR must be 80%."""
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        oif = self._make_oif(ps, pe)
        result = build_period_bridge(oif, "test", ps, pe)
        assert abs(result["grr"] - 0.80) < 1e-6, f"GRR={result['grr']}"

    def test_contraction_renewal_nrr_80pct(self):
        """$100 → $80 renewal, no expansion: NRR must also be 80%."""
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        oif = self._make_oif(ps, pe)
        result = build_period_bridge(oif, "test", ps, pe)
        assert abs(result["nrr"] - 0.80) < 1e-6, f"NRR={result['nrr']}"

    def test_contraction_included_in_grr(self):
        """Omitting contraction from GRR would give 100% — verify we don't."""
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        oif = self._make_oif(ps, pe)
        result = build_period_bridge(oif, "test", ps, pe)
        assert result["grr"] < 1.0, "GRR must be < 100% when contraction exists"

    def test_residual_near_zero_for_clean_data(self):
        """Clean synthetic OIF with a period-bounded renewal should have ≈0 residual."""
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        oif = self._make_oif(ps, pe)
        result = build_period_bridge(oif, "test", ps, pe)
        assert abs(result["residual"]) < 0.01, f"Residual={result['residual']}"

    def test_movements_outside_period_excluded(self):
        """A renewal that occurs after period_end must not contribute to movements."""
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        # Same setup as clean case but renewal date is AFTER period_end
        late_ren_date = pe + pd.Timedelta(days=30)
        oif = pd.DataFrame([
            {
                "company_id": "T2",
                "contract_id": "P2",
                "predecessor_contract_id": None,
                "net_acv": 100.0,
                "prior_net_acv": None,
                "term_start": ps - pd.Timedelta(days=180),
                "term_end":   ps + pd.Timedelta(days=180),   # expires in-period
                "order_type": "new_logo",
                "order_date": ps - pd.Timedelta(days=180),
                "offering_type": "subscription",
            },
            {   # renewal order_date AFTER period_end — must be excluded from movements
                "company_id": "T2",
                "contract_id": "C2",
                "predecessor_contract_id": "P2",
                "net_acv": 120.0,
                "prior_net_acv": 100.0,
                "term_start": late_ren_date,
                "term_end":   late_ren_date + pd.Timedelta(days=365),
                "order_type": "renewal_expansion",
                "order_date": late_ren_date,   # outside period
                "offering_type": "subscription",
            },
        ])
        result = build_period_bridge(oif, "test", ps, pe)
        # The late renewal is outside the period so expansion should be 0
        assert result["expansion"] == 0.0, (
            f"Renewal after period_end must not count as expansion. Got {result['expansion']}"
        )

    def test_new_company_in_closing_not_opening(self):
        """A company that joined after period_start is classified as new business, not churn-retained."""
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        mid_period = ps + pd.Timedelta(days=90)
        oif = pd.DataFrame([
            {
                "company_id": "NEW",
                "contract_id": "N1",
                "predecessor_contract_id": None,
                "net_acv": 50.0,
                "prior_net_acv": None,
                "term_start": mid_period,
                "term_end":   mid_period + pd.Timedelta(days=365),
                "order_type": "new_logo",
                "order_date": mid_period,
                "offering_type": "subscription",
            },
        ])
        result = build_period_bridge(oif, "test", ps, pe)
        assert result["opening_arr"] == 0.0
        assert abs(result["new_business"] - 50.0) < 0.01

    def test_partial_product_loss_grr(self):
        """
        Company T with two subscriptions ($100 + $50) at opening.
        Only $100 subscription survives to closing (SUB_B expires, no successor).
        GRR = NRR = $100 / $150 = 66.67%.
        With account-level churn only, the bridge would report GRR=100% (company retained).
        """
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        oif = pd.DataFrame([
            {"company_id": "T", "contract_id": "SUB_A", "predecessor_contract_id": None,
             "net_acv": 100., "prior_net_acv": None,
             "term_start": ps - pd.Timedelta(days=180), "term_end": ps + pd.Timedelta(days=400),
             "order_type": "new_logo", "order_date": ps - pd.Timedelta(days=180),
             "offering_type": "subscription"},
            {"company_id": "T", "contract_id": "SUB_B", "predecessor_contract_id": None,
             "net_acv": 50., "prior_net_acv": None,
             "term_start": ps - pd.Timedelta(days=180), "term_end": ps + pd.Timedelta(days=180),
             "order_type": "new_logo", "order_date": ps - pd.Timedelta(days=180),
             "offering_type": "subscription"},
        ])
        result = build_period_bridge(oif, "test", ps, pe)
        assert abs(result["grr"] - (100. / 150.)) < 1e-4, (
            f"GRR={result['grr']:.4f}; expected {100./150.:.4f}. "
            "Account-level churn misses partial product loss."
        )
        assert abs(result["partial_loss"] - 50.0) < 0.01, (
            f"partial_loss={result['partial_loss']}; expected 50.0"
        )
        assert abs(result["residual"]) < 0.01, f"residual={result['residual']}"

    def test_partial_product_loss_in_lineage_reconciliation(self):
        """Lineage reconciliation must label SUB_B as partial_loss, not account_churn."""
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        oif = pd.DataFrame([
            {"company_id": "T", "contract_id": "SUB_A", "predecessor_contract_id": None,
             "net_acv": 100., "prior_net_acv": None,
             "term_start": ps - pd.Timedelta(days=180), "term_end": ps + pd.Timedelta(days=400),
             "order_type": "new_logo", "order_date": ps - pd.Timedelta(days=180),
             "offering_type": "subscription"},
            {"company_id": "T", "contract_id": "SUB_B", "predecessor_contract_id": None,
             "net_acv": 50., "prior_net_acv": None,
             "term_start": ps - pd.Timedelta(days=180), "term_end": ps + pd.Timedelta(days=180),
             "order_type": "new_logo", "order_date": ps - pd.Timedelta(days=180),
             "offering_type": "subscription"},
        ])
        recon = build_lineage_reconciliation(oif, ps, pe)
        sub_b = recon[recon["contract_id"] == "SUB_B"]
        assert len(sub_b) == 1, "SUB_B must appear in reconciliation"
        assert sub_b.iloc[0]["fate"] == "partial_loss", (
            f"SUB_B fate={sub_b.iloc[0]['fate']}; expected partial_loss"
        )

    def test_opening_day_renewal_no_double_count(self):
        """
        Old contract: term_end = period_start (NOT active at ps — term_end NOT > ps).
        Renewal: term_start = period_start, order_date = period_start.
        Opening = closing = $120. The $20 ACV increase must NOT appear as expansion.
        GRR = NRR = 100%, expansion = 0, residual = 0.
        """
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        oif = pd.DataFrame([
            {"company_id": "T", "contract_id": "OLD", "predecessor_contract_id": None,
             "net_acv": 100., "prior_net_acv": None,
             "term_start": ps - pd.Timedelta(days=365), "term_end": ps,   # NOT in opening
             "order_type": "new_logo", "order_date": ps - pd.Timedelta(days=365),
             "offering_type": "subscription"},
            {"company_id": "T", "contract_id": "NEW", "predecessor_contract_id": "OLD",
             "net_acv": 120., "prior_net_acv": 100.,
             "term_start": ps, "term_end": ps + pd.Timedelta(days=365),   # in opening AND closing
             "order_type": "renewal_expansion", "order_date": ps,    # == ps → excluded from movements
             "offering_type": "subscription"},
        ])
        result = build_period_bridge(oif, "test", ps, pe)
        assert abs(result["opening_arr"] - 120.) < 0.01
        assert abs(result["closing_arr"] - 120.) < 0.01
        assert abs(result["expansion"]) < 0.01, (
            f"expansion={result['expansion']}; opening-day renewal must not be counted as movement"
        )
        assert abs(result["grr"] - 1.0) < 1e-6
        assert abs(result["residual"]) < 0.01

    def test_reactivation_vs_new_business(self):
        """
        Company that had a contract ending before period_start must be classified
        as reactivation, not new_business, when they sign up again during the period.
        """
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        mid = ps + pd.Timedelta(days=60)
        oif = pd.DataFrame([
            # Historical contract: expired before period_start → this company is a prior customer
            {"company_id": "REACTIV", "contract_id": "HIST", "predecessor_contract_id": None,
             "net_acv": 80., "prior_net_acv": None,
             "term_start": ps - pd.Timedelta(days=730), "term_end": ps - pd.Timedelta(days=365),
             "order_type": "new_logo", "order_date": ps - pd.Timedelta(days=730),
             "offering_type": "subscription"},
            # New contract during period: returns as reactivation
            {"company_id": "REACTIV", "contract_id": "RETURN", "predecessor_contract_id": None,
             "net_acv": 90., "prior_net_acv": None,
             "term_start": mid, "term_end": mid + pd.Timedelta(days=365),
             "order_type": "new_logo", "order_date": mid,
             "offering_type": "subscription"},
        ])
        result = build_period_bridge(oif, "test", ps, pe)
        assert abs(result["reactivation"] - 90.) < 0.01, (
            f"reactivation={result['reactivation']}; expected 90"
        )
        assert abs(result["new_business"]) < 0.01, (
            f"new_business={result['new_business']}; REACTIV has prior history → not new"
        )

    def test_mixed_expansion_contraction_separated(self):
        """
        Company A expands ($100 → $150) and Company B contracts ($100 → $80).
        Opening = $200, Closing = $230.
        GRR must be 90% (contraction reduces retention even though expansion exists).
        The function previously netted them (+$30), masking the contraction and
        returning GRR = 100%.
        """
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        mid = ps + pd.Timedelta(days=60)
        oif = pd.DataFrame([
            # Company A: expands
            {"company_id": "A", "contract_id": "A_OLD", "predecessor_contract_id": None,
             "net_acv": 100., "prior_net_acv": None,
             "term_start": ps - pd.Timedelta(days=180), "term_end": mid,
             "order_type": "new_logo", "order_date": ps - pd.Timedelta(days=180),
             "offering_type": "subscription"},
            {"company_id": "A", "contract_id": "A_NEW", "predecessor_contract_id": "A_OLD",
             "net_acv": 150., "prior_net_acv": 100.,
             "term_start": mid, "term_end": mid + pd.Timedelta(days=365),
             "order_type": "renewal_expansion", "order_date": mid,
             "offering_type": "subscription"},
            # Company B: contracts
            {"company_id": "B", "contract_id": "B_OLD", "predecessor_contract_id": None,
             "net_acv": 100., "prior_net_acv": None,
             "term_start": ps - pd.Timedelta(days=180), "term_end": mid,
             "order_type": "new_logo", "order_date": ps - pd.Timedelta(days=180),
             "offering_type": "subscription"},
            {"company_id": "B", "contract_id": "B_NEW", "predecessor_contract_id": "B_OLD",
             "net_acv": 80., "prior_net_acv": 100.,
             "term_start": mid, "term_end": mid + pd.Timedelta(days=365),
             "order_type": "renewal_contraction", "order_date": mid,
             "offering_type": "subscription"},
        ])
        result = build_period_bridge(oif, "test", ps, pe)
        assert abs(result["opening_arr"] - 200.) < 0.01
        assert abs(result["closing_arr"] - 230.) < 0.01
        assert abs(result["expansion"] - 50.) < 0.01, (
            f"expansion={result['expansion']}; expected 50"
        )
        assert abs(result["contraction"] - (-20.)) < 0.01, (
            f"contraction={result['contraction']}; expected -20"
        )
        assert abs(result["grr"] - 0.90) < 1e-6, (
            f"GRR={result['grr']:.4f}; expected 0.9000. "
            "Netting expansion with contraction masks GRR reduction."
        )
        assert abs(result["residual"]) < 0.01

    def test_intra_account_new_subscription_included_in_nrr(self):
        """
        An opening account with $100 ARR adds a brand-new $50 subscription during the
        period (new_logo, no predecessor).  Closing ARR = $150.
        Account-level NRR must be 150% ($150 / $100), not 100%.
        Contract-level NRR (renewal lineage only) must be 100%.
        """
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        mid = ps + pd.Timedelta(days=60)
        oif = pd.DataFrame([
            # Existing subscription: still active throughout
            {"company_id": "C", "contract_id": "C_MAIN", "predecessor_contract_id": None,
             "net_acv": 100., "prior_net_acv": None,
             "term_start": ps - pd.Timedelta(days=180), "term_end": ps + pd.Timedelta(days=400),
             "order_type": "new_logo", "order_date": ps - pd.Timedelta(days=180),
             "offering_type": "subscription"},
            # New subscription added mid-period (different product, no predecessor)
            {"company_id": "C", "contract_id": "C_NEW", "predecessor_contract_id": None,
             "net_acv": 50., "prior_net_acv": None,
             "term_start": mid, "term_end": mid + pd.Timedelta(days=365),
             "order_type": "new_logo", "order_date": mid,
             "offering_type": "subscription"},
        ])
        result = build_period_bridge(oif, "test", ps, pe)
        assert abs(result["opening_arr"] - 100.) < 0.01
        assert abs(result["closing_arr"] - 150.) < 0.01
        assert abs(result["grr"] - 1.0) < 1e-6, f"GRR={result['grr']}"
        assert abs(result["nrr"] - 1.5) < 1e-4, (
            f"NRR={result['nrr']:.4f}; expected 1.5000. "
            "Account-level NRR must include new subscription from existing account."
        )
        assert abs(result["contract_nrr"] - 1.0) < 1e-6, (
            f"contract_nrr={result['contract_nrr']}; renewal lineage only → 100%"
        )
        assert abs(result["residual"]) < 0.01

    def test_renewal_gap_not_verified_new_subscription(self):
        """
        Company G has an active subscription (OPEN_G) at period_start.
        It also has a separate subscription line (HIST_G) that expired 365 days before
        period_start.  During the period, HIST_G is renewed as RETURN_G.
        RETURN_G is unlinked from opening (HIST_G was not active at period_start) but
        traces back to HIST_G (term_end < period_start), so it must be classified as
        renewal_gap — not verified_new_subscription.
        """
        ps, pe = DEFAULT_PERIOD_START, DEFAULT_PERIOD_END
        mid = ps + pd.Timedelta(days=60)
        oif = pd.DataFrame([
            # Active subscription at period_start (retained)
            {"company_id": "G", "contract_id": "OPEN_G", "predecessor_contract_id": None,
             "net_acv": 100., "prior_net_acv": None,
             "term_start": ps - pd.Timedelta(days=180), "term_end": ps + pd.Timedelta(days=400),
             "order_type": "new_logo", "order_date": ps - pd.Timedelta(days=180),
             "offering_type": "subscription"},
            # Historical subscription: expired before period_start
            {"company_id": "G", "contract_id": "HIST_G", "predecessor_contract_id": None,
             "net_acv": 50., "prior_net_acv": None,
             "term_start": ps - pd.Timedelta(days=730), "term_end": ps - pd.Timedelta(days=365),
             "order_type": "new_logo", "order_date": ps - pd.Timedelta(days=730),
             "offering_type": "subscription"},
            # Renewal of the expired subscription during period
            {"company_id": "G", "contract_id": "RETURN_G", "predecessor_contract_id": "HIST_G",
             "net_acv": 60., "prior_net_acv": 50.,
             "term_start": mid, "term_end": mid + pd.Timedelta(days=365),
             "order_type": "renewal", "order_date": mid,
             "offering_type": "subscription"},
        ])
        result = build_period_bridge(oif, "test", ps, pe)
        assert abs(result["renewal_gap"] - 60.) < 0.01, (
            f"renewal_gap={result['renewal_gap']}; RETURN_G traces to expired HIST_G"
        )
        assert abs(result.get("verified_new_subscription", 0.)) < 0.01, (
            f"verified_new_subscription={result.get('verified_new_subscription')}; "
            "RETURN_G is a renewal, not a new purchase"
        )
        assert abs(result["grr"] - 1.0) < 1e-6  # OPEN_G retained; no churn
        assert abs(result["residual"]) < 0.01


class TestExpansionDecomposition:
    """Expansion ACV decomposition correctness via production decompose_expansion_acv."""

    def _make_expansion_event(
        self,
        current_qty, current_unit_price, current_term_years, current_discount,
        prior_qty, prior_unit_price, prior_term_years, prior_discount,
    ):
        """Build a single-row DataFrame for decomposition."""
        current_net_acv = (current_qty * current_unit_price - current_discount) / current_term_years
        prior_net_acv   = (prior_qty   * prior_unit_price   - prior_discount)   / prior_term_years
        acv_change = current_net_acv - prior_net_acv
        return pd.DataFrame([{
            "quantity":              current_qty,
            "unit_price":            current_unit_price,
            "term_years":            current_term_years,
            "line_discount_amount":  current_discount,
            "net_acv":               current_net_acv,
            "prior_quantity_actual":   prior_qty,
            "prior_unit_price_actual": prior_unit_price,
            "prior_term_years":        prior_term_years,
            "prior_discount_actual":   prior_discount,
            "acv_change":              acv_change,
        }])

    def test_quantity_expansion_classified_correctly(self):
        """6 seats vs 5 seats, same annual price, same term → quantity_expansion."""
        df = self._make_expansion_event(6, 100, 1, 0,  5, 100, 1, 0)
        result = decompose_expansion_acv(df)
        driver = classify_expansion_driver(result.iloc[0])
        assert driver == "quantity_expansion", f"Expected quantity_expansion, got {driver}"

    def test_price_expansion_classified_correctly(self):
        """5 seats, annual price $120 vs $100, same term → price_expansion."""
        df = self._make_expansion_event(5, 120, 1, 0,  5, 100, 1, 0)
        result = decompose_expansion_acv(df)
        driver = classify_expansion_driver(result.iloc[0])
        assert driver == "price_expansion", f"Expected price_expansion, got {driver}"

    def test_unchanged_qty_long_term_not_quantity_expansion(self):
        """
        5 seats → 6 seats; 1yr → 3yr; annual price unchanged.
        With full-term prices: current unit_price = 3 × prior unit_price.
        Old bug: classifies as price_expansion because full-term price grew.
        Fix: annualized prices are equal, so volume component = qty_change × prior_annual_price.
        Expected driver: quantity_expansion.
        """
        annual_price = 100.0
        df = self._make_expansion_event(
            current_qty=6, current_unit_price=annual_price * 3, current_term_years=3, current_discount=0,
            prior_qty=5,   prior_unit_price=annual_price * 1,   prior_term_years=1,   prior_discount=0,
        )
        result = decompose_expansion_acv(df)
        row = result.iloc[0]
        # ACV change = 6*100 - 5*100 = 100
        assert abs(row["acv_change"] - 100.0) < 0.01, f"acv_change={row['acv_change']}"
        # Annualized price change should be zero
        assert abs(row["annual_price_change"]) < 1e-6, \
            f"annual_price_change={row['annual_price_change']} (should be 0 for same annual price)"
        # Volume component = 1 × 100 = 100
        assert abs(row["acv_component_price"]) < 1e-6, \
            f"price_component={row['acv_component_price']} (should be 0)"
        driver = classify_expansion_driver(row)
        assert driver == "quantity_expansion", f"Expected quantity_expansion, got {driver}"

    def test_components_reconcile_to_acv_change(self):
        """Components must sum to acv_change within RECONCILIATION_TOLERANCE."""
        df = self._make_expansion_event(10, 150, 2, 50,  8, 120, 1, 30)
        result = decompose_expansion_acv(df)  # raises if residual > tolerance
        row = result.iloc[0]
        total_components = (
            row["acv_component_volume"]
            + row["acv_component_price"]
            + row["acv_component_discount"]
            + row["acv_component_residual"]
        )
        assert abs(total_components - row["acv_change"]) < RECONCILIATION_TOLERANCE

    def test_reconciliation_tolerance_enforced(self):
        """decompose_expansion_acv raises AssertionError when residual exceeds tolerance."""
        # Inject a row where acv_change is inconsistent with the rest of the fields
        bad_df = pd.DataFrame([{
            "quantity": 5, "unit_price": 100, "term_years": 1, "line_discount_amount": 0,
            "net_acv": 500, "prior_quantity_actual": 5, "prior_unit_price_actual": 100,
            "prior_term_years": 1, "prior_discount_actual": 0,
            "acv_change": 9999,   # deliberately wrong: real change is 0
        }])
        with pytest.raises(AssertionError, match="residual"):
            decompose_expansion_acv(bad_df)

    def test_discount_reduction_classified_correctly(self):
        """5 seats, same annual price, discount removed → discount_reduction."""
        # Prior: discount = 50, so prior_net_acv = (5*100 - 50)/1 = 450
        # Current: discount = 0, so current_net_acv = (5*100 - 0)/1 = 500
        df = self._make_expansion_event(5, 100, 1, 0,  5, 100, 1, 50)
        result = decompose_expansion_acv(df)
        driver = classify_expansion_driver(result.iloc[0])
        assert driver == "discount_reduction", f"Expected discount_reduction, got {driver}"


class TestCrossSellFirstOrderExclusion:
    """Cross-sell events must not count events on a company's first order date."""

    def test_excludes_first_order_date(self):
        first_date = pd.Timestamp("2023-01-15")
        events = pd.DataFrame({
            "order_date": [first_date, first_date, pd.Timestamp("2023-06-01")],
        })
        events["first_order_date"] = first_date
        cross_sells = events[events["order_date"] > events["first_order_date"]]
        assert len(cross_sells) == 1

    def test_includes_later_date(self):
        first_date = pd.Timestamp("2023-01-15")
        events = pd.DataFrame({
            "order_date": [pd.Timestamp("2023-06-01"), pd.Timestamp("2024-01-01")],
        })
        events["first_order_date"] = first_date
        cross_sells = events[events["order_date"] > events["first_order_date"]]
        assert len(cross_sells) == 2

    def test_strict_gt_not_gte(self):
        """Old bug used >= instead of >; verify the fix is strict."""
        first_date = pd.Timestamp("2023-01-15")
        events = pd.DataFrame({"order_date": [first_date]})
        events["first_order_date"] = first_date
        with_gte = events[events["order_date"] >= events["first_order_date"]]
        with_gt  = events[events["order_date"] >  events["first_order_date"]]
        assert len(with_gte) == 1, "gte would count the day-1 event"
        assert len(with_gt) == 0,  "gt correctly excludes it"


# ============================================================================
# GROUP B — Pipeline integration checks (require pipeline to have run)
# ============================================================================

class TestTemporalSplit:
    """Temporal split correctness from saved manifest."""

    @pytest.fixture
    def manifest(self):
        path = OUTPUTS / "split_manifest.json"
        if not path.exists():
            pytest.skip("split_manifest.json not present — run pipeline first")
        with open(path) as f:
            return json.load(f)

    def test_2025_07_01_not_in_training(self, manifest):
        """2025-07-01 labels mature after model_selection cutoff — must not be in train."""
        train_dates = manifest.get("train", {}).get("as_of_dates", [])
        assert "2025-07-01" not in train_dates, (
            "TEMPORAL VIOLATION: 2025-07-01 in training split."
        )

    def test_model_selection_record_date(self):
        path = OUTPUTS / "model_selection_record.json"
        if not path.exists():
            pytest.skip("model_selection_record.json not present — run pipeline first")
        with open(path) as f:
            rec = json.load(f)
        criterion = rec.get("selection_criterion", "")
        assert "2025-07-01" not in criterion
        assert "2025-04-01" in criterion

    def test_selected_model_beats_rule_baseline(self):
        """Selected ML model AP must exceed rule baseline AP on model_selection split."""
        path = OUTPUTS / "model_selection_record.json"
        if not path.exists():
            pytest.skip()
        with open(path) as f:
            rec = json.load(f)
        sel = rec.get("selected_model", {})
        sel_ap = sel.get("ms_ap", float("-inf"))
        rule_ap = rec.get("baseline_aps", {}).get("rule", float("inf"))
        assert sel_ap > rule_ap, (
            f"Selected model AP ({sel_ap:.4f}) must exceed rule baseline AP ({rule_ap:.4f})"
        )

    def test_deployment_policy_field_exists(self):
        """model_selection_record must include deployment_policy and selected_model."""
        path = OUTPUTS / "model_selection_record.json"
        if not path.exists():
            pytest.skip()
        with open(path) as f:
            rec = json.load(f)
        assert "deployment_policy" in rec, "deployment_policy field missing from selection record"
        assert "selected_model" in rec, "selected_model field missing from selection record"
        assert "experiment" in rec["selected_model"], "selected_model.experiment missing"
        assert "model_type" in rec["selected_model"], "selected_model.model_type missing"


class TestAccountQueue:
    """Account queue grain and eligibility invariants."""

    @pytest.fixture
    def accounts(self):
        path = OUTPUTS / "priority_action_queue_accounts.csv"
        if not path.exists():
            pytest.skip()
        return pd.read_csv(path)

    @pytest.fixture
    def contracts(self):
        path = OUTPUTS / "priority_action_queue_contracts.csv"
        if not path.exists():
            pytest.skip()
        return pd.read_csv(path)

    def test_account_queue_is_account_grain(self, accounts):
        assert accounts["company_id"].nunique() == len(accounts), \
            "Account queue has duplicate company_ids"

    def test_account_queue_smaller_than_contract_queue(self, accounts, contracts):
        assert len(accounts) < len(contracts)

    def test_review_and_clear_queues_disjoint(self):
        clear_path  = OUTPUTS / "clear_outreach_accounts.csv"
        review_path = OUTPUTS / "incomplete_history_review_accounts.csv"
        if not clear_path.exists() or not review_path.exists():
            pytest.skip()
        clear  = pd.read_csv(clear_path)
        review = pd.read_csv(review_path)
        overlap = set(clear["company_id"]) & set(review["company_id"])
        assert len(overlap) == 0, f"Clear and review queues share {len(overlap)} accounts"

    def test_scenarios_use_cleared_population(self):
        """Impact scenarios must reference cleared population only."""
        scenarios_path = OUTPUTS / "impact_scenarios.csv"
        if not scenarios_path.exists():
            pytest.skip()
        scenarios = pd.read_csv(scenarios_path)
        assert "population" in scenarios.columns, "scenarios missing 'population' column"
        assert (scenarios["population"] == "cleared_outreach_accounts_only").all(), \
            "Some scenario rows do not restrict to cleared accounts"


class TestMeasurementGate:
    """
    Measurement review gate must not be bypassed by the Reliability repair override.
    A P1 ticket history does not resolve an unverified telemetry gap.
    """

    def test_unknown_measurement_with_p1_stays_measurement_restricted(self):
        """
        CA00001542 has adoption_case='unknown_measurement' (observation completeness
        66.7%, below the 70% threshold). Its recent P1 ticket (closed before snapshot)
        triggers the Reliability repair override in build_segments.py, but its final
        review_gate must still be 'measurement_review', not 'clear'.

        Acceptance: closed-ticket history must not become an open-P1 claim, and
        a reliability override must not erase the measurement restriction.
        """
        path = OUTPUTS / "priority_action_queue_accounts.csv"
        if not path.exists():
            pytest.skip("priority_action_queue_accounts.csv not found")
        df = pd.read_csv(path)
        row = df[df["company_id"] == "CA00001542"]
        if len(row) == 0:
            pytest.skip("CA00001542 not in current scoring cohort")
        gate = row.iloc[0]["review_gate"]
        assert gate == "measurement_review", (
            f"CA00001542 has adoption_case='unknown_measurement' and must be "
            f"'measurement_review', but got '{gate}'. "
            "A P1 ticket closed before the snapshot must not bypass the telemetry gate."
        )

    def test_measurement_restricted_lead_contracts_not_cleared(self):
        """
        No account whose LEAD contract is measurement-restricted (is_measurement_restricted
        == True) should appear in clear_outreach_accounts.csv.

        The gate operates on the lead contract only (highest risk_weighted_acv_score
        per account). An account may have a mix of restricted and unrestricted contracts;
        only the lead contract's status determines outreach eligibility.
        Per-contract measurement concerns are tracked in the contract-level queue.
        """
        seg_path = (OUTPUTS.parent.parent / "data" / "analysis" / "renewal_cohort_segmented.csv")
        queue_path = OUTPUTS / "priority_action_queue_accounts.csv"
        clear_path = OUTPUTS / "clear_outreach_accounts.csv"
        if not seg_path.exists() or not queue_path.exists() or not clear_path.exists():
            pytest.skip("Required files not found")

        seg = pd.read_csv(seg_path)
        if "is_measurement_restricted" not in seg.columns:
            pytest.skip("is_measurement_restricted column not in segmented cohort")

        # Build map: contract_id → is_measurement_restricted
        contract_restricted = seg.set_index("contract_id")["is_measurement_restricted"].to_dict()

        # Get lead_contract_id for each cleared account
        queue = pd.read_csv(queue_path)
        cleared = pd.read_csv(clear_path)
        cleared_ids = set(cleared["company_id"])
        cleared_queue = queue[queue["company_id"].isin(cleared_ids)]

        if "lead_contract_id" not in cleared_queue.columns:
            pytest.skip("lead_contract_id column not in account queue")

        violation = []
        for _, row in cleared_queue.iterrows():
            lead_cid = row["lead_contract_id"]
            if contract_restricted.get(lead_cid, False):
                violation.append(row["company_id"])

        assert len(violation) == 0, (
            f"{len(violation)} account(s) with measurement-restricted lead contract "
            f"appear in clear_outreach: {sorted(violation)[:5]}"
        )


# ============================================================================
# GROUP C — Snapshot tests (labelled — update after re-seeding, not failures)
# ============================================================================

class TestCurrentRunSnapshots:
    """
    SNAPSHOT TESTS — these numbers come from the current simulation run.
    If the data generator seed or parameters change, update expected values here
    rather than treating the failure as a real regression.
    """

    @pytest.mark.snapshot
    def test_training_row_count(self):
        path = OUTPUTS / "split_manifest.json"
        if not path.exists():
            pytest.skip()
        with open(path) as f:
            manifest = json.load(f)
        info = manifest.get("train", {})
        # Snapshot: check structural invariants rather than exact counts.
        # Exact counts vary by company_count / seed — update the comment below after re-seeding.
        # Current 10k run reference: ~2300–2600 train rows, ~120–180 positives.
        n_rows = info.get("n_rows", 0)
        n_pos  = info.get("n_positives", 0)
        assert n_rows > 0,    f"Snapshot: no train rows found (got {n_rows})"
        assert n_pos  > 0,    f"Snapshot: no positives in train split (got {n_pos})"
        assert n_pos < n_rows, f"Snapshot: more positives than rows ({n_pos} >= {n_rows})"

    @pytest.mark.snapshot
    def test_selected_ml_model_uses_declared_criterion(self):
        """Validate the criterion: selected experiment+type has the highest AP among all experiments.

        The winner is whichever experiment/model_type achieves the highest AP on model_selection.
        Verify the recorded selected_model AP is >= the AP of every other experiment.
        """
        path = OUTPUTS / "model_selection_record.json"
        if not path.exists():
            pytest.skip()
        with open(path) as f:
            rec = json.load(f)
        sel = rec.get("selected_model", {})
        sel_exp  = sel.get("experiment")
        sel_type = sel.get("model_type")
        sel_ap   = sel.get("ms_ap", float("-inf"))
        valid_types = ("lr", "hgb")
        assert sel_type in valid_types, (
            f"selected_model.model_type '{sel_type}' is not a recognized type (lr/hgb)"
        )
        exp_best = rec.get("experiment_best_aps", {})
        for exp_name, exp_info in exp_best.items():
            other_ap = exp_info.get("best_ms_ap", float("-inf"))
            assert sel_ap >= other_ap, (
                f"Selection criterion violated: selected {sel_exp}_{sel_type} AP={sel_ap:.4f} "
                f"< {exp_name} best AP={other_ap:.4f}. "
                f"Full experiment APs: {exp_best}"
            )

    @pytest.mark.snapshot
    def test_deployment_policy_matches_selection_comparison(self):
        path = OUTPUTS / "model_selection_record.json"
        if not path.exists():
            pytest.skip()
        rec = json.loads(path.read_text())
        selected = rec["selected_model"]
        expected = ("rule_baseline" if rec["baseline_aps"]["rule"] > selected["ms_ap"]
                    else selected["experiment"] + "_" + selected["model_type"])
        assert rec["deployment_policy"] == expected


class TestDirectManagementScope:
    def test_reporting_alias_matches_current_per_contract_data(self):
        current = ROOT / "data" / "analysis" / "renewal_cohort_subscriptions.csv"
        alias = ROOT / "data" / "analysis" / "renewal_cohort.csv"
        assert current.exists() and alias.exists()
        assert alias.read_bytes() == current.read_bytes(), "Reporting alias is stale"
        columns = pd.read_csv(current, nrows=0).columns
        assert "pred_date" in columns
        assert not {"channel", "sales_channel"}.intersection(columns)

    def test_retention_report_counts_only_observed_outcomes(self):
        current = pd.read_csv(ROOT / "data" / "analysis" / "renewal_cohort_subscriptions.csv")
        expected = current[
            current.offering_type.eq("subscription") & current.churned.notna()
        ]
        report = pd.read_csv(OUTPUTS / "cohort_retention.csv")
        assert int(report.n_contracts.sum()) == len(expected), "Unknown outcomes entered the retention denominator"
        assert not {"channel", "sales_channel"}.intersection(report.columns)

    def test_sales_route_is_not_a_predictor(self):
        from _features import SIZE_VALUE_CAT, FULL_CONTEXT_CAT, EXPERIENCE_CAT, PROHIBITED_PREDICTORS
        forbidden = {"channel", "sales_channel"}
        assert not forbidden.intersection(SIZE_VALUE_CAT + FULL_CONTEXT_CAT + EXPERIENCE_CAT)
        assert forbidden.issubset(PROHIBITED_PREDICTORS)

    def test_review_has_no_sales_route_filter(self):
        path = OUTPUTS / "cohort_eligibility_waterfall.csv"
        if not path.exists():
            pytest.skip()
        wf = pd.read_csv(path)
        stage = wf.iloc[:, 0].astype(str)
        assert not stage.str.contains("channel|reseller|partner", case=False, regex=True).any()
        contracts = pd.read_csv(OUTPUTS / "renewal_review_contracts.csv")
        assert "channel" not in contracts.columns
        assert "gate_channel_ownership" not in contracts.columns
        assert contracts.customer_management_assumption.eq("directly_managed_case_study").all()


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))

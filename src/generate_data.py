"""
generate_data.py  (v2 — corrected)
====================================
Corrected synthetic data generator.  Addresses all confirmed bugs from the
Part 1 corrections plan:

Bug fixes
---------
1. ZERO CHURN LABELS  – cancellation now carries expiring_contract_id and
   prior_net_acv so the label join in integrate.py can find it.
2. RECONCILIATION ERRORS – correct TCV/ACV accounting throughout:
   annual_catalog_price → full_term_unit_price → gross_TCV → net_TCV → net_ACV.
   line identity: qty × full_term_unit_price − discount = line_net_amount.
3. ABANDONED DEPS WITH NO TELEMETRY – telemetry is generated for ALL
   deployment archetypes; state is set AFTER the trajectory is built.
4. FUTURE TELEMETRY IN SNAPSHOT – available_at is set correctly so
   integrate.py can filter on it.
5. NON-DETERMINISTIC ORDERING – sorted(set(chosen)) for product selection;
   stage-specific seeded RNGs throughout.

Structural changes
------------------
* 5-year observation window: 2021-10-01 → 2026-10-01.
* Opening installed base: ~40% of accounts have pre-window first purchases.
* parent_orgs.csv + customer_accounts hierarchy (company_id kept as alias).
* product_launch_date enforced (Cortex Cloud: 2025-02-13, etc.).
* Commercial decisions drawn from operational signals, not pre-assigned.
* Multiple historical scoring snapshots supported.
* Support ACV linked to eligible covered product ACV (not sampled independently).

Run
---
    python3 src/generate_data.py              # full scale (70k accounts)
    python3 src/generate_data.py --dev        # dev mode  (2k accounts)
    python3 src/generate_data.py --config PATH
"""

import argparse
import fcntl
from functools import lru_cache
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from faker import Faker

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "eval"


# ── Write-lock and file-hash helpers ─────────────────────────────────────────

_LOCK_FILE = "._generating.lock"

# Explicit inventory of every file whose hash must be recorded in the manifest.
# Any file consumed by downstream steps must appear here so hash-based coherence
# gates can detect same-size overwrites.  Missing entries are a hard failure.
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
    "deployment_team_map.csv",       # consumed by integrate.py
    "account_interactions.csv",
    "renewal_quotes.csv",            # renewal quote proposals (120d before term_end)
    # Note: deployments_prelim.csv is intermediate only and NOT hashed.
})


def _acquire_write_lock(directory: Path) -> object:
    """
    Acquire an OS-backed exclusive write lock via fcntl.flock(LOCK_EX | LOCK_NB).

    Returns the open file object — caller must keep it open for the lifetime of
    the process.  The kernel releases the lock automatically when the fd is closed
    (normal exit, exception, or SIGKILL), so no atexit registration is needed and
    there is no stale-timeout logic that could evict a still-running generator.

    Raises RuntimeError immediately if another process already holds any lock on
    the same file (TOCTOU-free: the flock syscall is atomic).
    """
    lock_path = directory / _LOCK_FILE
    lock_fh = open(lock_path, "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_fh.close()
        raise RuntimeError(
            f"[write-lock] {lock_path} is held by another process.\n"
            "Another generate_data.py run is active — wait for it to finish.\n"
            f"To inspect: lsof {lock_path}"
        )
    return lock_fh


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file's full contents. Returns hex string."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):  # 1 MiB chunks
            h.update(block)
    return h.hexdigest()


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path=None, dev=False):
    cfg_path = path or (ROOT / "config" / "simulation.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if dev:
        cfg["company_count"] = cfg.get("dev_overrides", {}).get("company_count", 2000)
        cfg["dev_mode"] = True
        print(f"[generate] DEV mode: company_count={cfg['company_count']}")
    return cfg


def rng(seed):
    return np.random.default_rng(int(seed))


def _det_seed(key: str) -> int:
    """Return a deterministic 31-bit integer seed from a string key.
    Uses SHA-256, so the result is stable across Python processes (unlike hash()).
    """
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2 ** 31)


# ── Forward customer experience state ────────────────────────────────────────

def _forward_customer_state(seg, archetype, tenure_days, g_state):
    """
    Generate a forward customer experience state for one contract at renewal time.

    Returns a dict with experience signals that drive renewal probability:
      deployment_gap       — 0.0 (fully deployed) to 1.0 (not deployed)
      value_decline        — 0.0 (stable/growing value) to 1.0 (severe decline)
      service_problem      — 0.0 (no issues) to 1.0 (critical unresolved issues)
      commercial_pressure  — 0.0 (no pressure) to 1.0 (high price/budget pressure)
      owner_gap_days       — days since last confirmed owner (0 if current)
      onboarding_delay_days — days beyond first-value milestone (0 if complete)

    These are hidden simulation states stored in eval/ only.
    The archetype shapes the distribution but does not deterministically set values.
    """
    # Base experience state by archetype
    archetype_profiles = {
        "healthy_steady":      dict(dep_gap=0.05, val_dec=0.05, svc_prob=0.05, com_pres=0.05, owner_gap=0),
        "delayed_onboarding":  dict(dep_gap=0.55, val_dec=0.20, svc_prob=0.10, com_pres=0.05, owner_gap=0),
        "gradual_abandonment": dict(dep_gap=0.20, val_dec=0.65, svc_prob=0.15, com_pres=0.15, owner_gap=15),
        "workload_growth":     dict(dep_gap=0.08, val_dec=-0.05, svc_prob=0.05, com_pres=0.05, owner_gap=0),
        "workload_shrinkage":  dict(dep_gap=0.10, val_dec=0.25, svc_prob=0.05, com_pres=0.20, owner_gap=0),
        "seasonal":            dict(dep_gap=0.10, val_dec=0.10, svc_prob=0.05, com_pres=0.05, owner_gap=0),
        "failed_deployment":   dict(dep_gap=0.75, val_dec=0.60, svc_prob=0.30, com_pres=0.10, owner_gap=30),
        "platform_migration":  dict(dep_gap=0.30, val_dec=0.20, svc_prob=0.10, com_pres=0.05, owner_gap=0),
        "champion_turnover":   dict(dep_gap=0.15, val_dec=0.20, svc_prob=0.10, com_pres=0.05, owner_gap=60),
        "budget_pressure":     dict(dep_gap=0.10, val_dec=0.15, svc_prob=0.05, com_pres=0.65, owner_gap=0),
    }
    profile = archetype_profiles.get(archetype, archetype_profiles["healthy_steady"])

    # Add noise — healthy accounts can still cancel; struggling accounts can renew
    noise_scale = 0.15
    dep_gap    = float(np.clip(profile["dep_gap"]   + g_state.normal(0, noise_scale), 0.0, 1.0))
    val_dec    = float(np.clip(profile["val_dec"]   + g_state.normal(0, noise_scale), 0.0, 1.0))
    svc_prob   = float(np.clip(profile["svc_prob"]  + g_state.normal(0, noise_scale*0.5), 0.0, 1.0))
    com_pres   = float(np.clip(profile["com_pres"]  + g_state.normal(0, noise_scale), 0.0, 1.0))
    owner_gap  = max(0, int(profile["owner_gap"] + g_state.normal(0, 15)))

    # Onboarding delay: nonzero if tenure < 6 months and dep_gap is high
    onboard_delay = 0
    if tenure_days < 180 and dep_gap > 0.4:
        onboard_delay = max(0, int(g_state.uniform(30, 90)))

    return {
        "deployment_gap":        dep_gap,
        "value_decline":         val_dec,
        "service_problem":       svc_prob,
        "commercial_pressure":   com_pres,
        "owner_gap_days":        owner_gap,
        "onboarding_delay_days": onboard_delay,
    }


def _experience_interaction_score(state, params):
    """Declared nonlinear experience assumptions; hidden simulator input only.

    Interactions represent combinations of unrealized value and friction. They are
    not fitted to any observed labels and are never supplied as model features.
    Accepts scalar or numpy-array state values for reference-population anchoring.
    """
    d, v = state["deployment_gap"], state["value_decline"]
    service, commercial = state["service_problem"], state["commercial_pressure"]
    return (
        1.2 * d + 0.8 * v + 0.6 * service + 0.3 * commercial
        + params["deployment_value_failure"] * ((d > 0.45) & (v > 0.35))
        + params["deterioration_with_incomplete_reach"] * ((v > 0.55) & (d > 0.15))
        + params["commercial_value_conflict"] * ((commercial > 0.60) & (v > 0.20))
        + params["service_with_deployment_problem"] * ((service > 0.20) & (d > 0.30))
    )


@lru_cache(maxsize=16)
def _reference_experience_intercept(seg, base_rate, scenario_json, mix_json):
    """Anchor mean annual risk on a separate reference population, without labels.

    The reference seed and population are fixed independently of run/test seeds.
    Actual observed rates can differ because of term mix, survival, budgets and
    sibling effects. No outcome or holdout feature is consulted in this step.
    """
    params, mix = json.loads(scenario_json), json.loads(mix_json)
    rg = rng(_det_seed("reference_experience_v1:" + seg))
    archetypes = sorted(mix)
    draws = rg.choice(archetypes, size=int(params["reference_population"]),
                      p=[mix[a] for a in archetypes])
    rows = [_forward_customer_state(seg, str(a), 365, rg) for a in draws]
    states = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    # Reference renewal variation matches the main generator's event noise.
    for k, sd in [("deployment_gap", .08), ("value_decline", .08),
                  ("service_problem", .04), ("commercial_pressure", .08)]:
        states[k] = np.clip(states[k] + rg.normal(0, sd, len(rows)), 0, 1)
    score = _experience_interaction_score(states, params)
    lo, hi = -20., 5.
    for _ in range(70):
        mid = (lo + hi) / 2
        mean_p = np.mean(1 / (1 + np.exp(-(mid + score))))
        if mean_p < base_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _compute_renewal_probability(state, seg, cfg, base_rate, n_family_siblings: int = 0):
    """
    Logistic renewal probability from customer experience state.

    base_rate: annual subscription cancellation rate for this segment.
    n_family_siblings: number of OTHER currently-active products in the same product
        family at this renewal decision date (excludes the product being scored and
        any previously-cancelled siblings).
    Returns probability of CANCELLATION (not renewal).

    Switching-cost assumption (declared simulation design):
        We assume that owning multiple integrated products in the same family creates
        switching costs that modestly reduce cancellation probability for the scored
        product when its own deployment is healthy.  The benefit applies at the
        renewal decision date: a sibling product that has already cancelled provides
        no protection.  The benefit is conditional on low deployment_gap for the
        scored product and can be overcome by deployment failure, service problems,
        or budget pressure.
        This is a simulation assumption, not a measured Palo Alto Networks effect.
        Coefficient: 0.15 logit units per sibling (capped at 3), yielding a maximum
        benefit of ~30–35% reduction in cancellation probability for a fully healthy
        account with 3 active siblings.
    """
    w = cfg.get("experience_weights", {
        "deployment_gap_weight":     0.35,
        "value_decline_weight":      0.30,
        "service_problem_weight":    0.25,
        "commercial_pressure_weight": 0.10,
    })

    # Linear combination of experience signals
    score = (
        w["deployment_gap_weight"]     * state["deployment_gap"] +
        w["value_decline_weight"]      * state["value_decline"] +
        w["service_problem_weight"]    * state["service_problem"] +
        w["commercial_pressure_weight"] * state["commercial_pressure"]
    )

    # Default retains the previous additive scenario. The separate demonstration
    # copy opts into declared interactions, anchored without any outcome labels.
    import math
    scenario = cfg.get("experience_scenario", {})
    if scenario.get("enabled", False):
        intercept = _reference_experience_intercept(
            seg, float(base_rate), json.dumps(scenario, sort_keys=True),
            json.dumps(cfg["deployment_archetypes"], sort_keys=True),
        )
        logit = intercept + float(_experience_interaction_score(state, scenario))
    else:
        base_logit = math.log(base_rate / (1 - base_rate)) if 0 < base_rate < 1 else -3.0
        logit = base_logit + 6.0 * (score - 0.25)

    # Switching-cost benefit: reduces logit (lowers cancellation probability) when
    # the customer actively uses multiple products in the same family.
    # Adoption quality = 1 when deployment is healthy, 0 when fully failed.
    # Problems (high deployment_gap, service_problem, commercial_pressure) reduce
    # the quality signal, letting churn drivers override the platform benefit.
    if n_family_siblings > 0:
        adoption_quality = max(0.0, 1.0 - state["deployment_gap"] * 2.0)
        # Cap siblings at 3; 0.15 logit units per sibling → max ~30-35% reduction
        breadth_benefit = min(n_family_siblings, 3) * 0.15 * adoption_quality
        logit -= breadth_benefit  # negative shift → lower cancellation probability

    prob = 1.0 / (1.0 + math.exp(-logit))
    return float(np.clip(prob, 0.001, 0.999))


# ── Products ──────────────────────────────────────────────────────────────────

def build_products(cfg):
    rows = []
    for p in cfg["products"]:
        rows.append({
            "product_id":           p["id"],
            "product_name":         p["name"],
            "product_family":       p["family"],
            "offering_type":        p["type"],
            "billing_unit":         p["unit"],
            "recurring_flag":       p["recurring"],
            "annual_catalog_price": p["annual_price"],   # per unit per year
            "product_launch_date":  p["launch_date"],
            "retirement_date":      p.get("retirement_date", None),
        })
    return pd.DataFrame(rows)


def product_available(prod, order_date):
    """Return True if the product can be sold on order_date."""
    launch = pd.Timestamp(prod["product_launch_date"])
    retire = prod.get("retirement_date")
    if pd.Timestamp(order_date) < launch:
        return False
    if retire and pd.Timestamp(order_date) > pd.Timestamp(retire):
        return False
    return True


# ── Parent orgs ───────────────────────────────────────────────────────────────

INDUSTRIES = ["Financial Services", "Healthcare", "Technology", "Manufacturing",
              "Retail", "Government", "Energy", "Education",
              "Telecommunications", "Professional Services"]
REGIONS = ["North America", "EMEA", "APAC", "LATAM"]

def build_parent_orgs(n_accounts, cfg, g):
    """Build parent organizations. ~15% of parents own multiple accounts."""
    multi_share = cfg.get("parent_org_share_multi_account", 0.15)
    avg_multi = cfg.get("avg_accounts_per_multi_parent", 2.4)

    fake = Faker(); fake.seed_instance(int(g.integers(1e9)))

    # Estimate parent count
    n_multi_parents = int(n_accounts * multi_share / avg_multi)
    n_single_parents = n_accounts - int(n_multi_parents * avg_multi)
    n_parents = n_single_parents + n_multi_parents

    parents = []
    for i in range(n_parents):
        industry = INDUSTRIES[int(g.integers(len(INDUSTRIES)))]
        region = REGIONS[int(g.integers(len(REGIONS)))]
        is_multi = i < n_multi_parents
        parents.append({
            "parent_org_id":   f"PO{i+1:07d}",
            "parent_name":     fake.company(),
            "industry":        industry,
            "region":          region,
            "is_multi_account": is_multi,
        })
    return pd.DataFrame(parents)


# ── Customer accounts ─────────────────────────────────────────────────────────

def build_customer_accounts(parents_df, cfg, g):
    """One purchasing account per parent (single), 2-4 for multi-account parents."""
    size_names = list(cfg["size_segments"].keys())
    size_shares = [cfg["size_segments"][s]["share"] for s in size_names]
    obs_start = pd.Timestamp(cfg["observation_start"])
    obs_end = pd.Timestamp(cfg["observation_end"])
    opening_start = pd.Timestamp(cfg.get("opening_base_start", "2018-10-01"))
    obs_span = (obs_end - obs_start).days
    pre_span = (obs_start - opening_start).days

    fake = Faker(); fake.seed_instance(int(g.integers(1e9)))

    accounts = []
    acc_id = 1
    avg_multi = cfg.get("avg_accounts_per_multi_parent", 2.4)

    for _, po in parents_df.iterrows():
        n_accts = int(g.integers(2, 5)) if po["is_multi_account"] else 1
        for _ in range(n_accts):
            seg = g.choice(size_names, p=size_shares)
            seg_cfg = cfg["size_segments"][seg]
            emp = int(g.integers(seg_cfg["emp_min"], seg_cfg["emp_max"] + 1))

            # ~40% of accounts have pre-window history (opening installed base)
            if g.random() < 0.40:
                first_purchase = opening_start + timedelta(
                    days=int(g.integers(0, pre_span)))
                account_status = "historical"
            else:
                first_purchase = obs_start + timedelta(
                    days=int(g.integers(0, obs_span)))
                account_status = "new_in_window"

            accounts.append({
                "customer_account_id": f"CA{acc_id:08d}",
                "company_id":          f"CA{acc_id:08d}",   # alias for backward compat
                "parent_org_id":       po["parent_org_id"],
                "account_name":        fake.company(),
                "industry":            po["industry"],
                "region":              po["region"],
                "size_segment":        seg,
                "employee_count":      emp,
                "channel":             g.choice(["direct", "channel", "digital"],
                                                p=[0.35, 0.45, 0.20]),
                "first_purchase_date": first_purchase.date().isoformat(),
                "account_status":      account_status,
            })
            acc_id += 1

    return pd.DataFrame(accounts)


# ── Teams ─────────────────────────────────────────────────────────────────────

TEAM_TYPES = ["cloud_security", "devsecops", "network_security", "soc",
              "endpoint_security", "iam", "compliance", "it_operations"]

def build_teams(accounts_df, cfg, g):
    avg = cfg.get("avg_teams_per_account", 1.5)
    teams = []
    tid = 1
    for _, acc in accounts_df.iterrows():
        n = max(1, int(g.poisson(avg)))
        if acc["size_segment"] in ("enterprise", "large_enterprise"):
            n = max(2, int(g.poisson(avg * 2.5)))
        chosen = list(g.choice(TEAM_TYPES, size=min(n, len(TEAM_TYPES)), replace=False))
        for ttype in sorted(chosen):   # deterministic ordering
            teams.append({
                "team_id":            f"TM{tid:08d}",
                "customer_account_id": acc["customer_account_id"],
                "company_id":         acc["company_id"],
                "team_type":          ttype,
                "team_name":          ttype.replace("_", " ").title(),
            })
            tid += 1
    return pd.DataFrame(teams)


# ── Users ─────────────────────────────────────────────────────────────────────

USER_ROLES = ["economic_buyer", "technical_buyer", "champion",
              "administrator", "end_user"]
ROLE_WEIGHTS = [0.05, 0.15, 0.30, 0.30, 0.20]

def build_users(accounts_df, teams_df, cfg, g):
    avg = cfg.get("avg_users_per_account", 3.0)
    obs_start = pd.Timestamp(cfg["observation_start"])
    obs_end = pd.Timestamp(cfg["observation_end"])
    opening_start = pd.Timestamp(cfg.get("opening_base_start", "2018-10-01"))
    full_span = (obs_end - opening_start).days

    fake = Faker(); fake.seed_instance(int(g.integers(1e9)))
    team_by_acc = teams_df.groupby("customer_account_id")["team_id"].apply(list).to_dict()

    users = []
    uid = 1
    for _, acc in accounts_df.iterrows():
        n = max(2, int(g.poisson(avg)))
        if acc["size_segment"] in ("enterprise", "large_enterprise"):
            n = max(3, int(g.poisson(avg * 3)))
        acc_teams = team_by_acc.get(acc["customer_account_id"], [])
        first_pur = pd.Timestamp(acc["first_purchase_date"])
        # Users must exist before their first order — create them beforehand.
        # window_start is up to 90 days before first_pur so that initial buyers
        # have plausible pre-purchase account provisioning time.
        window_start = max(opening_start, first_pur - timedelta(days=90))
        # Span for users created AFTER first purchase (most users)
        span = max(1, (obs_end - window_start).days)
        # Span that guarantees creation BEFORE first purchase (initial buyer)
        span_pre = max(1, (first_pur - window_start).days + 1)

        for i in range(n):
            role = g.choice(USER_ROLES, p=ROLE_WEIGHTS)
            if i == 0:
                # At least one user (the initial buyer) must pre-exist the first order
                created_offset = int(g.integers(0, span_pre))
            else:
                created_offset = int(g.integers(0, span))
            created_at = window_start + timedelta(days=created_offset)
            active_from = created_at

            # ~10% of users become inactive — but the FIRST user (account admin) is
            # permanent: every account needs at least one valid actor at all times.
            # This avoids "no active user" gaps that force assigning inactive contacts.
            #
            # Consume EXACTLY the same rng draws as before (random() + maybe integers())
            # to preserve downstream determinism, but override the result for i==0.
            _rng_draw = g.random()
            if _rng_draw < 0.10:
                active_days = int(g.integers(60, max(61, (obs_end - created_at).days)))
                active_to_candidate = min(created_at + timedelta(days=active_days), obs_end)
            else:
                active_to_candidate = None
            # Account admin (i==0) is always permanent; others use the rng draw
            active_to = None if i == 0 else active_to_candidate

            assigned_team = None
            if acc_teams and role not in ("economic_buyer",):
                assigned_team = sorted(acc_teams)[int(g.integers(len(acc_teams)))]

            users.append({
                "user_id":         f"U{uid:09d}",
                "customer_account_id": acc["customer_account_id"],
                "company_id":      acc["company_id"],
                "primary_team_id": assigned_team or (sorted(acc_teams)[0] if acc_teams else None),
                "user_role":       role,
                "first_name":      fake.first_name(),
                "last_name":       fake.last_name(),
                "email":           fake.email(),
                "created_at":      created_at.date().isoformat(),
                "active_from":     active_from.date().isoformat(),
                "active_to":       active_to.date().isoformat() if active_to else None,
            })
            uid += 1
    return pd.DataFrame(users)


# ── Financial accounting helpers ──────────────────────────────────────────────

def compute_line_financials(qty, annual_catalog_price, term_years, discount_rate):
    """
    Correct TCV/ACV accounting (Step 2 fix):
      full_term_unit_price = annual_catalog_price × term_years
      gross_TCV            = qty × full_term_unit_price
      line_discount        = gross_TCV × discount_rate
      line_net_amount      = gross_TCV − line_discount   (= net_TCV)
      net_ACV              = line_net_amount / term_years
    """
    full_term_unit_price = annual_catalog_price * term_years
    gross_tcv = qty * full_term_unit_price
    line_discount = round(gross_tcv * discount_rate, 2)
    line_net = round(gross_tcv - line_discount, 2)
    net_acv = round(line_net / term_years, 2) if term_years > 0 else 0.0
    return full_term_unit_price, gross_tcv, line_discount, line_net, net_acv


def term_years_for(seg, cfg, g):
    mix = cfg["term_mix"][seg]
    return int(g.choice([1, 2, 3], p=[mix["yr1"], mix["yr2"], mix["yr3"]]))


def discount_rate_for(seg, g):
    rates = {"smb": (0.0, 0.05), "midmarket": (0.02, 0.10),
             "enterprise": (0.05, 0.18), "large_enterprise": (0.10, 0.28)}
    lo, hi = rates.get(seg, (0.0, 0.05))
    return float(g.uniform(lo, hi))


def classify_renewal(prior_net_acv, new_net_acv):
    """Classify by ACV delta only. Term-length changes alone are retention."""
    if prior_net_acv <= 0:
        return "renewal"
    delta_pct = (new_net_acv - prior_net_acv) / prior_net_acv
    if delta_pct > 0.02:
        return "renewal_expansion"
    elif delta_pct < -0.02:
        return "renewal_contraction"
    return "renewal"


def pick_quantity(prod_id, billing_unit, seg, g):
    """Return quantity appropriate to billing unit and company size."""
    if billing_unit == "device":
        return 1 if seg in ("smb", "midmarket") else int(g.integers(2, 6))
    elif billing_unit == "operator_seat":
        # SOC analysts: dedicated security ops team, not workforce headcount
        base = {"smb": 5, "midmarket": 15, "enterprise": 40, "large_enterprise": 100}
        return max(3, int(abs(g.normal(base[seg], base[seg] * 0.30))))
    elif billing_unit == "user":
        base = {"smb": 50, "midmarket": 300, "enterprise": 2000, "large_enterprise": 10000}
        return max(10, int(abs(g.normal(base[seg], base[seg] * 0.25))))
    elif billing_unit == "endpoint":
        base = {"smb": 200, "midmarket": 1000, "enterprise": 5000, "large_enterprise": 30000}
        return max(50, int(abs(g.normal(base[seg], base[seg] * 0.25))))
    elif billing_unit in ("vcpu", "credit"):
        base = {"smb": 8, "midmarket": 32, "enterprise": 128, "large_enterprise": 512}
        return max(2, int(abs(g.normal(base[seg], base[seg] * 0.25))))
    elif billing_unit == "cloud_account":
        base = {"smb": 5, "midmarket": 25, "enterprise": 100, "large_enterprise": 500}
        return max(1, int(abs(g.normal(base[seg], base[seg] * 0.25))))
    elif billing_unit == "workload":
        base = {"smb": 30, "midmarket": 150, "enterprise": 700, "large_enterprise": 3000}
        return max(5, int(abs(g.normal(base[seg], base[seg] * 0.25))))
    elif billing_unit == "mbps":
        base = {"smb": 100, "midmarket": 500, "enterprise": 2000, "large_enterprise": 10000}
        return max(10, int(abs(g.normal(base[seg], base[seg] * 0.25))))
    elif billing_unit == "tb":
        base = {"smb": 1, "midmarket": 5, "enterprise": 20, "large_enterprise": 100}
        return max(1, int(abs(g.normal(base[seg], base[seg] * 0.25))))
    elif billing_unit == "app":
        base = {"smb": 3, "midmarket": 15, "enterprise": 50, "large_enterprise": 200}
        return max(1, int(abs(g.normal(base[seg], base[seg] * 0.25))))
    elif billing_unit == "tb_ingest":
        base = {"smb": 1, "midmarket": 3, "enterprise": 12, "large_enterprise": 60}
        return max(1, int(abs(g.normal(base[seg], base[seg] * 0.25))))
    elif billing_unit == "asset":
        base = {"smb": 200, "midmarket": 800, "enterprise": 3000, "large_enterprise": 15000}
        return max(50, int(abs(g.normal(base[seg], base[seg] * 0.25))))
    elif billing_unit == "pct_net_acv":
        return 1   # price is a rate; handled separately
    elif billing_unit == "engagement":
        return 1
    return 1


# ── Orders and order items ────────────────────────────────────────────────────

def pick_initial_products(seg, order_date, cfg, prod_map, g):
    """Return deterministically-ordered list of product_ids to purchase."""
    pen = cfg["penetration"]
    family_prods = {}
    for pid, p in prod_map.items():
        if not product_available(p, order_date):
            continue
        fam = p["product_family"]
        family_prods.setdefault(fam, []).append(pid)
    # Sort product lists within each family for determinism
    for fam in family_prods:
        family_prods[fam] = sorted(family_prods[fam])

    boost = {"smb": 1.0, "midmarket": 1.1, "enterprise": 1.3, "large_enterprise": 1.5}
    b = boost.get(seg, 1.0)
    chosen = []
    for fam, prob in sorted(pen.items()):   # sorted for determinism
        if fam not in family_prods:
            continue
        if g.random() < min(prob * b, 0.95):
            prods = family_prods[fam]
            n_pick = 1 if g.random() < 0.70 else min(2, len(prods))
            picks = list(g.choice(prods, size=n_pick, replace=False))
            chosen.extend(picks)

    # Support: attach to first recurring product if any
    recur_prods = [pid for pid in chosen if prod_map[pid]["recurring_flag"]]
    if recur_prods and "Support" in family_prods:
        # Pick one support tier
        sup_tier = sorted(family_prods["Support"])[0]
        chosen.append(sup_tier)

    return sorted(set(chosen))   # FIX #5: sorted(set()) for determinism


def _pick_active_user(candidate_ids, tx_date, user_window, g, fallback=None):
    """Return a user_id who was active at tx_date.

    user_window maps uid → (created_date, active_to_date or None) where both are
    Python date objects.  Comparisons are at DATE granularity to avoid false
    "inactive" results from sub-day time components in ticket/order timestamps.

    Strategy:
      1. Fully active (created <= tx, active_to >= tx or None) → random pick
      2. No active user → most recently active (highest active_to) among created-before
      3. No user created before tx → use ``fallback`` (initial buyer, always permanent)
    """
    from datetime import date as _date
    if isinstance(tx_date, pd.Timestamp):
        tx_d = tx_date.date()
    elif isinstance(tx_date, _date):
        tx_d = tx_date
    else:
        tx_d = pd.Timestamp(tx_date).date()

    # Primary: fully active at tx_date
    active = [
        uid for uid in candidate_ids
        if uid in user_window
        and user_window[uid][0] <= tx_d
        and (user_window[uid][1] is None or user_window[uid][1] >= tx_d)
    ]
    if active:
        return active[int(g.integers(len(active)))]

    # Secondary: most recently active among users created before tx_date
    created_before = [
        uid for uid in candidate_ids
        if uid in user_window and user_window[uid][0] <= tx_d
    ]
    if created_before:
        from datetime import date as _date2
        _max = _date2(9999, 12, 31)
        def latest_active(uid):
            ato = user_window[uid][1]
            return ato if ato is not None else _max
        return max(created_before, key=latest_active)

    # No user existed before tx_date — use original buyer (always permanent per i==0 rule)
    return fallback


def build_orders_and_items(accounts_df, users_df, teams_df, products_df,
                            cfg, g):
    obs_start = pd.Timestamp(cfg["observation_start"])
    obs_end = pd.Timestamp(cfg["observation_end"])
    opening_start = pd.Timestamp(cfg.get("opening_base_start", "2018-10-01"))

    prod_map = products_df.set_index("product_id").to_dict("index")
    user_by_acc = users_df.groupby("customer_account_id")["user_id"].apply(list).to_dict()
    team_by_acc = teams_df.groupby("customer_account_id")["team_id"].apply(list).to_dict()
    churn_rates = cfg.get("subscription_cancellation_rate", cfg.get("annual_logo_churn", {}))

    # Per-user activity window for chronology enforcement
    # user_id → (created_at_date, active_to_date or None)
    # Stored as date objects so comparisons are always at date granularity,
    # avoiding false "inactive" results from sub-day time components in tx timestamps.
    _user_window = {}
    for _, ur in users_df.iterrows():
        ato_raw = ur.get("active_to")
        _user_window[ur["user_id"]] = (
            pd.Timestamp(ur["created_at"]).date(),
            pd.Timestamp(ato_raw).date() if pd.notna(ato_raw) else None,
        )

    # Pre-compute weighted-average archetype churn weight for relative-risk anchoring.
    # relative_risk = archetype_weight / pop_weighted_avg → preserves segment base rates.
    arc_mix     = cfg.get("deployment_archetypes", {})
    arc_weights = cfg.get("churn_risk_weights", {})
    _pop_avg_arc_churn = sum(
        arc_mix.get(a, 0) * arc_weights.get(a, 0.10) for a in arc_weights
    )
    if _pop_avg_arc_churn <= 0:
        _pop_avg_arc_churn = 0.10  # fallback

    # Archetype list for per-account draw
    _arc_names = sorted(arc_mix.keys())
    _arc_probs = [arc_mix[a] for a in _arc_names]

    orders_rows = []
    items_rows = []
    oid = 1
    iid = 1
    cid_box = [1]   # mutable counter shared with _generate_renewal_chain
    lin_id = 1
    acc_archetype_map = {}   # company_id → archetype (used by build_deployments)
    _experience_state_rows = []  # eval-only: hidden experience states per renewal
    _co_base_states = {}         # company_id → base experience state (shared with operational generators)
    # Point-in-time sibling history:
    # (company_id, product_family) → {lineage_id: [(order_date, term_end_or_None, is_cancel), ...]}
    # Entries are appended in chronological order as each order is generated.
    # Sibling count at renewal date d = lineages whose latest entry with order_date <= d
    # is NOT a cancellation and has term_end >= d.  This prevents a product that
    # cancels in year 4 from removing itself from a year-1 sibling count.
    _co_lineage_snapshots = {}

    for _, acc in accounts_df.iterrows():
        seg = acc["size_segment"]
        acc_users = sorted(user_by_acc.get(acc["customer_account_id"], []))
        acc_teams = sorted(team_by_acc.get(acc["customer_account_id"], []))
        if not acc_users:
            continue

        # Assign a stable behavioral archetype to this account (drives renewal churn)
        account_archetype = str(g.choice(_arc_names, p=_arc_probs))
        acc_archetype_map[acc["company_id"]] = account_archetype

        first_purchase = pd.Timestamp(acc["first_purchase_date"])
        initial_order_date = first_purchase

        # Pre-compute base experience state for this company using a deterministic seed.
        # This single draw is shared with the renewal chain (which adds per-event noise)
        # and with telemetry/support/quote generators, ensuring all operational signals
        # are coherent with the same underlying company experience.
        _scoring_dt = pd.Timestamp(cfg.get("scoring_date", cfg["observation_end"]))
        _tenure_at_scoring = max(0, (_scoring_dt - first_purchase).days)
        _base_state = _forward_customer_state(
            seg, account_archetype, _tenure_at_scoring,
            rng(_det_seed(str(cfg["seed"]) + ":" + acc["company_id"] + "_base_exp"))
        )

        # Draw synthetic annual budget from segment lognormal distribution.
        # DESIGN NOTE — budget signals commercial pressure but does NOT constrain spending:
        #   The budget figure is used to raise commercial_pressure in the base experience
        #   state when initial ACV > 80% of drawn budget.  That pressure flows into the
        #   logistic cancellation model so over-budget accounts churn more readily.
        #   However, product selection and quantity generation are not gated by the budget —
        #   a company can buy more than its budget and will simply carry elevated churn risk.
        #   This is a simulation simplification: implementing joint allocation across
        #   products would require a constraint-satisfaction pass before order generation,
        #   which is out of scope for the current simulation design.
        _seg_cfg = cfg.get("size_segments", {}).get(seg, {})
        _bud_med = float(_seg_cfg.get("budget_median", 0.0))
        _bud_p90 = float(_seg_cfg.get("budget_p90", 0.0))
        if _bud_med > 0 and _bud_p90 > _bud_med:
            import math as _math
            _bud_mu = _math.log(_bud_med)
            _bud_sigma = (_math.log(_bud_p90) - _bud_mu) / 1.2816
            _budget_rng = rng(_det_seed(acc["company_id"] + "_budget"))
            _annual_budget = float(_budget_rng.lognormal(_bud_mu, _bud_sigma))
        else:
            _annual_budget = float("inf")

        # Increase commercial pressure when ACV approaches or exceeds budget.
        # Resolved below after items_for_order is assembled.
        _base_state["_annual_budget"] = _annual_budget  # stored temporarily

        _co_base_states[acc["company_id"]] = _base_state

        disc = discount_rate_for(seg, rng(g.integers(1e9)))

        initial_products = pick_initial_products(
            seg, initial_order_date, cfg, prod_map, rng(g.integers(1e9))
        )
        if not initial_products:
            # Fallback: force one product from the highest-penetration family that
            # has an available product on this date.  Every simulated account is a
            # purchasing account; an empty draw is a model artefact, not intent.
            pen = cfg["penetration"]
            best_fam = max(
                (fam for fam, pids in {
                    f: [pid for pid, p in prod_map.items()
                        if p["product_family"] == f and product_available(p, initial_order_date)]
                    for f in pen
                }.items() if pids),
                key=lambda f: pen[f],
                default=None,
            )
            if best_fam is not None:
                candidates = sorted(
                    pid for pid, p in prod_map.items()
                    if p["product_family"] == best_fam
                    and product_available(p, initial_order_date)
                )
                initial_products = [candidates[0]]
            else:
                continue  # no product available on this date at all — genuine skip

        order_id = f"OR{oid:010d}"
        buyer = _pick_active_user(acc_users, initial_order_date, _user_window, g)
        channel = acc["channel"]

        items_for_order = []
        for prod_id in initial_products:
            prod = prod_map[prod_id]
            if not product_available(prod, initial_order_date):
                continue
            is_recurring = prod["recurring_flag"]
            t_yrs = term_years_for(seg, cfg, rng(g.integers(1e9))) if is_recurring else 0
            term_start = initial_order_date
            term_end = term_start + pd.DateOffset(years=t_yrs) if t_yrs > 0 else None

            # Support: base ACV on covered products' net_ACV
            if prod["offering_type"] == "support":
                covered_acv = sum(
                    it.get("_net_acv", 0) for it in items_for_order
                    if prod_map.get(it["product_id"], {}).get("recurring_flag")
                    and prod_map.get(it["product_id"], {}).get("offering_type") != "support"
                )
                qty = 1
                annual_price = covered_acv * prod["annual_catalog_price"]
                if annual_price <= 0:
                    continue
            else:
                qty = pick_quantity(prod_id, prod["billing_unit"], seg,
                                    rng(g.integers(1e9)))
                annual_price = prod["annual_catalog_price"]

            ftu, gross_tcv, line_disc, line_net, net_acv = compute_line_financials(
                qty, annual_price, t_yrs if t_yrs > 0 else 1, disc
            )

            owner_team = acc_teams[int(g.integers(len(acc_teams)))] if acc_teams else None
            contract_id = f"CTR{cid_box[0]:010d}"; cid_box[0] += 1
            lineage_id = f"LIN{lin_id:010d}"; lin_id += 1

            items_for_order.append({
                "order_item_id":          f"OI{iid:012d}",
                "order_id":               order_id,
                "product_id":             prod_id,
                "contract_id":            contract_id,
                "renewal_lineage_id":     lineage_id,
                "predecessor_contract_id": None,
                "owner_team_id":          owner_team,
                "quantity":               qty,
                "annual_catalog_price":   round(annual_price, 4),
                "full_term_unit_price":   round(ftu, 4),
                "unit_price":             round(ftu, 4),  # stored as full-term for line identity
                "line_discount_amount":   line_disc,
                "line_net_amount":        line_net,
                "net_acv":                net_acv,
                "term_start":             term_start.date().isoformat(),
                "term_end":               term_end.date().isoformat() if term_end else None,
                "term_years":             t_yrs,
                "recurring_flag":         is_recurring,
                # internal only — stripped before export
                "_lineage_id":   lineage_id,
                "_term_years":   t_yrs,
                "_prod_id":      prod_id,
                "_seg":          seg,
                "_acc_id":       acc["customer_account_id"],
                "_net_acv":      net_acv,
                "_disc":         disc,
                "_qty":          qty,
                "_annual_price": annual_price,
            })
            iid += 1

        if not items_for_order:
            # pick_initial_products may return only support/services products, which
            # require covered_acv > 0 and therefore produce no items on their own.
            # Fall back to one guaranteed core product (same logic as the empty-list case).
            _pen = cfg["penetration"]
            _excl = {"Support", "Services"}
            _core_fam = max(
                (fam for fam, pids in {
                    f: [pid for pid, p in prod_map.items()
                        if p["product_family"] == f and product_available(p, initial_order_date)]
                    for f in _pen if f not in _excl
                }.items() if pids),
                key=lambda f: _pen[f],
                default=None,
            )
            if _core_fam is not None:
                _core_pid = sorted(
                    pid for pid, p in prod_map.items()
                    if p["product_family"] == _core_fam
                    and product_available(p, initial_order_date)
                )[0]
                _cp = prod_map[_core_pid]
                _t_yrs = term_years_for(seg, cfg, rng(g.integers(int(1e9)))) if _cp["recurring_flag"] else 0
                _qty = pick_quantity(_core_pid, _cp.get("billing_unit", ""), seg, rng(g.integers(int(1e9))))
                _ap = _cp["annual_catalog_price"]
                _ftu, _gtcv, _ld, _ln, _nacv = compute_line_financials(_qty, _ap, _t_yrs if _t_yrs > 0 else 1, disc)
                _ot = acc_teams[int(g.integers(len(acc_teams)))] if acc_teams else None
                _cid = f"CTR{cid_box[0]:010d}"; cid_box[0] += 1
                _lid = f"LIN{lin_id:010d}"; lin_id += 1
                items_for_order.append({
                    "order_item_id": f"OI{iid:012d}", "order_id": order_id,
                    "product_id": _core_pid, "contract_id": _cid,
                    "renewal_lineage_id": _lid, "predecessor_contract_id": None,
                    "owner_team_id": _ot, "quantity": _qty,
                    "annual_catalog_price": round(_ap, 4),
                    "full_term_unit_price": round(_ftu, 4),
                    "unit_price": round(_ftu, 4),
                    "line_discount_amount": _ld, "line_net_amount": _ln, "net_acv": _nacv,
                    "term_start": initial_order_date.date().isoformat(),
                    "term_end": (initial_order_date + pd.DateOffset(years=_t_yrs)).date().isoformat() if _t_yrs > 0 else None,
                    "term_years": _t_yrs, "recurring_flag": _cp["recurring_flag"],
                    "_lineage_id": _lid, "_term_years": _t_yrs, "_prod_id": _core_pid,
                    "_seg": seg, "_acc_id": acc["customer_account_id"],
                    "_net_acv": _nacv, "_disc": disc, "_qty": _qty, "_annual_price": _ap,
                })
                iid += 1
            if not items_for_order:
                continue  # no buildable product on this date — genuine skip

        # Budget enforcement: if total initial ACV exceeds budget, raise commercial pressure.
        # This connects the budget signal to the experience state used for churn.
        _total_initial_acv = sum(it.get("_net_acv", 0) for it in items_for_order)
        _cur_budget = _co_base_states[acc["company_id"]].get("_annual_budget", float("inf"))
        if _cur_budget < float("inf") and _total_initial_acv > 0:
            _budget_util = _total_initial_acv / _cur_budget
            if _budget_util > 0.8:
                # ACV approaching or exceeding budget — add commercial pressure
                _cp_boost = min(0.3, (_budget_util - 0.8) * 1.5)
                _co_base_states[acc["company_id"]]["commercial_pressure"] = float(np.clip(
                    _co_base_states[acc["company_id"]]["commercial_pressure"] + _cp_boost, 0.0, 1.0
                ))

        order_net = round(sum(i["line_net_amount"] for i in items_for_order), 2)
        orders_rows.append({
            "order_id":               order_id,
            "user_id":                buyer,
            "customer_account_id":    acc["customer_account_id"],
            "company_id":             acc["company_id"],
            "order_date":             initial_order_date.date().isoformat(),
            "order_type":             "new_logo",
            "order_status":           "completed",
            "sales_channel":          channel,
            "currency":               "USD",
            "order_net_amount":       order_net,
        })
        items_rows.extend(items_for_order)
        oid += 1

        # ── Renewals and expansions ───────────────────────────────────────────
        # Seed the point-in-time lineage history with each initial order, then
        # process all renewal events for this account in CALENDAR ORDER using a
        # min-heap.  This ensures that when product A's 2023 renewal is evaluated,
        # product B's 2022 renewal has already been committed to the snapshot —
        # regardless of which product was listed first in items_for_order.
        #
        # Each lineage uses its own deterministic RNG (seeded from lineage_id) so
        # the sequence of random draws within each chain is identical regardless of
        # when its events are dispatched from the heap.  Only the sibling count
        # (computed from the snapshot at decision time) changes with ordering.
        import heapq as _heapq
        _heap = []  # (booking_date_isoformat, lineage_id, chain_state)

        _base_annual_churn_map = cfg.get("subscription_cancellation_rate",
                                          cfg.get("annual_logo_churn", {}))

        for _it in items_for_order:
            if not _it.get("recurring_flag") or not _it.get("term_end"):
                continue
            _it_fam = prod_map.get(_it["product_id"], {}).get("product_family", "")
            _it_key = (acc["company_id"], _it_fam)
            # Seed snapshot with initial purchase
            _co_lineage_snapshots.setdefault(_it_key, {}).setdefault(
                _it["_lineage_id"], []
            ).append((initial_order_date, pd.Timestamp(_it["term_end"]), False))
            # Build per-lineage chain state and draw first booking date
            _lin_rng = rng(_det_seed(str(cfg["seed"]) + ":" + _it["_lineage_id"] + "_chain"))
            _chain = {
                "lineage_id":          _it["_lineage_id"],
                "prod_id":             _it["_prod_id"],
                "seed_item":           _it,
                "seg":                 seg,
                "base_annual_churn":   _base_annual_churn_map.get(seg, 0.10),
                "current_term_end":    pd.Timestamp(_it["term_end"]),
                "current_net_acv":     _it["_net_acv"],
                "current_qty":         _it["_qty"],
                "current_annual_price": _it["_annual_price"],
                "current_disc":        _it["_disc"],
                "current_contract_id": _it["contract_id"],
                "current_term_years":  _it.get("_term_years", 1) or 1,
                "rng":                 _lin_rng,
            }
            _first_booking = pd.Timestamp(_it["term_end"]) + timedelta(days=int(_lin_rng.integers(1, 30)))
            if _first_booking < obs_end:
                _heapq.heappush(_heap,
                    (_first_booking.isoformat(), _it["_lineage_id"], _chain))

        _base_exp = _co_base_states.get(acc["company_id"])
        while _heap:
            _date_str, _, _chain = _heapq.heappop(_heap)
            _renewal_date = pd.Timestamp(_date_str)
            _next_date = _one_renewal_step(
                _renewal_date, _chain,
                acc, buyer, channel, cfg, prod_map,
                obs_end, orders_rows, items_rows, oid, iid, cid_box,
                account_archetype, acc_users, _user_window,
                _experience_state_rows, _base_exp, _co_lineage_snapshots,
            )
            if _next_date is not None:
                _heapq.heappush(_heap,
                    (_next_date.isoformat(), _chain["lineage_id"], _chain))

        oid = int(orders_rows[-1]["order_id"].lstrip("OR")) + 1 if orders_rows else oid
        iid = int(items_rows[-1]["order_item_id"].lstrip("OI")) + 1 if items_rows else iid

    internal = ["_lineage_id", "_term_years", "_prod_id", "_seg", "_acc_id",
                "_net_acv", "_disc", "_qty", "_annual_price"]
    items_df = pd.DataFrame(items_rows).drop(columns=internal, errors="ignore")
    exp_state_df = pd.DataFrame(_experience_state_rows)
    # Strip the temporary _annual_budget key before returning (budget enforcement is
    # applied within the loop; it should not appear in external outputs)
    for _st in _co_base_states.values():
        _st.pop("_annual_budget", None)
    return pd.DataFrame(orders_rows), items_df, acc_archetype_map, exp_state_df, _co_base_states


def _one_renewal_step(renewal_order_date, chain_state,
                       acc, buyer, channel, cfg, prod_map,
                       obs_end, orders_rows, items_rows, oid, iid, cid_box,
                       account_archetype, acc_users, user_window,
                       experience_state_rows, base_exp_state, co_lineage_snapshots):
    """
    Process ONE renewal-decision event for a recurring subscription lineage.

    Uses the per-lineage RNG stored in chain_state["rng"] for all draws, so the
    sequence of random numbers within each lineage is identical regardless of the
    order in which events from different lineages are dispatched.

    renewal_order_date: booking date (pre-drawn by caller, 1-30 days after prior term_end)
    chain_state: mutable dict with keys:
        lineage_id, prod_id, seed_item, seg, base_annual_churn,
        current_term_end, current_net_acv, current_qty, current_annual_price,
        current_disc, current_contract_id, current_term_years, rng

    Returns the next booking date (pd.Timestamp) if the subscription renewed so
    the caller can push it back onto the priority queue, or None if it cancelled
    or fell outside the observation window.  chain_state is updated in-place on
    renewal.
    """
    g              = chain_state["rng"]
    lineage_id     = chain_state["lineage_id"]
    prod_id        = chain_state["prod_id"]
    prod           = prod_map[prod_id]
    seg            = chain_state["seg"]
    seed_item      = chain_state["seed_item"]
    base_annual_churn = chain_state["base_annual_churn"]

    current_term_end     = chain_state["current_term_end"]
    current_net_acv      = chain_state["current_net_acv"]
    current_qty          = chain_state["current_qty"]
    current_annual_price = chain_state["current_annual_price"]
    current_disc         = chain_state["current_disc"]
    current_contract_id  = chain_state["current_contract_id"]
    current_term_years   = chain_state["current_term_years"]

    # Re-pick an active user for this renewal date (buyer may have left)
    if acc_users and user_window:
        renewal_buyer = _pick_active_user(acc_users, renewal_order_date, user_window, g,
                                          fallback=buyer)
    else:
        renewal_buyer = buyer

    # Per-renewal-event churn probability.
    # Use the pre-computed base experience state (shared with telemetry/support/quotes)
    # plus small per-event noise, so the renewal decision is coherent with all
    # operational signals for this company.  Falls back to a fresh draw when no
    # base state is available (e.g., in tests).
    if base_exp_state is not None:
        _ns = 0.08  # smaller noise than standalone draw (preserves company-level correlation)
        exp_state = {
            "deployment_gap":      float(np.clip(base_exp_state["deployment_gap"]      + g.normal(0, _ns),       0.0, 1.0)),
            "value_decline":       float(np.clip(base_exp_state["value_decline"]       + g.normal(0, _ns),       0.0, 1.0)),
            "service_problem":     float(np.clip(base_exp_state["service_problem"]     + g.normal(0, _ns * 0.5), 0.0, 1.0)),
            "commercial_pressure": float(np.clip(base_exp_state["commercial_pressure"] + g.normal(0, _ns),       0.0, 1.0)),
            "owner_gap_days":      max(0, int(base_exp_state.get("owner_gap_days", 0)  + g.normal(0, 10))),
            "onboarding_delay_days": base_exp_state.get("onboarding_delay_days", 0),
        }
    else:
        first_purchase_dt = pd.Timestamp(acc["first_purchase_date"])
        tenure_at_renewal = max(0, (renewal_order_date - first_purchase_dt).days)
        exp_state = _forward_customer_state(seg, account_archetype, tenure_at_renewal, g)

    # Point-in-time sibling count: how many OTHER lineages in the same family
    # were active at this renewal decision date?
    # Since events are processed in calendar order (via heap), ALL sibling events
    # before renewal_order_date have already been committed to co_lineage_snapshots.
    # We filter to order_date <= renewal_order_date and check the latest status.
    _prod_family = prod.get("product_family", "")
    _sib_key = (acc["company_id"], _prod_family)
    if co_lineage_snapshots is not None and _sib_key in co_lineage_snapshots:
        _live_n_siblings = 0
        for _lid, _history in co_lineage_snapshots[_sib_key].items():
            if _lid == lineage_id:
                continue
            _known = [(od, te, ic) for od, te, ic in _history if od <= renewal_order_date]
            if _known:
                _last_od, _last_te, _last_cancel = _known[-1]
                if not _last_cancel and _last_te is not None and _last_te >= renewal_order_date:
                    _live_n_siblings += 1
    else:
        _live_n_siblings = 0

    # Annual cancellation probability from experience signals.
    annual_cancel_prob = _compute_renewal_probability(
        exp_state, seg, cfg, base_annual_churn, n_family_siblings=_live_n_siblings
    )

    # Multi-year contracts accumulate annual risk correctly via the geometric formula.
    # Segment caps prevent extreme probabilities even for struggling accounts.
    _seg_max = {"smb": 0.65, "midmarket": 0.45, "enterprise": 0.28, "large_enterprise": 0.15}
    per_renewal_prob = 1.0 - (1.0 - annual_cancel_prob) ** current_term_years
    per_renewal_prob = min(per_renewal_prob, _seg_max.get(seg, 0.55))
    churn_prob = float(np.clip(per_renewal_prob, 0.003, 0.90))

    # Accumulate eval row (hidden state — stored in eval/ only)
    if experience_state_rows is not None:
        experience_state_rows.append({
            "contract_id":          current_contract_id,
            "company_id":           acc["company_id"],
            "product_id":           prod_id,
            "archetype":            account_archetype,
            "deployment_gap":       round(exp_state["deployment_gap"], 4),
            "value_decline":        round(exp_state["value_decline"], 4),
            "service_problem":      round(exp_state["service_problem"], 4),
            "commercial_pressure":  round(exp_state["commercial_pressure"], 4),
            "owner_gap_days":       exp_state["owner_gap_days"],
            "onboarding_delay_days": exp_state["onboarding_delay_days"],
            "computed_cancel_prob": round(churn_prob, 6),
        })

    if g.random() < churn_prob:
        # ── CANCELLATION: carries expiring contract and prior ACV ──────
        new_oid = f"OR{len(orders_rows)+oid:010d}"
        new_cid = f"CTR{cid_box[0]:010d}"
        orders_rows.append({
            "order_id":               new_oid,
            "user_id":                renewal_buyer,
            "customer_account_id":    acc["customer_account_id"],
            "company_id":             acc["company_id"],
            "order_date":             renewal_order_date.date().isoformat(),
            "order_type":             "cancellation",
            "order_status":           "completed",
            "sales_channel":          channel,
            "currency":               "USD",
            "order_net_amount":       0.0,
        })
        items_rows.append({
            "order_item_id":           f"OI{len(items_rows)+iid:012d}",
            "order_id":                new_oid,
            "product_id":              prod_id,
            "contract_id":             new_cid,
            "renewal_lineage_id":      lineage_id,
            "predecessor_contract_id": current_contract_id,
            "owner_team_id":           seed_item["owner_team_id"],
            "quantity":                0,
            "annual_catalog_price":    0.0,
            "full_term_unit_price":    0.0,
            "unit_price":              0.0,
            "line_discount_amount":    0.0,
            "line_net_amount":         0.0,
            "net_acv":                 0.0,
            "prior_net_acv":           current_net_acv,
            "term_start":              renewal_order_date.date().isoformat(),
            "term_end":                current_term_end.date().isoformat(),
            "term_years":              0,
            "recurring_flag":          True,
            "_lineage_id":   lineage_id,
            "_term_years":   0,
            "_prod_id":      prod_id,
            "_seg":          seg,
            "_acc_id":       acc["customer_account_id"],
            "_net_acv":      0.0,
            "_disc":         0.0,
            "_qty":          0,
            "_annual_price": 0.0,
        })
        cid_box[0] += 1
        # Record cancellation in point-in-time history.
        if co_lineage_snapshots is not None:
            co_lineage_snapshots.setdefault(_sib_key, {}).setdefault(
                lineage_id, []
            ).append((renewal_order_date, None, True))
        return None   # chain ends; caller does not push to heap

    # ── RENEWAL (retention / expansion / contraction) ─────────────────
    new_t_yrs = term_years_for(seg, cfg, rng(g.integers(1e9)))
    new_term_end = renewal_order_date + pd.DateOffset(years=new_t_yrs)

    r = g.random()
    if r < 0.12:
        qty_delta   = float(g.uniform(1.05, 1.30))
        price_delta = float(g.uniform(1.00, 1.05))
    elif r < 0.17:
        qty_delta   = float(g.uniform(0.75, 0.98))
        price_delta = float(g.uniform(0.95, 1.00))
    else:
        qty_delta   = float(g.uniform(0.98, 1.02))
        price_delta = float(g.uniform(0.99, 1.02))

    new_qty          = max(1, int(round(current_qty * qty_delta)))
    new_annual_price = round(current_annual_price * price_delta, 4)
    new_disc         = current_disc

    _, _, new_disc_amt, new_line_net, new_net_acv = compute_line_financials(
        new_qty, new_annual_price, new_t_yrs, new_disc
    )

    order_type = classify_renewal(current_net_acv, new_net_acv)

    new_oid = f"OR{len(orders_rows)+oid:010d}"
    new_cid = f"CTR{cid_box[0]:010d}"
    orders_rows.append({
        "order_id":               new_oid,
        "user_id":                renewal_buyer,
        "customer_account_id":    acc["customer_account_id"],
        "company_id":             acc["company_id"],
        "order_date":             renewal_order_date.date().isoformat(),
        "order_type":             order_type,
        "order_status":           "completed",
        "sales_channel":          channel,
        "currency":               "USD",
        "order_net_amount":       new_line_net,
    })
    items_rows.append({
        "order_item_id":           f"OI{len(items_rows)+iid:012d}",
        "order_id":                new_oid,
        "product_id":              prod_id,
        "contract_id":             new_cid,
        "renewal_lineage_id":      lineage_id,
        "predecessor_contract_id": current_contract_id,
        "owner_team_id":           seed_item["owner_team_id"],
        "quantity":                new_qty,
        "annual_catalog_price":    round(new_annual_price, 4),
        "full_term_unit_price":    round(new_annual_price * new_t_yrs, 4),
        "unit_price":              round(new_annual_price * new_t_yrs, 4),
        "line_discount_amount":    new_disc_amt,
        "line_net_amount":         new_line_net,
        "net_acv":                 new_net_acv,
        "prior_net_acv":           current_net_acv,
        "term_start":              renewal_order_date.date().isoformat(),
        "term_end":                new_term_end.date().isoformat(),
        "term_years":              new_t_yrs,
        "recurring_flag":          True,
        "_lineage_id":   lineage_id,
        "_term_years":   new_t_yrs,
        "_prod_id":      prod_id,
        "_seg":          seg,
        "_acc_id":       acc["customer_account_id"],
        "_net_acv":      new_net_acv,
        "_disc":         new_disc,
        "_qty":          new_qty,
        "_annual_price": new_annual_price,
    })
    cid_box[0] += 1

    # Update chain state in-place for next event
    chain_state["current_term_end"]     = pd.Timestamp(new_term_end.date())
    chain_state["current_net_acv"]      = new_net_acv
    chain_state["current_qty"]          = new_qty
    chain_state["current_annual_price"] = new_annual_price
    chain_state["current_contract_id"]  = new_cid
    chain_state["current_term_years"]   = new_t_yrs

    # Append renewal to point-in-time history.
    if co_lineage_snapshots is not None:
        co_lineage_snapshots.setdefault(_sib_key, {}).setdefault(
            lineage_id, []
        ).append((renewal_order_date, chain_state["current_term_end"], False))

    # Draw next booking date and return it so the caller can schedule the event.
    next_booking = chain_state["current_term_end"] + timedelta(days=int(g.integers(1, 30)))
    if next_booking >= obs_end:
        return None  # no more events within observation window
    return next_booking


# ── Deployments ───────────────────────────────────────────────────────────────

RECURRING_FAMILIES = {"SASE", "Prisma", "Cortex", "Strata"}

def build_deployments(accounts_df, teams_df, orders_df, items_df,
                       products_df, cfg, g,
                       company_archetype_map: dict = None):
    """
    company_archetype_map: dict mapping company_id → commercial archetype
    (from build_orders_and_items).  When provided, deployments inherit the
    company's archetype with 75% probability, giving operational and commercial
    signals coherence: a company in gradual_abandonment will have abandoned
    deployments AND higher contract churn.
    """
    obs_start = pd.Timestamp(cfg["observation_start"])
    obs_end = pd.Timestamp(cfg["observation_end"])

    prod_fam = products_df.set_index("product_id")["product_family"].to_dict()
    prod_rec = products_df.set_index("product_id")["recurring_flag"].to_dict()
    prod_map = products_df.set_index("product_id").to_dict("index")  # full product attrs including billing_unit

    # Only first-purchase of each lineage gets a deployment (renewals reuse it)
    dep_items = items_df[
        (items_df["recurring_flag"] == True) &
        (items_df["predecessor_contract_id"].isna()) &
        (items_df["quantity"] > 0)
    ].copy()
    dep_items["product_family"] = dep_items["product_id"].map(prod_fam)
    dep_items = dep_items[dep_items["product_family"].isin(RECURRING_FAMILIES)]

    archetypes = sorted(cfg["deployment_archetypes"].keys())
    archetype_probs = [cfg["deployment_archetypes"][a] for a in archetypes]
    team_by_acc = teams_df.groupby("customer_account_id")["team_id"].apply(list).to_dict()
    order_acc_map = orders_df.set_index("order_id")["customer_account_id"].to_dict()
    # Build company_id from customer_account_id via accounts_df
    acc_to_company = accounts_df.set_index("customer_account_id")["company_id"].to_dict()

    deployments, dep_team_map, ent_dep_map, archetype_labels = [], [], [], []
    did = 1

    for _, item in dep_items.iterrows():
        acc_id = order_acc_map.get(item["order_id"])
        if not acc_id:
            continue
        acc_teams = sorted(team_by_acc.get(acc_id, []))
        if not acc_teams:
            continue

        term_start = pd.Timestamp(item["term_start"])
        # FIX #3: do NOT pre-decide deployment_state before telemetry is generated
        # State is set in build_telemetry based on archetype trajectory
        dep_start = term_start + timedelta(days=int(g.integers(0, 14)))

        # Inherit commercial archetype with 75% probability for coherence
        company_id = acc_to_company.get(acc_id, acc_id)
        commercial_arch = (company_archetype_map or {}).get(company_id)
        if commercial_arch and g.random() < 0.75:
            archetype = commercial_arch
        else:
            archetype = str(g.choice(archetypes, p=archetype_probs))

        dep_id = f"DEP{did:010d}"
        # B6: Planned first-value milestone: 30-90 days from deployment start
        fv_offset = int(g.integers(30, 91))
        planned_fv_date = (dep_start + timedelta(days=fv_offset)).date().isoformat()
        # Whether it was achieved (only meaningful for non-failed archetypes)
        achieved = archetype not in ("failed_deployment", "delayed_onboarding") or g.random() > 0.6
        fv_achieve_offset = fv_offset + int(g.integers(0, 61))  # up to 60 days late
        actual_fv_date = (dep_start + timedelta(days=fv_achieve_offset)).date().isoformat() if achieved else None
        deployments.append({
            "deployment_id":    dep_id,
            "customer_account_id": acc_id,
            "company_id":       acc_id,
            "product_id":       item["product_id"],
            "product_family":   item["product_family"],
            "billing_unit":     prod_map.get(item["product_id"], {}).get("billing_unit", ""),
            "deployment_start": dep_start.date().isoformat(),
            "deployment_end":   None,   # filled in later by telemetry stage
            "deployment_state": "active_paid",   # placeholder; updated after telemetry
            "eligible_units":   max(1, int(item["quantity"])),
            "renewal_lineage_id": item["renewal_lineage_id"],
            "planned_first_value_date":  planned_fv_date,
            "first_value_achieved_date": actual_fv_date,
        })

        n_teams = 1 if g.random() < 0.80 else min(2, len(acc_teams))
        assigned = acc_teams[:n_teams]
        for t in assigned:
            dep_team_map.append({
                "deployment_id":  dep_id,
                "team_id":        t,
                "effective_from": dep_start.date().isoformat(),
                "effective_to":   None,
                "allocation_share": round(1.0 / n_teams, 4),
            })

        ent_dep_map.append({
            "contract_id":    item["contract_id"],
            "deployment_id":  dep_id,
            "product_id":     item["product_id"],
            "allocated_units": item["quantity"],
            "effective_from": dep_start.date().isoformat(),
            "effective_to":   item["term_end"],
        })

        archetype_labels.append({"deployment_id": dep_id, "archetype": archetype})
        did += 1

    return (pd.DataFrame(deployments), pd.DataFrame(dep_team_map),
            pd.DataFrame(ent_dep_map), pd.DataFrame(archetype_labels))


# ── Cancellation lifecycle (must run before telemetry) ────────────────────────

def compute_cancellation_lifetimes(items_df, deps_df, cfg):
    """
    Scan the renewal chains in items_df for any lineage that ends in a
    cancellation and set deployment_end + deployment_state on deps_df BEFORE
    telemetry is generated, so telemetry rows are clipped at the correct date.

    A cancellation item has term_years==0 and net_acv==0.  It carries
    term_end = the EXPIRING contract's term_end (the date coverage ceases).
    We define:
        deployment_end = expiring_term_end + grace_period_days
        deployment_state = "grace_period"  (telemetry continues through grace)

    Returns updated deps_df.
    """
    grace_days = cfg.get("renewal_grace_period_days", 30)
    obs_end    = pd.Timestamp(cfg["observation_end"])

    lin_col = "renewal_lineage_id"
    # Build lineage → deployment_id from deps_df
    if lin_col not in deps_df.columns:
        return deps_df
    lin_to_dep = deps_df.set_index(lin_col)["deployment_id"].to_dict()

    dep_df = deps_df.copy()

    # Find all cancellation items (net_acv=0, term_years=0, has predecessor)
    cancels = items_df[
        items_df["recurring_flag"].astype(bool) &
        items_df["predecessor_contract_id"].notna() &
        (items_df["net_acv"].fillna(0) == 0.0) &
        (items_df["term_years"].fillna(0) == 0)
    ]

    for _, row in cancels.iterrows():
        lin = row.get(lin_col)
        dep_id = lin_to_dep.get(lin)
        if dep_id is None:
            continue
        idx = dep_df.index[dep_df["deployment_id"] == dep_id].tolist()
        if not idx:
            continue
        i = idx[0]
        # term_end on cancellation item = last day of paid coverage
        te_raw = row.get("term_end")
        if not te_raw:
            continue
        te = pd.Timestamp(te_raw)
        actual_grace_end = te + timedelta(days=grace_days)
        # deployment_end is capped at obs_end for telemetry generation
        # (we don't want telemetry weeks beyond the observation window).
        # grace_end_date stores the ACTUAL uncapped end so lifecycle state
        # decisions can correctly distinguish "grace window still open at
        # scoring_date" from "capped by obs_end".
        dep_end = min(actual_grace_end, obs_end)
        dep_df.at[i, "deployment_end"]   = dep_end.date().isoformat()
        dep_df.at[i, "grace_end_date"]   = actual_grace_end.date().isoformat()
        dep_df.at[i, "deployment_state"] = "grace_period"

    return dep_df


# ── Entitlement / deployment continuity across renewals ───────────────────────

def build_entitlement_renewal_map(items_df, ent_dep_df, deployments_df, cfg):
    """
    Extend entitlement_deployment_map to include ALL renewal and cancellation
    contracts, not just the initial purchase.  Also updates deployment
    eligible_units (capacity changes on renewal) and marks cancelled lineages
    as decommissioned after the grace period.

    Returns:
        ent_dep_df_full  — original rows + one row per renewal/cancellation
        deployments_df   — updated deployment_end, deployment_state, eligible_units
    """
    grace_days = cfg.get("renewal_grace_period_days", 30)
    obs_end    = pd.Timestamp(cfg["observation_end"])

    # lineage_id → deployment_id (from initial entitlement rows)
    lineage_to_dep = {}
    lin_col = "renewal_lineage_id" if "renewal_lineage_id" in items_df.columns else None
    if lin_col:
        # Build via items: map each contract_id → lineage, then ent_dep → dep_id
        ctr_to_lin = items_df.set_index("contract_id")[lin_col].to_dict()
        for _, row in ent_dep_df.iterrows():
            lin = ctr_to_lin.get(row["contract_id"])
            if lin:
                lineage_to_dep[lin] = row["deployment_id"]

    # All subsequent contracts (have predecessor → not initial purchase)
    subsequent = items_df[
        items_df["recurring_flag"].astype(bool) &
        items_df["predecessor_contract_id"].notna()
    ].copy()
    if lin_col:
        subsequent["_lineage"] = subsequent[lin_col]

    # Build set of contract_ids that exist in items_df (for validation)
    valid_contract_ids = set(items_df["contract_id"].dropna())

    new_rows = []
    skipped_missing_contract = 0
    skipped_backwards_interval = 0
    # Track the latest contract per lineage for capacity and state updates
    lineage_latest: dict = {}   # lin → {term_end, quantity, is_cancel}

    for _, item in subsequent.iterrows():
        lin = item.get("_lineage") or item.get("renewal_lineage_id")
        dep_id = lineage_to_dep.get(lin)
        if dep_id is None:
            continue

        # Validate contract_id exists (guard against FK drift after cleaning)
        ctr_id = item["contract_id"]
        if ctr_id not in valid_contract_ids:
            skipped_missing_contract += 1
            continue

        is_cancel = (int(item.get("term_years", 0) or 0) == 0 and
                     float(item.get("net_acv", 0) or 0) == 0.0)
        qty = max(0, int(item.get("quantity", 0) or 0))
        term_end_raw   = item.get("term_end")
        term_start_raw = item.get("term_start")

        # For cancellations, term_start is the order date (can be after term_end).
        # Normalise effective_from to min(term_start, term_end) so the interval
        # is never backwards.  Track lineage state BEFORE skipping any rows.
        is_backwards = False
        if term_start_raw and term_end_raw:
            try:
                if str(term_start_raw) > str(term_end_raw):
                    # Cancellation order date after expiry — use term_end as start
                    term_start_raw = term_end_raw
                    is_backwards = True
                    skipped_backwards_interval += 1
            except TypeError:
                pass

        # Always update lineage_latest (even for backwards-interval cancellations)
        # so that deployment state is correctly set to decommissioned.
        cur = lineage_latest.get(lin)
        if cur is None or (term_end_raw and (cur["term_end"] is None or
                                              str(term_end_raw) > str(cur["term_end"]))):
            lineage_latest[lin] = {
                "term_end":  term_end_raw,
                "quantity":  qty,
                "is_cancel": is_cancel,
            }

        new_rows.append({
            "contract_id":     ctr_id,
            "deployment_id":   dep_id,
            "product_id":      item["product_id"],
            "allocated_units": qty,
            "effective_from":  term_start_raw,   # normalised — never > effective_to
            "effective_to":    term_end_raw,
            "is_renewal":      not is_cancel,
            "is_cancellation": is_cancel,
        })

    if skipped_missing_contract > 0 or skipped_backwards_interval > 0:
        print(f"  [entitlement_map] skipped: missing_contract={skipped_missing_contract}  "
              f"backwards_interval={skipped_backwards_interval}")

    # Apply capacity and lifecycle updates to deployments
    dep_df = deployments_df.copy()
    for lin, latest in lineage_latest.items():
        dep_id = lineage_to_dep.get(lin)
        if dep_id is None:
            continue
        idx = dep_df.index[dep_df["deployment_id"] == dep_id].tolist()
        if not idx:
            continue
        i = idx[0]

        if latest["is_cancel"]:
            # Cancelled: deployment enters grace period.
            # deployment_end is capped at obs_end (for telemetry generation).
            # grace_end_date stores the ACTUAL uncapped end for lifecycle decisions.
            # Final state (grace_period vs decommissioned) is resolved in _merge_state()
            # after this function returns, using grace_end_date vs scoring_date.
            te = pd.Timestamp(latest["term_end"]) if latest["term_end"] else obs_end
            actual_grace_end = te + timedelta(days=grace_days)
            dep_end_ts = min(actual_grace_end, obs_end)
            dep_df.at[i, "deployment_end"]   = dep_end_ts.date().isoformat()
            dep_df.at[i, "grace_end_date"]   = actual_grace_end.date().isoformat()
            dep_df.at[i, "deployment_state"] = "grace_period"  # finalized by _merge_state()
        else:
            # Latest renewal: update eligible_units if capacity changed
            if latest["quantity"] > 0:
                dep_df.at[i, "eligible_units"] = latest["quantity"]

    # Merge new rows back (add is_renewal / is_cancellation cols to existing too)
    if "is_renewal" not in ent_dep_df.columns:
        ent_dep_df = ent_dep_df.copy()
        ent_dep_df["is_renewal"]      = True
        ent_dep_df["is_cancellation"] = False
    combined = pd.concat([ent_dep_df, pd.DataFrame(new_rows)], ignore_index=True)
    return combined, dep_df


# ── Telemetry ─────────────────────────────────────────────────────────────────

# Product families → telemetry signal semantics
_PROD_FAMILY = {
    "PA-SMB": "Strata", "PA-ENT": "Strata", "VM-SERIES": "Strata", "CN-SERIES": "Strata",
    "PA-ACCESS-U": "SASE", "PA-ACCESS-N": "SASE", "GP": "SASE",
    "PC-CSPM": "Prisma", "PC-CWP": "Prisma", "PC-DSPM": "Prisma",
    "PC-APPSEC": "Prisma", "CC": "Prisma",
    "XDR": "Cortex", "XSOAR": "Cortex", "XSIAM": "Cortex", "XPANSE": "Cortex",
}

def _engagement_metrics(product_id, ratio, archetype, eligible, g):
    """
    Return product-specific engagement metrics for one telemetry week.

    Fields:
      remediation_events  — threats blocked (Strata), findings remediated (Prisma),
                            issues fixed (Xpanse), cases closed (Cortex XDR/SIAM/SOAR)
      automation_events   — automated responses / playbooks executed (high for SOAR/SIAM,
                            moderate for Prisma auto-remediation, low for Strata/SASE)
      scan_success_rate   — reliability of the scan/data-collection process
                            (1.0 for operational families; meaningful for Prisma Cloud)
    """
    fam = _PROD_FAMILY.get(product_id, "other")
    active = max(0, int(ratio * eligible))

    if fam == "Strata":
        # Threats blocked scales with traffic volume and coverage
        remediation = max(0, int(active * float(abs(g.normal(120, 40)))))
        automation  = max(0, int(active * float(abs(g.normal(0.3, 0.1)))))
        scan_ok     = None   # not applicable — Strata has no cloud-scan concept

    elif fam == "SASE":
        remediation = max(0, int(active * float(abs(g.normal(8, 3)))))   # DLP events
        automation  = max(0, int(active * float(abs(g.normal(1.5, 0.5)))))
        scan_ok     = None   # not applicable — SASE has no cloud-scan concept

    elif fam == "Prisma":
        # Scan success rate is meaningful — degrades with low coverage / bad config
        scan_ok = float(np.clip(
            g.normal(0.92 if ratio > 0.5 else 0.65, 0.07), 0.0, 1.0
        ))
        # Findings remediated depends on whether team is actively working
        remediation = max(0, int(active * float(abs(g.normal(2.5, 1.0)))))
        # Auto-remediation policy adoption varies by archetype
        auto_base = 0.8 if archetype in ("healthy_steady", "workload_growth") else 0.2
        automation = max(0, int(active * float(abs(g.normal(auto_base, 0.3)))))

    elif fam == "Cortex":
        if product_id == "XSOAR":
            # SOAR: playbooks executed — high automation is the VALUE signal
            automation  = max(0, int(abs(g.normal(45 if ratio > 0.4 else 5, 15))))
            remediation = max(0, int(abs(g.normal(12 if ratio > 0.4 else 2, 5))))
        elif product_id == "XSIAM":
            automation  = max(0, int(active * float(abs(g.normal(3.5, 1.0)))))
            remediation = max(0, int(active * float(abs(g.normal(1.2, 0.5)))))
        elif product_id == "XPANSE":
            remediation = max(0, int(abs(g.normal(8 * ratio, 4))))   # issues remediated
            automation  = 0
        else:  # XDR
            remediation = max(0, int(active * float(abs(g.normal(0.8, 0.3)))))
            automation  = max(0, int(active * float(abs(g.normal(0.4, 0.2)))))
        scan_ok = None   # not applicable — Cortex uses event-based ingestion, not periodic scans

    else:
        remediation = 0
        automation  = 0
        scan_ok     = None   # unknown family — not applicable

    return int(remediation), int(automation), (float(round(scan_ok, 4)) if scan_ok is not None else None)


def _dep_rng(deployment_id: str, master_seed: int) -> np.random.Generator:
    """
    Create a stable per-deployment RNG derived only from the deployment's ID
    and the simulation master seed.  Completely independent of row order,
    other deployments' history lengths, or eventual deployment state.

    This means: changing deployment A's dep_end does NOT shift deployment B's
    random sequence — the cross-deployment invariant is guaranteed by construction.
    """
    token = f"{master_seed}:{deployment_id}".encode()
    dep_seed = int(hashlib.sha256(token).hexdigest(), 16) % (2 ** 63)
    return np.random.default_rng(dep_seed)


def build_telemetry(deployments_df, cfg, _g_unused, archetype_labels, out_path, exp_state_map=None):
    """
    FIX #3: Generate telemetry for ALL deployment archetypes, including
    abandoned ones (showing the decline trajectory that precedes abandonment).
    FIX #4: Set available_at correctly so feature pipeline can filter on it.
    FIX: Per-deployment RNG — each deployment draws from its own independent
    stream derived from its ID and the simulation seed.  Changing any one
    deployment's dep_end cannot shift the random sequence for any other deployment.
    """
    obs_start = pd.Timestamp(cfg["observation_start"])
    obs_end = pd.Timestamp(cfg["observation_end"])
    missing_rate = cfg["dirty_data"]["missing_telemetry_rate"]
    late_rate = cfg["dirty_data"]["late_telemetry_rate"]
    batch_size = cfg.get("telemetry_batch_size", 5000)
    opening_start = pd.Timestamp(cfg.get("opening_base_start", "2018-10-01"))
    master_seed = cfg.get("seed", 42)

    # ALL deployments get telemetry — including grace_period and decommissioned.
    # Processing order no longer matters for reproducibility: each deployment has
    # its own isolated RNG stream, so row position is irrelevant.
    active_deps = deployments_df.copy().reset_index(drop=True)

    arch_map = archetype_labels.set_index("deployment_id")["archetype"].to_dict()
    weeks = pd.date_range(start=opening_start, end=obs_end, freq="W-MON")

    first_batch = True
    total = len(active_deps)
    updated_states = {}   # dep_id → final state after trajectory

    for batch_start in range(0, total, batch_size):
        batch = active_deps.iloc[batch_start:batch_start + batch_size]
        rows = []
        for _, dep in batch.iterrows():
            dep_start = pd.Timestamp(dep["deployment_start"])
            dep_end_raw = dep["deployment_end"]
            dep_end = pd.Timestamp(dep_end_raw) if dep_end_raw else obs_end
            archetype = arch_map.get(dep["deployment_id"], "healthy_steady")
            eligible = max(1, int(dep["eligible_units"]))
            dep_weeks = [w for w in weeks if dep_start <= w <= min(dep_end, obs_end)]

            # Per-deployment isolated RNG — independent of all other deployments
            g = _dep_rng(dep["deployment_id"], master_seed)

            # Deployment scope: for infrastructure products, the environment to protect
            # is larger than the licensed capacity (scope_expansion_by_billing_unit config).
            # For seat/user products, scope = licensed units exactly.
            _billing_unit = str(dep.get("billing_unit", "")).lower()
            _scope_exp_map = cfg.get("scope_expansion_by_billing_unit", {})
            _scope_max = float(_scope_exp_map.get(_billing_unit, 1.0))
            if _scope_max > 1.0:
                scope_eligible = max(eligible, int(g.uniform(1.0, _scope_max) * eligible))
            else:
                scope_eligible = eligible

            # B4: Look up company-level experience state to modulate coverage ratio
            co_id = dep.get("company_id")
            _exp_state = (exp_state_map or {}).get(co_id, {}) if co_id else {}
            _coverage_multiplier = 1.0
            if _exp_state.get("deployment_gap", 0.0) > 0.5:
                _coverage_multiplier = 1.0 - _exp_state["deployment_gap"] * 0.3

            peak_usage = 0.0
            final_usage = 0.0
            for idx, wk in enumerate(dep_weeks):
                # FIX #3: missing_rate applies to ALL archetypes
                if g.random() < missing_rate:
                    rows.append({
                        "deployment_id":      dep["deployment_id"],
                        "product_id":         dep["product_id"],
                        "week_start":         wk.date().isoformat(),
                        "active_units":       None,
                        "eligible_units":     eligible,
                        "coverage_pct":       None,
                        "policy_events":      None,
                        "admin_logins":       None,
                        "remediation_events": None,
                        "automation_events":  None,
                        "scan_success_rate":  None,
                        "telemetry_status":   "missing",
                        "available_at":       None,
                        "is_late":            0,
                        # Extended columns: null for missing telemetry weeks
                        "scope_targets":            scope_eligible,  # scope is always known
                        "deployed_targets":         None,
                        "productive_opportunities": None,
                        "productive_reach":         None,
                        "workflow_attempts":        None,
                        "workflow_successes":       None,
                    })
                    continue

                # FIX: use elapsed calendar time normalised to a fixed horizon so
                # that a later cancellation (which shrinks dep_end / n_weeks) does not
                # rewrite already-generated historical coverage values.
                _FIXED_HORIZON_DAYS = 5 * 365  # 5-year reference span
                t = min(1.0, (wk - dep_start).days / _FIXED_HORIZON_DAYS)
                ratio, policy_ev, admin_log = _usage_by_archetype(
                    archetype, eligible, t, g)
                # Shared forward experience is the common cause of behavior and risk.
                # Retain trajectory variation; do not condition history on cancellation.
                if cfg.get("experience_scenario", {}).get("enabled", False) and _exp_state:
                    mix = float(cfg["experience_scenario"]["shared_coverage_weight"])
                    ratio = (mix * (1.0 - _exp_state["deployment_gap"])
                             + (1.0 - mix) * ratio + g.normal(0, 0.025))
                else:
                    ratio *= _coverage_multiplier
                ratio = float(np.clip(ratio, 0.0, 1.0))
                remediation_ev, automation_ev, scan_ok = _engagement_metrics(
                    dep["product_id"], ratio, archetype, eligible, g)

                late_days = int(g.integers(3, 22)) if g.random() < late_rate else 0
                # FIX #4: available_at = week + 7 days + late_days
                available_at = (wk + timedelta(days=7 + late_days)).date().isoformat()

                peak_usage = max(peak_usage, ratio)
                final_usage = ratio

                # Extended telemetry columns: scope/deployed targets and workflow metrics
                # scope_targets: infrastructure to protect (may exceed licensed capacity
                # for non-seat products per scope_expansion_by_billing_unit config).
                # deployed_targets: units actively protected = ratio × licensed eligible.
                scope_targets = scope_eligible
                deployed_targets = max(0, int(ratio * eligible))
                # B2: Archetype-dependent success probability (varies meaningfully by archetype)
                wf_success_base = {
                    "healthy_steady": 0.85, "workload_growth": 0.82, "seasonal": 0.78,
                    "delayed_onboarding": 0.55, "workload_shrinkage": 0.70,
                    "gradual_abandonment": 0.40, "failed_deployment": 0.20,
                    "platform_migration": 0.50, "champion_turnover": 0.65,
                    "budget_pressure": 0.60,
                }.get(archetype, 0.70)
                if cfg.get("experience_scenario", {}).get("enabled", False) and _exp_state:
                    wf_success_base = 0.90 - 0.80 * _exp_state["value_decline"]
                wf_success_prob = float(np.clip(g.normal(wf_success_base, 0.08), 0.0, 1.0))
                # Draw attempts independently: deployed units × attempt rate per unit per week
                attempt_rate = float(abs(g.normal(3.5, 1.0)))  # avg attempts per deployed unit per week
                workflow_attempts = max(0, int(deployed_targets * attempt_rate))
                # Draw successes from attempts × success probability
                workflow_successes = int(g.binomial(workflow_attempts, wf_success_prob)) if workflow_attempts > 0 else 0
                # productive_reach: subset that reached the workflow (may differ from successes)
                productive_opportunities = workflow_attempts  # opportunities = attempts
                productive_reach = workflow_successes

                rows.append({
                    "deployment_id":          dep["deployment_id"],
                    "product_id":             dep["product_id"],
                    "week_start":             wk.date().isoformat(),
                    "active_units":           max(0, int(ratio * eligible)),
                    "eligible_units":         eligible,
                    "coverage_pct":           round(float(np.clip(ratio, 0.0, 1.0)), 4),
                    "policy_events":          max(0, int(ratio * eligible * float(abs(g.normal(50, 15))))),
                    "admin_logins":           max(0, int(abs(g.normal(3 if ratio > 0.4 else 0.5, 1)))),
                    "remediation_events":     remediation_ev,
                    "automation_events":      automation_ev,
                    "scan_success_rate":      scan_ok,
                    "telemetry_status":       "received",
                    "available_at":           available_at,
                    "is_late":                int(late_days > 0),
                    # Forward simulation: deployment reach and workflow productivity columns
                    "scope_targets":          scope_targets,
                    "deployed_targets":       deployed_targets,
                    "productive_opportunities": productive_opportunities,
                    "productive_reach":       productive_reach,
                    "workflow_attempts":      workflow_attempts,
                    "workflow_successes":     workflow_successes,
                })

            # Set final deployment_state from trajectory (FIX #3)
            final_state = _final_state_from_trajectory(archetype, final_usage, peak_usage)
            updated_states[dep["deployment_id"]] = final_state

        pd.DataFrame(rows).to_csv(out_path, mode="w" if first_batch else "a",
                                   header=first_batch, index=False)
        first_batch = False
        done = min(batch_start + batch_size, total)
        print(f"  [telemetry] {done}/{total} deps written...", end="\r")

    print()
    return updated_states


def _usage_by_archetype(archetype, eligible, t, g):
    import math
    if archetype == "healthy_steady":
        ratio = g.normal(0.75, 0.05)
    elif archetype == "delayed_onboarding":
        ratio = min(0.78, g.normal(0.08 + 0.70 * min(t * 4, 1.0), 0.04))
    elif archetype == "gradual_abandonment":
        ratio = max(0.0, g.normal(0.72 - 0.68 * t, 0.05))
    elif archetype == "workload_growth":
        ratio = min(1.0, g.normal(0.35 + 0.55 * t, 0.05))
    elif archetype == "workload_shrinkage":
        ratio = max(0.08, g.normal(0.80 - 0.55 * t, 0.05))
    elif archetype == "seasonal":
        ratio = g.normal(0.5 + 0.28 * math.sin(2 * math.pi * t * 2), 0.04)
    elif archetype == "failed_deployment":
        ratio = max(0.0, g.normal(0.08, 0.04))
    elif archetype == "sensor_outage":
        ratio = g.normal(0.68, 0.06) if g.random() > 0.20 else 0.0
    elif archetype == "platform_migration":
        ratio = max(0.0, g.normal(0.72 - 0.60 * t, 0.05))
    elif archetype == "champion_turnover":
        dip = 1.0 if (t < 0.20 or t > 0.45) else max(0.1, 0.3 + 0.7 * abs((t - 0.32) / 0.12))
        ratio = g.normal(0.70 * dip, 0.05)
    elif archetype == "budget_pressure":
        ratio = max(0.04, g.normal(0.62 - 0.35 * t, 0.07))
    else:
        ratio = g.normal(0.60, 0.10)

    ratio = float(np.clip(ratio, 0.0, 1.0))
    policy_events = ratio * eligible * float(abs(g.normal(50, 15)))
    admin_logins = max(0.0, float(abs(g.normal(3 if ratio > 0.4 else 0.5, 1))))
    return ratio, policy_events, admin_logins


def _final_state_from_trajectory(archetype, final_usage, peak_usage):
    """Determine deployment state from its actual usage trajectory."""
    if archetype == "platform_migration":
        return "migrating"
    if archetype in ("failed_deployment",) or (peak_usage < 0.05 and final_usage < 0.05):
        return "abandoned"
    if archetype == "gradual_abandonment" and final_usage < 0.10:
        return "abandoned"
    if archetype == "budget_pressure" and final_usage < 0.08:
        return "abandoned"
    return "active_paid"


# ── Support tickets ───────────────────────────────────────────────────────────

def build_support_tickets(accounts_df, users_df, cfg, g,
                           items_df=None, deps_df=None, archetype_map=None, exp_state_map=None):
    """
    Generate support_tickets.csv.

    Extended schema includes deployment/contract linkage, customer impact signals,
    and SLA tracking.  items_df and deps_df are optional; when supplied they enable
    deployment_id and contract_id FK columns on each ticket.
    """
    obs_start = pd.Timestamp(cfg["observation_start"])
    obs_end = pd.Timestamp(cfg["observation_end"])
    opening_start = pd.Timestamp(cfg.get("opening_base_start", "2018-10-01"))
    full_span = (obs_end - opening_start).days
    user_by_acc = users_df.groupby("customer_account_id")["user_id"].apply(list).to_dict()

    # Build lookups for coherent ticket FK assignment.
    # Approach: pick the affected deployment first (product is determined), then
    # resolve its contract from the same product's active subscriptions at the
    # ticket date.  This ensures dep_id and contract_id reference the same product.
    #
    # _dep_product: deployment_id → product_id
    # _co_prod_contracts: (company_id, product_id) →
    #     list of (contract_id, term_start, term_end) for positive-value subscriptions
    # _acc_deps: company_id → list of deployment_ids
    _dep_product: dict = {}       # deployment_id → product_id
    _co_prod_contracts: dict = {} # (co_id, prod_id) → [(ctr_id, ts, te)]
    _acc_deps: dict = {}          # company_id → list of deployment_ids
    if deps_df is not None and len(deps_df) > 0:
        for _, dr in deps_df.iterrows():
            _dep_product[dr["deployment_id"]] = dr.get("product_id")
        for co, grp in deps_df.groupby("company_id"):
            _acc_deps[co] = grp["deployment_id"].tolist()
    if items_df is not None and len(items_df) > 0:
        # Only positive-value subscriptions — exclude zero-value cancellation transactions
        _items_sub = items_df[
            (items_df["recurring_flag"] == True) &
            (items_df["net_acv"].fillna(0) > 0)
        ].copy()
        if "company_id" in _items_sub.columns:
            for _, row in _items_sub.iterrows():
                co = row.get("company_id")
                pid = row.get("product_id")
                cid = row.get("contract_id")
                if not co or not pid or not cid:
                    continue
                ts = pd.Timestamp(row["term_start"]).date() if pd.notna(row.get("term_start")) else None
                te = pd.Timestamp(row["term_end"]).date() if pd.notna(row.get("term_end")) else None
                _co_prod_contracts.setdefault((co, pid), []).append((cid, ts, te))

    # Per-user activity window for chronology enforcement
    # Stored as date objects (same as build_orders_and_items) so that sub-day time
    # components in ticket timestamps don't cause false "inactive" comparisons.
    _user_window = {}
    for _, ur in users_df.iterrows():
        ato_raw = ur.get("active_to")
        _user_window[ur["user_id"]] = (
            pd.Timestamp(ur["created_at"]).date(),
            pd.Timestamp(ato_raw).date() if pd.notna(ato_raw) else None,
        )

    TICKET_TYPES = ["technical", "billing", "feature_request", "onboarding", "incident"]
    PRIORITIES = ["P1", "P2", "P3", "P4"]
    PRIORITY_WEIGHTS = [0.05, 0.20, 0.45, 0.30]

    # Onboarding spike: tickets arrive faster in the first 90 days (higher-touch
    # as the customer learns the product), then settle to the mature annual rate.
    ONBOARD_DAYS = 90
    ONBOARD_MULT = 2.5   # tickets/yr in the onboarding window vs steady-state

    rows = []
    tid = 1
    for _, acc in accounts_df.iterrows():
        seg = acc["size_segment"]
        annual_rate = {"smb": 3, "midmarket": 5, "enterprise": 12, "large_enterprise": 30}.get(seg, 5)
        # B4: Scale annual_rate by service_problem from experience state
        co_id_acc = acc.get("company_id") or acc.get("customer_account_id")
        _acc_exp = (exp_state_map or {}).get(co_id_acc, {}) if co_id_acc else {}
        _svc_prob = _acc_exp.get("service_problem", 0.0)
        if _svc_prob > 0.0:
            annual_rate = annual_rate * (1.0 + _svc_prob)
        acc_users = sorted(user_by_acc.get(acc["customer_account_id"], []))
        if not acc_users:
            continue
        first_pur = pd.Timestamp(acc["first_purchase_date"])
        span = max(1, (obs_end - first_pur).days)

        # Scale expected ticket count by actual customer exposure, not a fixed 5-year window.
        # Onboarding period (first 90 days): 2.5× the mature rate — higher-touch as the
        # customer learns the product.  Tickets are drawn from each period separately so
        # the elevated arrival rate is time-localised within the onboarding window.
        onboard_days_actual = min(ONBOARD_DAYS, span)
        mature_days_actual  = max(0, span - onboard_days_actual)
        onboard_yrs = onboard_days_actual / 365.25
        mature_yrs  = mature_days_actual  / 365.25
        n_onboard = int(g.poisson(max(annual_rate * onboard_yrs * ONBOARD_MULT, 0.0)))
        n_mature  = int(g.poisson(max(annual_rate * mature_yrs,                 0.0)))

        # Build a flat list of (created_date) by drawing offsets from each period.
        ticket_dates = []
        for _ in range(n_onboard):
            offset = int(g.integers(0, max(1, onboard_days_actual)))
            ticket_dates.append(first_pur + timedelta(days=offset))
        for _ in range(n_mature):
            if mature_days_actual <= 0:
                break
            offset = int(g.integers(0, max(1, mature_days_actual)))
            ticket_dates.append(first_pur + timedelta(days=onboard_days_actual + offset))

        # Company deployment/contract pools for FK linking
        co_id = acc["company_id"]
        co_deps = _acc_deps.get(co_id, [])

        # B5: Build dep_start/dep_end maps for chronology filtering (per company)
        _dep_start_map = {}
        _dep_end_map = {}
        if deps_df is not None and len(deps_df) > 0:
            co_deps_rows = deps_df[deps_df["company_id"] == co_id] if "company_id" in deps_df.columns else pd.DataFrame()
            for _, dr in co_deps_rows.iterrows():
                did = dr["deployment_id"]
                _dep_start_map[did] = pd.Timestamp(dr["deployment_start"])
                raw_end = dr.get("deployment_end")
                _dep_end_map[did] = pd.Timestamp(raw_end) if raw_end and pd.notna(raw_end) else obs_end

        # Per-account archetype shapes priority distribution and resolution speed
        co_archetype = (archetype_map or {}).get(co_id, "healthy_steady")
        _high_risk = co_archetype in ("failed_deployment", "gradual_abandonment", "budget_pressure")
        _med_risk  = co_archetype in ("delayed_onboarding", "champion_turnover", "workload_shrinkage")
        # High-risk accounts: more P1/P2, slower resolution, more likely to leave issues open
        if _high_risk:
            _pw = [0.12, 0.30, 0.40, 0.18]   # elevated P1/P2
        elif _med_risk:
            _pw = [0.07, 0.25, 0.43, 0.25]
        else:
            _pw = PRIORITY_WEIGHTS             # default [0.05, 0.20, 0.45, 0.30]

        for created in ticket_dates:
            if created > obs_end:
                continue
            priority = g.choice(PRIORITIES, p=_pw)
            # Resolution time: high-risk accounts resolve P1/P2 much more slowly
            _slow = _high_risk and priority in ("P1", "P2")
            res_hrs = {"P1": g.normal(12 if _slow else 4, 6 if _slow else 2),
                       "P2": g.normal(48 if _slow else 24, 20 if _slow else 8),
                       "P3": g.normal(72, 24), "P4": g.normal(168, 48)}.get(priority, 48)
            res_hrs = max(1.0, float(res_hrs))
            resolved = created + timedelta(hours=res_hrs)
            if resolved > obs_end:
                resolved = None
            # High-risk accounts: 35% chance of leaving P1/P2 ticket unresolved (vs 5%)
            if priority in ("P1", "P2") and resolved is not None:
                _unresolved_prob = 0.35 if _high_risk else (0.15 if _med_risk else 0.05)
                if g.random() < _unresolved_prob:
                    resolved = None
                    res_hrs = (obs_end - created).total_seconds() / 3600  # hours since creation
            ticket_user = _pick_active_user(acc_users, created, _user_window, g)

            # SLA targets by priority
            sla_target = {"P1": 4.0, "P2": 8.0, "P3": 24.0, "P4": 72.0}.get(priority, 24.0)
            sla_breached = (resolved is not None and res_hrs > sla_target)

            # Customer impact: P1/P2 tickets are customer-impacting
            customer_impact = priority in ("P1", "P2")
            # impact_hours_open: hours the customer was impacted (open until resolved or obs_end)
            if customer_impact:
                if resolved is None:
                    impact_hours = round((obs_end - created).total_seconds() / 3600, 1)
                else:
                    impact_hours = round(res_hrs, 1)
            else:
                impact_hours = 0.0
            impact_start = created.isoformat() if customer_impact else None
            impact_end = resolved.isoformat() if (customer_impact and resolved) else None

            # FK links: pick an active deployment first, then resolve its contract.
            # This ensures deployment_id and contract_id reference the same product.
            dep_id = None
            contract_id = None
            _ticket_date = created.date() if hasattr(created, "date") else created
            if co_deps and _dep_start_map:
                _dep_active = [d for d in co_deps
                               if _dep_start_map.get(d, obs_start) <= created <= _dep_end_map.get(d, obs_end)]
                dep_id = _dep_active[int(g.integers(len(_dep_active)))] if _dep_active else None
            elif co_deps:
                dep_id = co_deps[int(g.integers(len(co_deps)))]
            # Resolve contract from the deployment's product at the ticket date.
            # Only positive-value subscriptions (excludes cancellation transactions).
            if dep_id is not None:
                _dep_pid = _dep_product.get(dep_id)
                if _dep_pid:
                    _prod_contracts = _co_prod_contracts.get((co_id, _dep_pid), [])
                    # Contract must have started by ticket date; preferably also covering it
                    _covering = [
                        cid for cid, ts, te in _prod_contracts
                        if (ts is None or ts <= _ticket_date)
                        and (te is None or te >= _ticket_date)
                    ]
                    _grace_days = cfg.get("renewal_grace_period_days", 30)
                    _started = [
                        cid for cid, ts, te in _prod_contracts
                        if (ts is None or ts <= _ticket_date)
                        and (te is None or (_ticket_date - te).days <= _grace_days)
                    ] if not _covering else []
                    _eligible = _covering if _covering else _started
                    if _eligible:
                        contract_id = _eligible[int(g.integers(len(_eligible)))]

            rows.append({
                "ticket_id":            f"TKT{tid:010d}",
                "customer_account_id":  acc["customer_account_id"],
                "company_id":           co_id,
                "user_id":              ticket_user,
                "ticket_type":          g.choice(TICKET_TYPES),
                "priority":             priority,
                "created_at":           created.isoformat(),
                "resolved_at":          resolved.isoformat() if resolved else None,
                "resolution_hours":     round(res_hrs, 1) if resolved else None,
                "status":               "resolved" if resolved else "open",
                # Extended fields
                "deployment_id":        dep_id,
                "contract_id":          contract_id,
                "customer_impact":      customer_impact,
                "impact_hours_open":    impact_hours,
                "impact_start_at":      impact_start,
                "impact_end_at":        impact_end,
                "sla_target_hours":     sla_target,
                "sla_breached":         sla_breached,
            })
            tid += 1
    return pd.DataFrame(rows)


# ── Renewal quotes ────────────────────────────────────────────────────────────

def build_renewal_quotes(items_df, orders_df, cfg, g, exp_state_map=None):
    """
    Generate renewal quotes issued ~120 days before each contract term_end.

    Schema: quote_id, contract_id, company_id, issued_at, available_at,
            quoted_quantity, quoted_annual_unit_price, comparable_prior_unit_price,
            unit_price_change_pct, discount, currency

    Rules:
    - Only for recurring subscription contracts (term_end is not None)
    - Quote issued cfg["quote_lead_days"] (120) days before term_end
    - Must be within the observation window
    - future accepted_price and renewal_rejected are NOT included (leakage prevention)
    - available_at = issued_at + 0-2 days (CRM logging delay)
    """
    quote_lead = cfg.get("quote_lead_days", 120)
    obs_start = pd.Timestamp(cfg["observation_start"])
    obs_end   = pd.Timestamp(cfg["observation_end"])

    # Build company_id lookup from orders
    order_co_map = {}
    if "company_id" in orders_df.columns and "order_id" in orders_df.columns:
        order_co_map = orders_df.set_index("order_id")["company_id"].to_dict()

    # Get company_id for each item via order
    items_with_co = items_df.copy()
    if "company_id" not in items_with_co.columns:
        items_with_co["company_id"] = items_with_co["order_id"].map(order_co_map)

    # Get subscription items (recurring, non-support, with term_end); exclude cancellation rows (B3)
    recurring_items = items_with_co[
        (items_with_co["recurring_flag"] == True) &
        (items_with_co["term_end"].notna()) &
        (~items_with_co["product_id"].isin(["SUP-PREM", "SUP-ENH"])) &
        (items_with_co["term_years"].fillna(0) > 0) &   # exclude cancellation rows
        (items_with_co["line_net_amount"].fillna(0) > 0)  # exclude zero-value rows
    ].copy()

    rows = []
    qid = 1
    for _, item in recurring_items.iterrows():
        term_end_raw = item.get("term_end")
        if not term_end_raw:
            continue
        try:
            term_end = pd.Timestamp(term_end_raw)
        except Exception:
            continue
        quote_date = term_end - timedelta(days=quote_lead)

        # Only quote if within observation window and at least 14 days from obs_start
        if quote_date < obs_start + timedelta(days=14) or quote_date > obs_end:
            continue

        # available_at = issued_at + 0-2 days
        avail_delay = int(g.integers(0, 3))
        available_at = quote_date + timedelta(days=avail_delay)

        current_qty = int(item.get("quantity", 1) or 1)
        current_price = float(item.get("annual_catalog_price", 0) or 0)

        # B4: Use experience state to modulate price change
        # High commercial pressure → higher price change range
        co_id_item = item.get("company_id")
        _item_exp = (exp_state_map or {}).get(co_id_item, {}) if co_id_item else {}
        _com_pres = _item_exp.get("commercial_pressure", 0.0)
        if _com_pres > 0.5:
            # High commercial pressure: drawn from higher range
            price_change = float(g.uniform(0.02, 0.12))  # +2% to +12%
        else:
            # Quote may have slight price change (0-5% increase typical)
            price_change = float(g.uniform(-0.02, 0.06))  # -2% to +6%
        quoted_price = round(current_price * (1 + price_change), 4)

        line_net = float(item.get("line_net_amount", 1) or 1)
        line_disc = float(item.get("line_discount_amount", 0) or 0)
        discount_ratio = round(line_disc / max(line_net + line_disc, 1), 6)

        rows.append({
            "quote_id":                    f"QT{qid:010d}",
            "contract_id":                 item["contract_id"],
            "company_id":                  item.get("company_id", ""),
            "issued_at":                   quote_date.date().isoformat(),
            "available_at":                available_at.date().isoformat(),
            "quoted_quantity":             current_qty,
            "quoted_annual_unit_price":    quoted_price,
            "comparable_prior_unit_price": current_price,
            "unit_price_change_pct":       round(price_change * 100, 4),
            "discount":                    discount_ratio,
            "currency":                    "USD",
        })
        qid += 1

    return pd.DataFrame(rows)


# ── Contract events ───────────────────────────────────────────────────────────

def build_contract_events(orders_df, items_df, cfg):
    """
    FIX #1: Contract events now carry expiring_contract_id and prior_net_acv
    so that integrate.py can find cancellations via the expiring entitlement.
    """
    obs_end = pd.Timestamp(cfg["observation_end"])
    label_window = cfg.get("label_window_days", 90)

    order_map = orders_df.set_index("order_id").to_dict("index")
    events = []
    eid = 1

    for _, item in items_df[items_df["recurring_flag"] == True].iterrows():
        order = order_map.get(item["order_id"])
        if not order:
            continue
        order_type = order["order_type"]
        order_date = order["order_date"]
        event_type = {
            "new_logo":            "initial_purchase",
            "renewal":             "renewal",
            "renewal_expansion":   "renewal_expansion",
            "renewal_contraction": "renewal_contraction",
            "cancellation":        "cancellation",
        }.get(order_type, "other")

        # Label window: outcome_end is term_end + label_window
        term_end_dt = pd.Timestamp(item["term_end"]) if item.get("term_end") else None
        label_window_end = (term_end_dt + timedelta(days=label_window)).date().isoformat() \
            if term_end_dt else None
        label_matured = (term_end_dt + timedelta(days=label_window)) <= obs_end \
            if term_end_dt else False

        events.append({
            "event_id":               f"EVT{eid:012d}",
            "contract_id":            item["contract_id"],
            "expiring_contract_id":   item.get("predecessor_contract_id"),   # FIX #1
            "renewal_lineage_id":     item["renewal_lineage_id"],
            "customer_account_id":    order["customer_account_id"],
            "company_id":             order.get("company_id", order["customer_account_id"]),
            "product_id":             item["product_id"],
            "event_type":             event_type,
            "event_date":             order_date,
            "term_start":             item.get("term_start"),
            "term_end":               item.get("term_end"),
            "net_acv":                item.get("net_acv", 0),
            "prior_net_acv":          item.get("prior_net_acv", 0),   # FIX #1
            "label_window_end":       label_window_end,
            "label_matured":          label_matured,
            "churned":                1 if event_type == "cancellation" else 0,
        })
        eid += 1

    return pd.DataFrame(events)


# ── Dirty data injection ──────────────────────────────────────────────────────

def inject_dirty_data(orders_df, items_df, cfg, g):
    """
    FIX #5 (Step 7): Inject duplicates as exact field copies identified by a
    canonical_hash field, NOT by a _DUP suffix in the primary key.
    The ingest_clean.py pipeline must detect and remove them using the hash.
    """
    dd = cfg["dirty_data"]
    anomaly_records = []

    # 1. Duplicate orders — inject as exact-copy rows with same order_id
    n_dup = int(len(orders_df) * dd["duplicate_order_rate"])
    dup_idx = sorted(g.choice(len(orders_df), size=n_dup, replace=False).tolist())
    dup_orders = orders_df.iloc[dup_idx].copy()
    # Mark ground truth
    for oid in dup_orders["order_id"]:
        anomaly_records.append({"record_id": oid, "anomaly_type": "duplicate_order",
                                  "canonical_id": oid, "table": "orders"})
    orders_dirty = pd.concat([orders_df, dup_orders], ignore_index=True)

    # 2. Inconsistent product_id casing
    n_case = int(len(items_df) * dd["inconsistent_product_id_rate"])
    case_idx = sorted(g.choice(len(items_df), size=n_case, replace=False).tolist())
    items_dirty = items_df.copy()
    orig_vals = items_dirty.iloc[case_idx]["product_id"].values.tolist()
    for i, idx in enumerate(case_idx):
        iid = items_dirty.iloc[idx]["order_item_id"]
        dirty_val = orig_vals[i].lower() if g.random() > 0.5 else orig_vals[i].title()
        items_dirty.iat[idx, items_dirty.columns.get_loc("product_id")] = dirty_val
        anomaly_records.append({"record_id": iid, "anomaly_type": "product_id_case",
                                  "canonical_id": iid, "table": "order_items",
                                  "original_value": orig_vals[i], "dirty_value": dirty_val})

    # 3. Missing team mappings
    n_team = int(len(items_dirty) * dd["missing_team_mapping_rate"])
    team_idx = sorted(g.choice(len(items_dirty), size=n_team, replace=False).tolist())
    orig_teams = items_dirty.iloc[team_idx]["owner_team_id"].values.tolist()
    for i, idx in enumerate(team_idx):
        iid = items_dirty.iloc[idx]["order_item_id"]
        items_dirty.iat[idx, items_dirty.columns.get_loc("owner_team_id")] = None
        anomaly_records.append({"record_id": iid, "anomaly_type": "missing_team_mapping",
                                  "canonical_id": iid, "table": "order_items",
                                  "original_value": orig_teams[i]})

    # 4. Invalid dates
    n_date = int(len(orders_dirty) * dd["invalid_date_rate"])
    date_idx = sorted(g.choice(len(orders_dirty), size=n_date, replace=False).tolist())
    orig_dates = orders_dirty.iloc[date_idx]["order_date"].values.tolist()
    for i, idx in enumerate(date_idx):
        oid = orders_dirty.iloc[idx]["order_id"]
        orders_dirty.iat[idx, orders_dirty.columns.get_loc("order_date")] = "INVALID_DATE"
        anomaly_records.append({"record_id": oid, "anomaly_type": "invalid_date",
                                  "canonical_id": oid, "table": "orders",
                                  "original_value": orig_dates[i]})

    return orders_dirty, items_dirty, pd.DataFrame(anomaly_records)


# ── Account interactions ──────────────────────────────────────────────────────

INTERACTION_TYPES = ["executive_briefing", "marketing_event", "cs_meeting", "sales_meeting",
                      "budget_concern"]

def build_account_interactions(accounts_df: pd.DataFrame, cfg: dict, g,
                               archetype_map: dict = None) -> pd.DataFrame:
    """
    Generate account_interactions.csv: relationship touchpoints per account.

    Schema
    ------
    interaction_id  : IXN{N:08d}
    company_id      : FK → companies.company_id
    interaction_type: executive_briefing | marketing_event | cs_meeting | sales_meeting
    event_at        : timestamp when the interaction occurred
    available_at    : timestamp when it was logged (event_at + 0–3 days)

    Lifecycle rules
    ---------------
    All interaction types are customer-relationship activities and are generated
    from first_purchase_date onward (bounded below by observation_start for
    accounts whose first purchase predates the window).

    Volume is computed over each account's eligible duration (from the later of
    first_purchase_date and observation_start), not the full five-year window, to
    avoid inflating counts for accounts that became customers mid-period.

    Leakage control: available_at is always >= event_at and reflects realistic
    logging delays (0–3 days). Downstream features filter available_at <= as_of_date.

    Volume assumptions (per customer-year, by size segment):
      smb:              avg 1.5 total interactions/yr
      midmarket:        avg 3.0 total interactions/yr
      enterprise:       avg 6.0 total interactions/yr
      large_enterprise: avg 10.0 total interactions/yr
    """
    obs_start = pd.Timestamp(cfg["observation_start"])
    obs_end   = pd.Timestamp(cfg["observation_end"])

    rate_by_seg = {
        "smb":              1.5 / 365,
        "midmarket":        3.0 / 365,
        "enterprise":       6.0 / 365,
        "large_enterprise": 10.0 / 365,
    }
    # Weights: [executive_briefing, marketing_event, cs_meeting, sales_meeting, budget_concern]
    type_weights = {
        "smb":              [0.05, 0.33, 0.43, 0.14, 0.05],
        "midmarket":        [0.10, 0.28, 0.38, 0.18, 0.06],
        "enterprise":       [0.19, 0.24, 0.33, 0.19, 0.05],
        "large_enterprise": [0.29, 0.19, 0.28, 0.19, 0.05],
    }
    # Post-sale types require an existing customer relationship
    POST_SALE_TYPES = {"executive_briefing", "cs_meeting", "sales_meeting", "budget_concern"}

    # Vendor relationship owner roles (B7: these are vendor-side owners — CSM/AE/TAM —
    # not customer champion.  The owner_role field tracks which vendor employee owns
    # the relationship; days_without_confirmed_owner measures vendor coverage gaps.)
    OWNER_ROLES = ["csm", "account_executive", "technical_account_manager"]
    missing_interactions_rate = cfg.get("dirty_data", {}).get("missing_interactions_rate", 0.05)

    rows = []
    ixn_id = 1
    for _, acc in accounts_df.iterrows():
        co  = acc["company_id"]
        seg = acc.get("size_segment", "midmarket")
        daily_rate = rate_by_seg.get(seg, 2.0 / 365)
        weights    = type_weights.get(seg, [0.15, 0.30, 0.35, 0.20])

        # Customer relationship start: post-sale events cannot precede this date
        fp_raw = acc.get("first_purchase_date")
        customer_start = pd.Timestamp(fp_raw) if fp_raw else obs_start

        # Eligible window for volume calculation: customer inception to obs_end
        # (bounded by observation_start to avoid pre-window events)
        eligible_start = max(customer_start, obs_start)
        eligible_days  = (obs_end - eligible_start).days
        if eligible_days <= 0:
            continue

        # Poisson count over the eligible duration (not the full 5-year window)
        n_events = int(g.poisson(daily_rate * eligible_days))
        if n_events == 0:
            continue

        # Per-account archetype shapes handoff and budget-concern behaviour
        co_archetype = (archetype_map or {}).get(co, "healthy_steady")
        _is_turnover = co_archetype == "champion_turnover"
        _is_budget   = co_archetype == "budget_pressure"

        # Draw offsets within the eligible window
        offsets = sorted(int(x) for x in g.integers(0, eligible_days, size=n_events))
        # Track current owner for this account across sorted interactions
        current_owner_role = OWNER_ROLES[int(g.integers(len(OWNER_ROLES)))]
        current_owner_id = f"OWN{abs(int(g.integers(1, 999999))):06d}"
        owner_departed = False  # tracks whether the most recent owner has departed

        for offset in offsets:
            event_at = eligible_start + timedelta(days=offset)
            if event_at >= obs_end:
                continue
            itype = INTERACTION_TYPES[g.choice(len(INTERACTION_TYPES), p=weights)]

            # budget_pressure accounts: 3x more budget_concern interactions
            if _is_budget and itype != "budget_concern" and g.random() < 0.15:
                itype = "budget_concern"

            log_delay    = int(g.integers(0, 4))   # 0–3 day logging delay
            available_at = event_at + timedelta(days=log_delay)

            # Owner role: may be missing (dirty data) per missing_interactions_rate
            if g.random() < missing_interactions_rate:
                owner_role = None
            else:
                owner_role = current_owner_role

            # Owner handoff logic
            # - champion_turnover: elevated departure rate (15% of CS meetings), and 40% of
            #   departures have no immediate replacement (departure without succession)
            # - Other archetypes: 5% of CS meetings are handoffs, all have replacements
            _handoff_prob = 0.15 if _is_turnover else 0.05
            is_handoff = (itype == "cs_meeting" and g.random() < _handoff_prob)
            new_owner_id = None
            if is_handoff:
                if _is_turnover and g.random() < 0.40:
                    # Departure without immediate replacement — creates owner gap
                    new_owner_id = None
                    owner_departed = True
                    current_owner_id = None
                else:
                    # Replacement handoff — owner gap ends
                    new_role = OWNER_ROLES[int(g.integers(len(OWNER_ROLES)))]
                    new_owner_id = f"OWN{abs(int(g.integers(1, 999999))):06d}"
                    current_owner_role = new_role
                    current_owner_id = new_owner_id
                    owner_departed = False

            # Agreed action: ~10% of CS meetings create a success plan action
            agreed_action_id = None
            agreed_action_due = None
            agreed_action_completed = None
            if itype == "cs_meeting" and g.random() < 0.10:
                agreed_action_id = f"ACT{ixn_id:08d}"  # B8: deterministic counter-based ID
                due_offset = int(g.integers(30, 91))  # 30-90 days out
                agreed_action_due = (event_at + timedelta(days=due_offset)).date().isoformat()
                # ~60% of agreed actions are completed within the window
                if g.random() < 0.60:
                    _max_co = max(due_offset + 1, min(due_offset + 60, (obs_end - event_at).days + 1))
                    complete_offset = int(g.integers(due_offset, _max_co))
                    completed_dt = event_at + timedelta(days=complete_offset)
                    if completed_dt <= obs_end:
                        agreed_action_completed = completed_dt.date().isoformat()

            rows.append({
                "interaction_id":              f"IXN{ixn_id:08d}",
                "company_id":                  co,
                "interaction_type":            itype,
                "event_at":                    event_at.isoformat(),
                "available_at":                available_at.isoformat(),
                # Extended fields
                "owner_role":                  owner_role,
                "is_owner_handoff":            is_handoff,
                "new_owner_id":                new_owner_id,
                "agreed_action_id":            agreed_action_id,
                "agreed_action_due_date":      agreed_action_due,
                "agreed_action_completed_at":  agreed_action_completed,
            })
            ixn_id += 1

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, dev=args.dev)
    seed = cfg["seed"]
    g = rng(seed)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    # OS-backed exclusive lock — held open until process exits (no atexit needed)
    _lock_fh = _acquire_write_lock(DATA_RAW)
    cfg_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    manifest = {
        "run_id":           run_id,
        "config_hash":      cfg_hash,
        "seed": seed, "dev_mode": cfg.get("dev_mode", False),
        "company_count":    cfg["company_count"],
        "observation_start": cfg["observation_start"],
        "observation_end":   cfg["observation_end"],
        "run_timestamp":     datetime.utcnow().isoformat(),
        "tables": {},
    }
    t0 = time.time()

    print("[1/10] Building products catalog...")
    products_df = build_products(cfg)
    products_df.to_csv(DATA_RAW / "products.csv", index=False)
    manifest["tables"]["products"] = len(products_df)

    n_accounts = cfg["company_count"]
    print(f"[2/10] Building parent orgs + {n_accounts:,} customer accounts...")
    parents_df = build_parent_orgs(n_accounts, cfg, rng(seed + 1))
    parents_df.to_csv(DATA_RAW / "parent_orgs.csv", index=False)
    manifest["tables"]["parent_orgs"] = len(parents_df)

    accounts_df = build_customer_accounts(parents_df, cfg, rng(seed + 2))
    # Trim/pad to exactly n_accounts
    accounts_df = accounts_df.head(n_accounts).reset_index(drop=True)
    accounts_df.to_csv(DATA_RAW / "companies.csv", index=False)
    manifest["tables"]["customer_accounts"] = len(accounts_df)

    print("[3/10] Building teams...")
    teams_df = build_teams(accounts_df, cfg, rng(seed + 3))
    teams_df.to_csv(DATA_RAW / "teams.csv", index=False)
    manifest["tables"]["teams"] = len(teams_df)

    print("[4/10] Building users...")
    users_df = build_users(accounts_df, teams_df, cfg, rng(seed + 4))
    users_df.to_csv(DATA_RAW / "users.csv", index=False)
    manifest["tables"]["users"] = len(users_df)

    print("[5/10] Building orders and order items (corrected TCV/ACV)...")
    orders_df, items_df, company_archetype_map, exp_state_df, company_exp_states = build_orders_and_items(
        accounts_df, users_df, teams_df, products_df, cfg, rng(seed + 5)
    )
    manifest["tables"]["orders_canonical"] = len(orders_df)
    manifest["tables"]["order_items_canonical"] = len(items_df)
    # Save hidden experience states to eval/ (not in data/raw — no leakage)
    if len(exp_state_df) > 0:
        exp_state_df.to_csv(EVAL_DIR / "customer_experience_state.csv", index=False)
        manifest["tables"]["customer_experience_state"] = len(exp_state_df)

    print("[6/10] Injecting dirty data...")
    orders_dirty, items_dirty, anomaly_df = inject_dirty_data(
        orders_df, items_df, cfg, rng(seed + 6)
    )
    orders_dirty.to_csv(DATA_RAW / "orders.csv", index=False)
    items_dirty.to_csv(DATA_RAW / "order_items.csv", index=False)
    anomaly_df.to_csv(EVAL_DIR / "anomaly_ground_truth.csv", index=False)
    manifest["tables"]["orders"] = len(orders_dirty)
    manifest["tables"]["order_items"] = len(items_dirty)

    # Ensure company_id is on items_df (needed by support tickets and quotes)
    if "company_id" not in items_df.columns:
        _order_co = orders_df.set_index("order_id")["company_id"].to_dict()
        items_df = items_df.copy()
        items_df["company_id"] = items_df["order_id"].map(_order_co)
    # company_exp_states was computed and returned by build_orders_and_items()
    # using _det_seed() for process-stable reproducibility (issue 2 fix).

    print("[7/10] Building deployments...")
    deps_df, dep_team_df, ent_dep_df, arch_df = build_deployments(
        accounts_df, teams_df, orders_df, items_df, products_df, cfg, rng(seed + 7),
        company_archetype_map=company_archetype_map,
    )
    # Compute cancellation lifetimes BEFORE telemetry so telemetry is clipped correctly
    deps_df = compute_cancellation_lifetimes(items_df, deps_df, cfg)

    # State placeholder — updated after telemetry
    deps_df.to_csv(DATA_RAW / "deployments_prelim.csv", index=False)
    dep_team_df.to_csv(DATA_RAW / "deployment_team_map.csv", index=False)
    ent_dep_df.to_csv(DATA_RAW / "entitlement_deployment_map.csv", index=False)
    arch_df.to_csv(EVAL_DIR / "deployment_archetypes.csv", index=False)
    manifest["tables"]["deployments_prelim"] = len(deps_df)

    print("[8/10] Building telemetry (batched — includes abandoned arcs)...")
    telem_path = DATA_RAW / "telemetry_weekly.csv"
    updated_states = build_telemetry(deps_df, cfg, rng(seed + 8), arch_df, telem_path,
                                      exp_state_map=company_exp_states)

    # Extend entitlement map with renewals + update deployment capacity/lifecycle.
    # This must run BEFORE _merge_state() because it may update grace_end_date
    # for cancellations discovered in the renewal chain (not just initial items).
    ent_dep_df, deps_df = build_entitlement_renewal_map(items_df, ent_dep_df, deps_df, cfg)
    ent_dep_df.to_csv(DATA_RAW / "entitlement_deployment_map.csv", index=False)

    # Finalize deployment states from trajectory + cancellation lifecycle.
    # One consistent rule: grace_period → decommissioned only when the ACTUAL
    # grace window (grace_end_date, uncapped) has ended at scoring_date.
    scoring_date = pd.Timestamp(cfg["scoring_date"])

    def _merge_state(row, trajectory_state):
        current_state = row["deployment_state"]
        if current_state == "grace_period":
            # Use the uncapped grace_end_date, not the obs_end-capped deployment_end.
            # Capping deployment_end at obs_end (for telemetry) must not affect the
            # lifecycle decision — a grace period that ends after obs_end is still open.
            ged_raw = row.get("grace_end_date")
            if ged_raw and pd.notna(ged_raw):
                grace_end = pd.Timestamp(ged_raw)
                return "decommissioned" if grace_end <= scoring_date else "grace_period"
            # No grace_end_date stored → fall back conservatively to decommissioned
            return "decommissioned"
        return trajectory_state or current_state or "active_paid"

    deps_df["deployment_state"] = [
        _merge_state(r, updated_states.get(r["deployment_id"]))
        for _, r in deps_df.iterrows()
    ]

    deps_df.to_csv(DATA_RAW / "deployments.csv", index=False)
    telem_rows = sum(1 for _ in open(telem_path)) - 1
    manifest["tables"]["deployments"] = len(deps_df)
    manifest["tables"]["telemetry_weekly"] = telem_rows

    print("[9/10] Building support tickets...")
    tickets_df = build_support_tickets(accounts_df, users_df, cfg, rng(seed + 9),
                                        items_df=items_df, deps_df=deps_df,
                                        archetype_map=company_archetype_map,
                                        exp_state_map=company_exp_states)
    tickets_df.to_csv(DATA_RAW / "support_tickets.csv", index=False)
    manifest["tables"]["support_tickets"] = len(tickets_df)

    print("[10/12] Building contract events (with expiring_contract_id)...")
    events_df = build_contract_events(orders_df, items_df, cfg)
    events_df.to_csv(DATA_RAW / "contract_events.csv", index=False)
    manifest["tables"]["contract_events"] = len(events_df)

    print("[11/12] Building account interactions (relationship touchpoints)...")
    interactions_df = build_account_interactions(accounts_df, cfg, rng(seed + 10),
                                                  archetype_map=company_archetype_map)
    interactions_df.to_csv(DATA_RAW / "account_interactions.csv", index=False)
    manifest["tables"]["account_interactions"] = len(interactions_df)

    print("[12/12] Building renewal quotes (~120 days before term_end)...")
    quotes_df = build_renewal_quotes(items_df, orders_df, cfg, rng(seed + 11),
                                      exp_state_map=company_exp_states)
    quotes_df.to_csv(DATA_RAW / "renewal_quotes.csv", index=False)
    manifest["tables"]["renewal_quotes"] = len(quotes_df)

    # Outcome class check
    n_churn_events = (events_df["churned"] == 1).sum()
    n_labeled_churn = events_df[events_df["label_matured"] == True]["churned"].sum()
    manifest["outcome_check"] = {
        "total_cancellation_events": int(n_churn_events),
        "labeled_churn_events":      int(n_labeled_churn),
        "note": "Should be > 0; if 0, label-join bug not fixed"
    }

    elapsed = round(time.time() - t0, 1)
    manifest["runtime_seconds"] = elapsed

    # Per-file SHA-256 hashes — downstream coherence gates require all expected
    # files to be present and hashed.  Missing files are a hard failure here so
    # the manifest is never written with an incomplete inventory.
    missing_outputs = [f for f in sorted(_EXPECTED_HASH_FILES) if not (DATA_RAW / f).exists()]
    if missing_outputs:
        raise RuntimeError(
            "[hashes] Expected output files were not produced:\n"
            + "\n".join(f"  {f}" for f in missing_outputs)
        )
    manifest["file_hashes"] = {
        fname: _sha256_file(DATA_RAW / fname)
        for fname in sorted(_EXPECTED_HASH_FILES)
    }

    with open(ROOT / "outputs" / "generation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=lambda o: int(o) if hasattr(o, "item") else str(o))

    print(f"\n✓ Generation complete in {elapsed}s")
    for k, v in manifest["tables"].items():
        print(f"  {k:<35} {v:>10,}")
    print(f"\n  Outcome check: {n_churn_events:,} cancellations | {n_labeled_churn:,} labeled churn events")
    if n_labeled_churn == 0:
        print("  [WARN] Zero labeled churn — check contract_events.expiring_contract_id linking")


if __name__ == "__main__":
    main()

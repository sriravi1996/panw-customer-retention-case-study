#!/usr/bin/env bash
# run_part2.sh
# Sequential runner for Part 2: In-Depth Analysis & Modeling
# Must be run AFTER run_part1.sh has completed successfully.
# Usage:
#   bash part2/run_part2.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo ""
echo "========================================"
echo "  Part 2 | In-Depth Analysis & Modeling"
echo "========================================"

# ── Dependencies ────────────────────────────────────────────────────────────
echo ""
echo "[0/9] Installing Part 2 dependencies..."
pip install -q seaborn pytest

# ── Regression suite (GROUP A unit tests only — run before pipeline) ─────────
echo ""
echo "[0/9] Running business-logic unit tests (GROUP A: no file I/O)..."
python3 -m pytest part2/src/test_regression_fixtures.py -v --tb=short \
    -k "TestArrBridgeGRR or TestExpansionDecomposition or TestCrossSellFirstOrderExclusion"
echo "  Unit tests passed."

# ── Preflight: confirm Part 1 outputs exist ──────────────────────────────────
echo ""
echo "[0/9] Checking Part 1 outputs..."
for f in \
    "data/cleaned/renewal_unit_features.csv" \
    "data/cleaned/company_summary.csv" \
    "data/cleaned/order_item_fact.csv" \
    "data/cleaned/deployment_week_features.csv" \
    "outputs/generation_manifest.json"
do
    if [[ ! -f "$ROOT/$f" ]]; then
        echo "  ERROR: $f not found. Run run_part1.sh first."
        exit 1
    fi
done
echo "  Part 1 outputs verified."

# ── WP2: Build Features ──────────────────────────────────────────────────────
echo ""
echo "[2/9] WP2: Building renewal cohort features..."
python3 part2/src/build_features.py

# ── WP3: Analyze Cohorts ─────────────────────────────────────────────────────
echo ""
echo "[3/9] WP3: Cohort retention and ARR bridge analysis..."
python3 part2/src/analyze_cohorts.py

# ── WP4: Build Segments (RETIRED from main runner) ───────────────────────────
# build_segments.py requires renewal_cohort.csv with as_of_date=="2026-10-01",
# which is a genuine point-in-time snapshot (all features measured on that date).
# The modeling dataset uses per-contract pred_date and carries no as_of_date.
# Segmentation outputs are not used by the main presentation story; the April
# portfolio review (WP10) supersedes this analysis.  To regenerate October
# segmentation, produce a dedicated snapshot via src/integrate.py at 2026-10-01
# and save it separately (e.g. renewal_snapshot_2026_10_01.csv).

# ── WP5: Make Splits ─────────────────────────────────────────────────────────
echo ""
echo "[5/9] WP5: Temporal split manifest..."
python3 part2/src/make_splits.py

# ── WP6: Train Models ────────────────────────────────────────────────────────
echo ""
echo "[6/9] WP6: Training models (prevalence baseline, rule baseline, LR, HGB)..."
python3 part2/src/train.py

# ── WP7: Evaluate ────────────────────────────────────────────────────────────
echo ""
echo "[7/9] WP7: Evaluation (AP, ROC-AUC, Brier, Top-K, ablation, adversarial)..."
python3 part2/src/evaluate.py

# ── WP8: Analyze Growth ──────────────────────────────────────────────────────
echo ""
echo "[8/9] WP8: Expansion decomposition and exploratory product affinity..."
python3 part2/src/analyze_growth.py

# ── WP9: Prioritize (RETIRED from main runner) ───────────────────────────────
# prioritize.py shares the same October-snapshot dependency as build_segments.py
# (filters renewal_cohort.csv for as_of_date=="2026-10-01").  The April
# portfolio action queue from build_review_cohort.py (WP10) is the live output.

# ── WP10: Build Review Cohort ────────────────────────────────────────────────
echo ""
echo "[10/10] WP10: 90-day portfolio review cohort and policy comparison..."
python3 part2/src/build_review_cohort.py

# ── Run manifest ─────────────────────────────────────────────────────────────
echo ""
echo "[manifest] Writing run_manifest.json..."
python3 - "$ROOT" <<'PYEOF'
import hashlib, json, platform, sys, datetime
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

inputs = [
    "data/cleaned/company_summary.csv",
    "data/cleaned/order_item_fact.csv",
    "data/cleaned/deployments_clean.csv",
    "data/cleaned/telemetry_weekly_clean.csv",
    "data/cleaned/support_tickets_clean.csv",
    "data/cleaned/account_interactions_clean.csv",
    "data/cleaned/renewal_quotes_clean.csv",
    "data/cleaned/contract_events_clean.csv",
    "config/simulation.yaml",
    "part2/docs/customer_management_scope.md",
    "outputs/generation_manifest.json",
]
outputs = [
    "data/analysis/renewal_cohort.csv",
    "data/analysis/renewal_cohort_subscriptions.csv",
    "part2/outputs/split_cohort_full.csv",
    "part2/outputs/arr_bridge.csv",
    "part2/outputs/lineage_reconciliation.csv",
    "part2/outputs/cohort_retention.csv",
    "part2/outputs/metrics.json",
    "part2/outputs/feature_importance.csv",
    "part2/outputs/model_selection_record.json",
    "part2/outputs/boundary_purged.csv",
    "part2/outputs/cohort_eligibility_waterfall.csv",
    "part2/outputs/renewal_review_policy_comparison.csv",
    "part2/outputs/renewal_review_contracts.csv",
    "part2/outputs/renewal_review_accounts.csv",
    "part2/outputs/renewal_review_policy_diffs.csv",
    "part2/outputs/imputation_values.json",
    "part2/outputs/analyze_growth_summary.json",
    "part2/outputs/timing_audit.json",
    "part2/outputs/ablation.csv",
    "part2/outputs/eval_topk.csv",
    "part2/docs/renewal_review_examples.md",
]

import subprocess
git_sha = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True, cwd=ROOT
).stdout.strip() or "unknown"

# Pull key stats from metrics.json and model_selection_record.json
try:
    with open(ROOT / "part2/outputs/metrics.json") as f:
        metrics = json.load(f)
    # Keys use experiment_type format: full_context_hgb, full_context_lr, rule_baseline, etc.
    hgb_ms_ap  = metrics.get("full_context_hgb",  {}).get("model_selection", {}).get("ap", None)
    rule_ms_ap = metrics.get("rule_baseline",      {}).get("model_selection", {}).get("ap", None)
except Exception:
    hgb_ms_ap = rule_ms_ap = None

# Read selected model info from selection record
try:
    with open(ROOT / "part2/outputs/model_selection_record.json") as f:
        sel_rec = json.load(f)
    selected_model_name    = sel_rec.get("deployment_policy", None)
    selected_model_ms_ap   = sel_rec.get("selected_model", {}).get("ms_ap", None)
    selected_model_artifact = sel_rec.get("refit", {}).get("artifact", None)
except Exception:
    selected_model_name = selected_model_ms_ap = selected_model_artifact = None

if not selected_model_artifact:
    raise ValueError("Selected refit artifact is missing from selection record")
outputs.append("part2/models/" + selected_model_artifact)
code_paths = sorted((ROOT / "part2" / "src").glob("*.py"))
code_paths.append(ROOT / "part2" / "run_part2.sh")

import pandas as pd
bridge = pd.read_csv(ROOT / "part2/outputs/arr_bridge.csv")
total_rows = bridge[bridge["segment"] == "total"]
if len(total_rows) == 0:
    import sys
    print(
        "FATAL: segment='total' not found in arr_bridge.csv. "
        "Found segments: " + str(bridge["segment"].tolist()) + ". "
        "Check analyze_cohorts.py output.",
        file=sys.stderr
    )
    sys.exit(1)
if len(total_rows) > 1:
    import sys
    print(
        f"FATAL: Multiple 'total' rows in arr_bridge.csv ({len(total_rows)} rows). "
        "Duplicate segment entries — check analyze_cohorts.py.",
        file=sys.stderr
    )
    sys.exit(1)
total_row = total_rows.iloc[0]
key_stats = {
    "opening_arr": float(total_row["opening_arr"]),
    "closing_arr": float(total_row["closing_arr"]),
    "grr": float(total_row["grr"]),
    "nrr": float(total_row["nrr"]),
    "residual": float(total_row["residual"]),
}

manifest = {
    "run_timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "git_sha": git_sha,
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "input_hashes": {p: sha256(ROOT / p) for p in inputs},
    "output_hashes": {p: sha256(ROOT / p) for p in outputs},
    "code_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
    "pipeline_note": "Generated after all active Part 2 stages in this runner completed. Presentation exports have a separate validation record.",
    "key_stats": key_stats,
    "model_selection_metrics": {
        "full_context_hgb_ap": hgb_ms_ap,
        "rule_baseline_ap": rule_ms_ap,
        "selected_model": selected_model_name,
        "selected_model_ms_ap": selected_model_ms_ap,
        "selected_model_artifact": selected_model_artifact,
    },
    "data_note": "All data is synthesized for case-study purposes. Performance on real PANW data will differ.",
}

out_path = ROOT / "part2/outputs/run_manifest.json"
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"  Written -> part2/outputs/run_manifest.json")
PYEOF

# ── Regression suite (full — runs against fresh artifacts) ───────────────────
echo ""
echo "[post] Running full regression suite against fresh outputs..."
python3 -m pytest part2/src/test_regression_fixtures.py -v --tb=short
echo "  Regression tests passed."

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Part 2 complete."
echo "========================================"
echo ""
echo "  Analysis outputs:     part2/outputs/"
echo "  Figures:              part2/figures/"
echo "  Models:               part2/models/"
echo "  Documentation:        part2/docs/"
echo ""
echo "  Key files:"
echo "    Renewal cohort:        data/analysis/renewal_cohort_subscriptions.csv"
echo "    Split manifest:        part2/outputs/split_manifest.json"
echo "    Cohort retention:      part2/outputs/cohort_retention.csv"
echo "    ARR bridge:            part2/outputs/arr_bridge.csv"
echo "    Review accounts:       part2/outputs/renewal_review_accounts.csv"
echo "    Metrics:               part2/outputs/metrics.json"
echo "    Top-K eval:            part2/outputs/eval_topk.csv"
echo "    Ablation:              part2/outputs/ablation.csv"
echo "    Expansion analysis:    part2/outputs/renewal_expansion_analysis.csv"
echo "    Product affinity:      part2/outputs/exploratory_product_candidates.csv"
echo "    Review contracts:      part2/outputs/renewal_review_contracts.csv"
echo "    Policy comparison:     part2/outputs/renewal_review_policy_comparison.csv"
echo "    Pilot protocol:        part2/docs/pilot_protocol.md"
echo "    Run manifest:          part2/outputs/run_manifest.json"
echo ""

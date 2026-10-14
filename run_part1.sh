#!/usr/bin/env bash
# run_part1.sh
# One-command runner for Part 1: Data Engineering & EDA
# Usage:
#   bash run_part1.sh          # configured scale (10k accounts)
#   bash run_part1.sh --dev    # dev mode  (2k companies)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DEV_FLAG=""
if [[ "${1:-}" == "--dev" ]]; then
  DEV_FLAG="--dev"
  echo "========================================"
  echo "  Part 1 | DEV mode (2,000 companies)"
  echo "========================================"
else
  echo "========================================"
  echo "  Part 1 | Full scale (10,000 companies)"
  echo "========================================"
fi

# ── Environment ────────────────────────────────────────────────────────────────
echo ""
echo "[0/4] Installing dependencies..."
pip install -q -r requirements.txt

# ── Step 1: Generate data ──────────────────────────────────────────────────────
echo ""
echo "[1/4] Generating synthetic data..."
python3 src/generate_data.py $DEV_FLAG

# ── Step 2: Ingest and clean ───────────────────────────────────────────────────
echo ""
echo "[2/4] Ingesting and cleaning CSVs..."
python3 src/ingest_clean.py

# ── Step 3: Integrate ──────────────────────────────────────────────────────────
echo ""
echo "[3/4] Building analytical datasets..."
python3 src/integrate.py

# ── Step 4: EDA ───────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Running EDA..."
python3 src/eda.py

# ── Step 5: Validate ──────────────────────────────────────────────────────────
echo ""
echo "[5/5] Running validation suite (35 fixtures)..."
python3 src/validate.py

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Part 1 complete."
echo "========================================"
echo ""
echo "  Raw data:       data/raw/"
echo "  Cleaned data:   data/cleaned/"
echo "  Quality report: outputs/quality_report.json"
echo "  Integration:    outputs/integration_lineage.json"
echo "  EDA answers:    outputs/eda_summary.json"
echo "  Top 20 table:   outputs/eda_top20_ordered_items.csv"
echo "  AOV table:      outputs/eda_aov_analysis.csv"
echo "  Charts:         figures/"
echo "  Interactions:   data/cleaned/account_interactions_clean.csv"
echo ""

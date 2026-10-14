"""
train.py — Fit all four approaches across three declared feature experiments.

Four approaches (§4 of implementation plan):
  1. Prevalence baseline     — training cancellation rate applied to every row
  2. Operational rule        — _rule_baseline.py (experience-led; frozen formula)
  3. Logistic regression     — regularized; predeclared C × class_weight grid
  4. HistGradientBoosting    — shallow; predeclared depth/iter × class_weight grid

Three feature experiments (§2 of implementation plan):
  experience_led  — health/service/commercial/relationship + product_family
  size_value      — company size, ACV, renewal timing, portfolio count
  full_context    — experience + size/value + breadth + relationship context

Model selection criterion: highest Average Precision on the model_selection split
(2025-04-01). Prefer the simpler candidate when differences are within tolerance.

Predeclared hyperparameter grids (do not expand after seeing any evaluation data):
  LR:  C ∈ {0.01, 0.1, 1.0}  ×  class_weight ∈ {None, 'balanced'}  = 6 configs
  HGB: 4 shallow configs × class_weight ∈ {None, 'balanced'}          = 8 configs

Outputs
-------
part2/models/
  prevalence_baseline.json
  rule_baseline.json
  {experiment}_lr.pkl          — best LR config per experiment
  {experiment}_hgb.pkl         — best HGB config per experiment
part2/outputs/
  imputation_values.json       — train-fitted median for days_since_last_interaction
  training_config.json         — feature sets, grid, training counts
  model_selection_record.json  — all candidate APs; selected model
  candidate_selection_table.csv — full grid results for reporting
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

# Shared feature specification — do not duplicate lists here
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _features import (
    EXPERIENCE_NUM, EXPERIENCE_CAT,
    SIZE_VALUE_NUM, SIZE_VALUE_CAT,
    FULL_CONTEXT_NUM, FULL_CONTEXT_CAT,
    TARGET, PROHIBITED_PREDICTORS,
    add_indicator_cols, apply_imputation_values, make_preprocessor,
    assert_no_prohibited,
)
from _rule_baseline import apply_rule_baseline, RULE_CONFIG

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
# Read from split_cohort_full.csv (produced by make_splits.py with split column assigned)
# to keep renewal_cohort_subscriptions.csv immutable after build_features.py.
DATA_FILE  = ROOT / "part2" / "outputs" / "split_cohort_full.csv"
MODELS_DIR = ROOT / "part2" / "models"
OUT_DIR    = ROOT / "part2" / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("train.py — fit all approaches across three feature experiments")
print("=" * 70)

# ---------------------------------------------------------------------------
# Experiments: (name, numeric_features, categorical_features)
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "experience_led": (EXPERIENCE_NUM, EXPERIENCE_CAT),
    "size_value":     (SIZE_VALUE_NUM,  SIZE_VALUE_CAT),
    "full_context":   (FULL_CONTEXT_NUM, FULL_CONTEXT_CAT),
}

# ---------------------------------------------------------------------------
# Predeclared hyperparameter grids (frozen before any evaluation data is seen)
# ---------------------------------------------------------------------------
LR_GRID = [
    {"C": 0.01,  "class_weight": None},
    {"C": 0.01,  "class_weight": "balanced"},
    {"C": 0.1,   "class_weight": None},
    {"C": 0.1,   "class_weight": "balanced"},
    {"C": 1.0,   "class_weight": None},
    {"C": 1.0,   "class_weight": "balanced"},
]

# Shallow HGB with bounded iteration counts; early_stopping=False avoids
# automatic random holdout splits that vary run to run.
HGB_GRID = [
    {"max_depth": 3, "max_leaf_nodes": 15, "min_samples_leaf": 20, "max_iter": 100, "class_weight": None},
    {"max_depth": 3, "max_leaf_nodes": 15, "min_samples_leaf": 20, "max_iter": 100, "class_weight": "balanced"},
    {"max_depth": 4, "max_leaf_nodes": 31, "min_samples_leaf": 10, "max_iter": 150, "class_weight": None},
    {"max_depth": 4, "max_leaf_nodes": 31, "min_samples_leaf": 10, "max_iter": 150, "class_weight": "balanced"},
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"\n[1] Loading cohort from {DATA_FILE.relative_to(ROOT)}")
df_all = pd.read_csv(DATA_FILE)
print(f"    {len(df_all):,} rows loaded")

# Extract train and model_selection splits
df_train = df_all[df_all["split"] == "train"].copy()
df_ms    = df_all[df_all["split"] == "model_selection"].copy()

if len(df_train) == 0:
    raise ValueError("Train split is empty. Run make_splits.py first.")
if len(df_ms) == 0:
    raise ValueError("model_selection split is empty. Run make_splits.py first.")

# ---------------------------------------------------------------------------
# Verify no prohibited predictors in loaded data (before add_indicator_cols)
# ---------------------------------------------------------------------------
# churned is the target — must be extracted before X is formed; checked in per-exp block

# ---------------------------------------------------------------------------
# Add indicator columns and apply NULL fills (shared helper)
# ---------------------------------------------------------------------------
print("\n[2] Adding indicator columns and applying NULL fills")
df_train = add_indicator_cols(df_train)
df_ms    = add_indicator_cols(df_ms)

# Fit train-only imputation for days_since_last_interaction
_dsi_median = float(df_train["days_since_last_interaction"].median())
imputation_values = {
    "days_since_last_interaction_median": _dsi_median,
    "note": (
        "Fitted on training split only (2023-10-01 + 2024-10-01). "
        "Load from this file in evaluate.py and prioritize.py. "
        "Do not refit outside training."
    ),
}
with open(OUT_DIR / "imputation_values.json", "w") as f:
    json.dump(imputation_values, f, indent=2)
print(f"    days_since_last_interaction train median: {_dsi_median:.1f} days")
print(f"    Saved: part2/outputs/imputation_values.json")

df_train = apply_imputation_values(df_train, imputation_values)
df_ms    = apply_imputation_values(df_ms,    imputation_values)

y_train = df_train[TARGET].astype(int).values
y_ms    = df_ms[TARGET].astype(int).values

n_train   = len(y_train)
n_train_p = int(y_train.sum())
prevalence = float(y_train.mean())

n_ms    = len(y_ms)
n_ms_p  = int(y_ms.sum())

print(f"\n    Train: {n_train} rows, {n_train_p} positives, prevalence={prevalence:.4f}")
print(f"    MS:    {n_ms} rows, {n_ms_p} positives, prevalence={float(y_ms.mean()):.4f}")

# ---------------------------------------------------------------------------
# Model 1 — Prevalence baseline (same for all experiments)
# ---------------------------------------------------------------------------
print("\n[3] Prevalence baseline")
prevalence_model = {"type": "prevalence", "value": prevalence}
with open(MODELS_DIR / "prevalence_baseline.json", "w") as f:
    json.dump(prevalence_model, f, indent=2)
ms_prevalence_scores = np.full(n_ms, prevalence)
ms_ap_prevalence = average_precision_score(y_ms, ms_prevalence_scores)
print(f"    model_selection AP: {ms_ap_prevalence:.4f}")

# ---------------------------------------------------------------------------
# Model 2 — Rule baseline (same for all experiments; uses own 5 input columns)
# ---------------------------------------------------------------------------
print("\n[4] Rule baseline")
rule_model = {
    "type": "rule_baseline",
    "weights": RULE_CONFIG["weights"],
    "normalization": RULE_CONFIG["normalization"],
    "removed_term": RULE_CONFIG["removed_term"],
    "note": "Import apply_rule_baseline from _rule_baseline.py; do not re-implement.",
}
with open(MODELS_DIR / "rule_baseline.json", "w") as f:
    json.dump(rule_model, f, indent=2)
ms_rule_scores = apply_rule_baseline(df_ms)
ms_ap_rule = average_precision_score(y_ms, ms_rule_scores)
print(f"    model_selection AP: {ms_ap_rule:.4f}")

# ---------------------------------------------------------------------------
# Grid search across experiments and ML model types
# ---------------------------------------------------------------------------
print("\n[5] Fitting ML models across three feature experiments")
print(f"    LR grid:  {len(LR_GRID)} configs")
print(f"    HGB grid: {len(HGB_GRID)} configs")
print(f"    Total ML fits: {len(EXPERIMENTS) * (len(LR_GRID) + len(HGB_GRID))}\n")

candidate_rows = []  # collects all (experiment, model, config, train_ap, ms_ap)

# Per-experiment best artifacts (best AP from grid; saved as .pkl)
exp_best: dict = {}  # key=(exp, model_type), value=(ms_ap, pipeline, config)

for exp_name, (num_feats, cat_feats) in EXPERIMENTS.items():
    print(f"  ── Experiment: {exp_name} ──────────────────────────────────────────")

    # Slice feature columns — assert no prohibited predictors end up in X
    X_tr = df_train[num_feats + cat_feats].copy()
    X_ms_ = df_ms[num_feats + cat_feats].copy()
    assert_no_prohibited(X_tr, f"train/{exp_name}")

    # ── LR grid ────────────────────────────────────────────────────────────
    for cfg in LR_GRID:
        preprocessor = make_preprocessor(num_feats, cat_feats, model_type="lr")
        clf = LogisticRegression(
            C=cfg["C"],
            class_weight=cfg["class_weight"],
            max_iter=500,
            random_state=42,
            solver="lbfgs",
        )
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        pipe.fit(X_tr, y_train)

        tr_ap = average_precision_score(y_train, pipe.predict_proba(X_tr)[:, 1])
        ms_ap_ = average_precision_score(y_ms,    pipe.predict_proba(X_ms_)[:, 1])

        cfg_str = f"C={cfg['C']} cw={cfg['class_weight']}"
        print(f"    LR  {cfg_str:<25}  train={tr_ap:.4f}  ms={ms_ap_:.4f}")

        candidate_rows.append({
            "experiment": exp_name, "model_type": "lr",
            "config":      cfg_str,
            "C":           cfg["C"], "class_weight": str(cfg["class_weight"]),
            "train_ap":   round(tr_ap,  6),
            "ms_ap":      round(ms_ap_, 6),
        })

        key = (exp_name, "lr")
        if key not in exp_best or ms_ap_ > exp_best[key][0]:
            exp_best[key] = (ms_ap_, pipe, cfg)

    # ── HGB grid ───────────────────────────────────────────────────────────
    for cfg in HGB_GRID:
        preprocessor = make_preprocessor(num_feats, cat_feats, model_type="hgb")
        cw = cfg["class_weight"]
        try:
            clf = HistGradientBoostingClassifier(
                max_depth=cfg["max_depth"],
                max_leaf_nodes=cfg["max_leaf_nodes"],
                min_samples_leaf=cfg["min_samples_leaf"],
                max_iter=cfg["max_iter"],
                learning_rate=0.05,
                early_stopping=False,
                random_state=42,
                class_weight=cw,
            )
            pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
            pipe.fit(X_tr, y_train)
        except TypeError:
            # Older sklearn versions don't support class_weight in HGB
            clf = HistGradientBoostingClassifier(
                max_depth=cfg["max_depth"],
                max_leaf_nodes=cfg["max_leaf_nodes"],
                min_samples_leaf=cfg["min_samples_leaf"],
                max_iter=cfg["max_iter"],
                learning_rate=0.05,
                early_stopping=False,
                random_state=42,
            )
            pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
            if cw == "balanced":
                n_neg = (y_train == 0).sum()
                n_pos = (y_train == 1).sum()
                sw = np.where(y_train == 1, n_neg / n_pos, 1.0)
                X_tr_t = pipe.named_steps["pre"].fit_transform(X_tr)
                pipe.named_steps["clf"].fit(X_tr_t, y_train, sample_weight=sw)
            else:
                pipe.fit(X_tr, y_train)

        tr_ap  = average_precision_score(y_train, pipe.predict_proba(X_tr)[:, 1])
        ms_ap_ = average_precision_score(y_ms,    pipe.predict_proba(X_ms_)[:, 1])

        cfg_str = (f"d={cfg['max_depth']} l={cfg['max_leaf_nodes']} "
                   f"s={cfg['min_samples_leaf']} i={cfg['max_iter']} cw={cw}")
        print(f"    HGB {cfg_str:<40}  train={tr_ap:.4f}  ms={ms_ap_:.4f}")

        candidate_rows.append({
            "experiment":    exp_name,
            "model_type":    "hgb",
            "config":        cfg_str,
            "max_depth":     cfg["max_depth"],
            "max_leaf_nodes": cfg["max_leaf_nodes"],
            "min_samples_leaf": cfg["min_samples_leaf"],
            "max_iter":      cfg["max_iter"],
            "class_weight":  str(cw),
            "train_ap":      round(tr_ap,  6),
            "ms_ap":         round(ms_ap_, 6),
        })

        key = (exp_name, "hgb")
        if key not in exp_best or ms_ap_ > exp_best[key][0]:
            exp_best[key] = (ms_ap_, pipe, cfg)

# ---------------------------------------------------------------------------
# Save best model per (experiment, model_type)
# ---------------------------------------------------------------------------
print("\n[6] Saving best model per experiment")
saved_models = {}
for (exp_name, model_type), (best_ap, best_pipe, best_cfg) in sorted(exp_best.items()):
    fname = f"{exp_name}_{model_type}.pkl"
    joblib.dump(best_pipe, MODELS_DIR / fname)
    saved_models[f"{exp_name}_{model_type}"] = {
        "file":       fname,
        "ms_ap":      round(best_ap, 6),
        "config":     {k: str(v) for k, v in best_cfg.items()},
        "num_feats":  EXPERIMENTS[exp_name][0],
        "cat_feats":  EXPERIMENTS[exp_name][1],
    }
    print(f"    {fname:<40}  ms_ap={best_ap:.4f}")

# Also save legacy names for backward compatibility with evaluate.py/prioritize.py
# The overall selected model is identified by model_selection_record.json
# Legacy: logistic_regression.pkl and hist_gradient_boosting.pkl point to
# the best overall model from the experience_led experiment (primary experiment).
for mt, lname in [("lr", "logistic_regression"), ("hgb", "hist_gradient_boosting")]:
    key = ("experience_led", mt)
    if key in exp_best:
        _, pipe, _ = exp_best[key]
        joblib.dump(pipe, MODELS_DIR / f"{lname}.pkl")
print("    (legacy: logistic_regression.pkl, hist_gradient_boosting.pkl → experience_led best)")

# ---------------------------------------------------------------------------
# Model selection record
# ---------------------------------------------------------------------------
print("\n[7] Building model selection record")

# All candidate AP scores on model_selection
all_ms_aps = [row["ms_ap"] for row in candidate_rows]
best_ml_row = max(candidate_rows, key=lambda r: r["ms_ap"])
best_overall_ap = best_ml_row["ms_ap"]

# Per-experiment best AP summary
exp_summary = {}
for exp_name in EXPERIMENTS:
    exp_rows = [r for r in candidate_rows if r["experiment"] == exp_name]
    best_exp = max(exp_rows, key=lambda r: r["ms_ap"])
    exp_summary[exp_name] = {
        "best_ms_ap":   best_exp["ms_ap"],
        "best_config":  best_exp["config"],
        "model_type":   best_exp["model_type"],
    }

# Prespecified selection rule: highest model_selection AP among ML candidates.
# Prefer simpler model when difference is within 0.005 AP (practical tolerance).
SIMPLICITY_TOLERANCE = 0.005
selected_exp   = best_ml_row["experiment"]
selected_mt    = best_ml_row["model_type"]
selected_ap    = best_ml_row["ms_ap"]
selected_cfg   = best_ml_row["config"]

# Check if a simpler model is within tolerance
_lr_rows = [r for r in candidate_rows if r["model_type"] == "lr"]
if _lr_rows:
    best_lr = max(_lr_rows, key=lambda r: r["ms_ap"])
    if (selected_mt == "hgb" and
            abs(best_lr["ms_ap"] - selected_ap) <= SIMPLICITY_TOLERANCE):
        selected_exp = best_lr["experiment"]
        selected_mt  = "lr"
        selected_ap  = best_lr["ms_ap"]
        selected_cfg = best_lr["config"]
        print(f"    Simplicity preference applied: LR within {SIMPLICITY_TOLERANCE} AP of HGB")

# Rule vs ML comparison
rule_beats_ml = ms_ap_rule > selected_ap
deployment_policy = "rule_baseline" if rule_beats_ml else f"{selected_exp}_{selected_mt}"

model_selection_record = {
    "selection_criterion": "average_precision on model_selection split (2025-04-01)",
    "simplicity_tolerance": SIMPLICITY_TOLERANCE,
    "selected_model": {
        "experiment":  selected_exp,
        "model_type":  selected_mt,
        "ms_ap":       round(selected_ap, 6),
        "config":      selected_cfg,
        "artifact":    f"{selected_exp}_{selected_mt}.pkl",
        "num_features": EXPERIMENTS[selected_exp][0],
        "cat_features": EXPERIMENTS[selected_exp][1],
    },
    "deployment_policy": deployment_policy,
    "deployment_policy_note": (
        "rule_baseline has highest AP at model_selection. "
        "Priority index, not a calibrated probability. "
        "Validate independently before operational deployment."
        if rule_beats_ml else
        f"{selected_exp}_{selected_mt} used as deployment policy and ML challenger."
    ),
    "baseline_aps": {
        "prevalence": round(ms_ap_prevalence, 6),
        "rule":       round(ms_ap_rule,       6),
    },
    "experiment_best_aps": exp_summary,
    "final_test_note": (
        "Calendar windows were previously inspected in the original project. "
        "The current seed-137 population was generated after the synthetic scenario was frozen. "
        "This demonstrates known synthetic mechanisms, not real PANW performance."
    ),
    "train_counts": {
        "n_train":     n_train,
        "n_positives": n_train_p,
        "prevalence":  round(prevalence, 6),
    },
    "ms_counts": {
        "n_ms":        n_ms,
        "n_positives": n_ms_p,
        "prevalence":  round(float(y_ms.mean()), 6),
    },
}

with open(OUT_DIR / "model_selection_record.json", "w") as f:
    json.dump(model_selection_record, f, indent=2)
print(f"    Saved: part2/outputs/model_selection_record.json")

# ---------------------------------------------------------------------------
# Candidate selection table
# ---------------------------------------------------------------------------
cand_df = pd.DataFrame(candidate_rows)
cand_df.to_csv(OUT_DIR / "candidate_selection_table.csv", index=False)
print(f"    Saved: part2/outputs/candidate_selection_table.csv ({len(cand_df)} rows)")

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
training_config = {
    "part1_run_id": json.loads((ROOT / "outputs" / "generation_manifest.json").read_text())["run_id"],
    "train_dates": ["2023-10-01", "2024-10-01"],
    "n_train": n_train,
    "n_positives": n_train_p,
    "prevalence": round(prevalence, 6),
    "experiments": {k: {"numeric": v[0], "categorical": v[1]} for k, v in EXPERIMENTS.items()},
    "lr_grid": LR_GRID,
    "hgb_grid": HGB_GRID,
    "imputation_values": imputation_values,
}
with open(OUT_DIR / "training_config.json", "w") as f:
    json.dump(training_config, f, indent=2)
print(f"    Saved: part2/outputs/training_config.json")

# ---------------------------------------------------------------------------
# [8] Post-selection refit on train + model_selection combined
# Rationale: additional_train has only 33 usable rows / 2 positives — not worth
# the complexity. Simpler and more data-efficient: combine train (2,049) +
# model_selection (1,312) → 3,361 rows / 238 cancellations, then calibrate.
# ---------------------------------------------------------------------------
print("\n[8] Post-selection refit on train + model_selection combined")
print("    (additional_train excluded: 33 usable rows / 2 positives — negligible)")

df_refit = pd.concat([df_train, df_ms], ignore_index=True)
y_refit  = df_refit[TARGET].astype(int).values
n_refit  = len(y_refit)
n_refit_p = int(y_refit.sum())
print(f"    Refit rows: {n_refit}, positives: {n_refit_p}, "
      f"prevalence={n_refit_p/n_refit:.4f}")

sel_num_r, sel_cat_r = EXPERIMENTS[selected_exp]
X_refit = df_refit[sel_num_r + sel_cat_r].copy()
assert_no_prohibited(X_refit, f"refit/{selected_exp}")

_, _, sel_cfg_r = exp_best[(selected_exp, selected_mt)]
preprocessor_r = make_preprocessor(sel_num_r, sel_cat_r, model_type=selected_mt)

if selected_mt == "lr":
    clf_r = LogisticRegression(
        C=sel_cfg_r["C"],
        class_weight=sel_cfg_r["class_weight"],
        max_iter=500,
        random_state=42,
        solver="lbfgs",
    )
    refitted_pipe = Pipeline([("pre", preprocessor_r), ("clf", clf_r)])
    refitted_pipe.fit(X_refit, y_refit)
else:  # hgb
    cw_r = sel_cfg_r["class_weight"]
    try:
        clf_r = HistGradientBoostingClassifier(
            max_depth=sel_cfg_r["max_depth"],
            max_leaf_nodes=sel_cfg_r["max_leaf_nodes"],
            min_samples_leaf=sel_cfg_r["min_samples_leaf"],
            max_iter=sel_cfg_r["max_iter"],
            learning_rate=0.05,
            early_stopping=False,
            random_state=42,
            class_weight=cw_r,
        )
        refitted_pipe = Pipeline([("pre", preprocessor_r), ("clf", clf_r)])
        refitted_pipe.fit(X_refit, y_refit)
    except TypeError:
        clf_r = HistGradientBoostingClassifier(
            max_depth=sel_cfg_r["max_depth"],
            max_leaf_nodes=sel_cfg_r["max_leaf_nodes"],
            min_samples_leaf=sel_cfg_r["min_samples_leaf"],
            max_iter=sel_cfg_r["max_iter"],
            learning_rate=0.05,
            early_stopping=False,
            random_state=42,
        )
        refitted_pipe = Pipeline([("pre", preprocessor_r), ("clf", clf_r)])
        if cw_r == "balanced":
            n_neg_r = (y_refit == 0).sum()
            n_pos_r = (y_refit == 1).sum()
            sw_r = np.where(y_refit == 1, n_neg_r / n_pos_r, 1.0)
            X_r_t = preprocessor_r.fit_transform(X_refit)
            clf_r.fit(X_r_t, y_refit, sample_weight=sw_r)
        else:
            refitted_pipe.fit(X_refit, y_refit)

refit_fname = f"{selected_exp}_{selected_mt}_refitted.pkl"
joblib.dump(refitted_pipe, MODELS_DIR / refit_fname)
print(f"    Saved: {refit_fname}")

# Legacy refitted name for downstream callers
legacy_refit = f"{'logistic_regression' if selected_mt == 'lr' else 'hist_gradient_boosting'}_refitted.pkl"
joblib.dump(refitted_pipe, MODELS_DIR / legacy_refit)
print(f"    (legacy alias: {legacy_refit})")

# Persist refit metadata in model_selection_record.json
model_selection_record["refit"] = {
    "description": "post-selection refit on train + model_selection combined",
    "n_refit": n_refit,
    "n_refit_positives": n_refit_p,
    "artifact": refit_fname,
    "note": (
        f"Combined train and model_selection: {n_refit:,} rows, {n_refit_p:,} cancellations. "
        "No calibration or final-test labels are included in this refit."
    ),
}
with open(OUT_DIR / "model_selection_record.json", "w") as f:
    json.dump(model_selection_record, f, indent=2)
print(f"    Updated: part2/outputs/model_selection_record.json (refit info added)")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("train.py — COMPLETE")
print("=" * 70)
print(f"  Prevalence baseline MS AP : {ms_ap_prevalence:.4f}")
print(f"  Rule baseline     MS AP   : {ms_ap_rule:.4f}")
print(f"\n  Experiment best MS APs:")
for exp_name, info in exp_summary.items():
    marker = " ← selected" if exp_name == selected_exp and info["model_type"] == selected_mt else ""
    print(f"    {exp_name:<18} {info['model_type']:<4}  {info['best_ms_ap']:.4f}  {info['best_config']}{marker}")
print(f"\n  Selected model: {selected_exp}_{selected_mt}  AP={selected_ap:.4f}")
print(f"  Deployment policy: {deployment_policy}")
print(f"  Rule beats best ML: {rule_beats_ml}")
print("=" * 70)

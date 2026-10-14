"""
evaluate.py  –  WP7 PANW Customer Churn Analysis: Part 2 Evaluation

Evaluates all approaches across all three feature experiments on
model_selection, calibration, and final_test splits.

Feature lists and preprocessing are imported from _features.py — do NOT
duplicate them here. The selected deployment model is identified from
model_selection_record.json.

Evaluation flow (per §4 of implementation plan):
  • Baselines: prevalence, operational rule
  • All 6 ML models: 3 experiments × {lr, hgb}
  • Selected (refitted) model: highlighted in output
  • K_VALS = [25, 50, 100] absolute accounts per outreach wave
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

# ---------------------------------------------------------------------------
# Shared feature spec — do NOT inline feature lists here
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from _features import (
    EXPERIENCE_NUM, EXPERIENCE_CAT,
    SIZE_VALUE_NUM, SIZE_VALUE_CAT,
    FULL_CONTEXT_NUM, FULL_CONTEXT_CAT,
    TARGET, PROHIBITED_PREDICTORS,
    add_indicator_cols, apply_imputation_values, assert_no_prohibited,
    make_preprocessor,
)
from _rule_baseline import apply_rule_baseline  # canonical rule; do NOT re-implement

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
# Read from split_cohort_full.csv (produced by make_splits.py with split column assigned)
# to keep renewal_cohort_subscriptions.csv immutable after build_features.py.
DATA_PATH = ROOT / "part2" / "outputs" / "split_cohort_full.csv"
MODELS_DIR = ROOT / "part2" / "models"
OUTPUTS_DIR = ROOT / "part2" / "outputs"
FIGURES_DIR = ROOT / "part2" / "figures"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature sets per experiment (keyed by model artifact name)
# ---------------------------------------------------------------------------
EXPERIMENT_FEATURE_SETS: dict[str, tuple[list[str], list[str]]] = {
    "experience_led_lr":  (EXPERIENCE_NUM, EXPERIENCE_CAT),
    "experience_led_hgb": (EXPERIENCE_NUM, EXPERIENCE_CAT),
    "size_value_lr":      (SIZE_VALUE_NUM, SIZE_VALUE_CAT),
    "size_value_hgb":     (SIZE_VALUE_NUM, SIZE_VALUE_CAT),
    "full_context_lr":    (FULL_CONTEXT_NUM, FULL_CONTEXT_CAT),
    "full_context_hgb":   (FULL_CONTEXT_NUM, FULL_CONTEXT_CAT),
}

EVAL_SPLITS = ["model_selection", "calibration", "final_test"]

# Absolute capacity scenarios per plan §5 (top K accounts per scoring wave)
K_VALS = [25, 50, 100]

# ---------------------------------------------------------------------------
# Ablation blocks (one removal per block; run on selected experiment only)
# missing_telemetry_pct_90d is EXCLUDED from all feature sets — not ablated.
# ---------------------------------------------------------------------------
ABLATION_BLOCKS_EXPERIENCE = {
    "block_deployment": [
        "deployment_reach_30d", "deployment_reach_30d_missing",
        "onboarding_delay_days",
    ],
    "block_value": [
        "workflow_success_30d", "workflow_success_30d_missing",
        "productive_use_delta_30d",
    ],
    "block_telemetry": [
        "avg_coverage_90d", "coverage_trend_90d", "obs_completeness_90d",
    ],
    "block_service": [
        "unresolved_customer_impact_hours_90d", "repeat_sla_breaches_90d",
        "p1_tickets_90d", "open_tickets_90d", "avg_open_age_hrs_90d",
        "avg_resolution_hrs_90d", "avg_resolution_hrs_90d_missing",
    ],
    "block_commercial": [
        "quoted_unit_price_change_pct", "dated_budget_concern",
        "days_to_renewal", "tenure_days",
    ],
    "block_relationship": [
        "days_without_confirmed_owner", "overdue_agreed_actions",
        "days_since_last_interaction_missing", "median_prior_renewal_delay_missing",
    ],
}

ABLATION_BLOCKS_SIZE_VALUE = {
    "block_size": ["employee_count"],
    "block_acv": ["acv", "total_acv_in_window"],
    "block_portfolio": [
        "n_products_in_window", "n_families_in_window",
        "days_to_renewal", "tenure_days",
    ],
}

ABLATION_BLOCKS: dict[str, dict] = {
    "experience_led": ABLATION_BLOCKS_EXPERIENCE,
    "size_value": ABLATION_BLOCKS_SIZE_VALUE,
    "full_context": ABLATION_BLOCKS_EXPERIENCE,  # ablate same blocks as experience subset
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load cohort CSV and apply shared preprocessing helpers."""
    df = pd.read_csv(DATA_PATH)
    # add_indicator_cols: creates _missing indicators and applies NULL_FILLS
    df = add_indicator_cols(df)
    # apply_imputation_values: loads train-fitted median for days_since_last_interaction
    imp = _load_imputation_values()
    df = apply_imputation_values(df, imp)
    return df


def _load_imputation_values() -> dict:
    path = OUTPUTS_DIR / "imputation_values.json"
    if not path.exists():
        raise FileNotFoundError(
            f"imputation_values.json not found at {path}. "
            "Run train.py first to generate train-fitted imputation values. "
            "Using a hardcoded fallback would create a train/eval mismatch."
        )
    with open(path) as f:
        return json.load(f)


def load_selection_record() -> dict:
    path = OUTPUTS_DIR / "model_selection_record.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def get_split(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    sub = df[df["split"] == split].copy()
    labels = sub[TARGET].values.astype(float)
    return sub, labels


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_prevalence_baseline() -> dict:
    with open(MODELS_DIR / "prevalence_baseline.json") as f:
        return json.load(f)


def load_rule_baseline() -> dict:
    with open(MODELS_DIR / "rule_baseline.json") as f:
        return json.load(f)


def load_sklearn_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    if path.exists():
        return joblib.load(path)
    return None


def load_all_experiment_models() -> dict:
    """Load all saved experiment models. Returns {name: pipeline | None}."""
    models = {}
    for name in EXPERIMENT_FEATURE_SETS:
        models[name] = load_sklearn_model(name)
    return models


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_prevalence(df: pd.DataFrame, prevalence: float) -> np.ndarray:
    return np.full(len(df), prevalence)


def score_rule(df: pd.DataFrame) -> np.ndarray:
    return apply_rule_baseline(df)


def score_sklearn(pipeline, df: pd.DataFrame, num_feats: list[str], cat_feats: list[str]) -> np.ndarray:
    X = df[num_feats + cat_feats].copy()
    return pipeline.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_core_metrics(y_true: np.ndarray, y_score: np.ndarray, split: str) -> dict:
    if y_true.sum() == 0:
        warnings.warn(f"Split '{split}' has 0 positives — skipping core metrics.")
        return {}
    ap = average_precision_score(y_true, y_score)
    roc = roc_auc_score(y_true, y_score)
    bs = brier_score_loss(y_true, y_score)
    ll = log_loss(y_true, np.clip(y_score, 1e-15, 1 - 1e-15))
    return {"ap": ap, "roc_auc": roc, "brier_score": bs, "log_loss": ll}


def compute_topk_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    acv: np.ndarray,
    k: int,
    model: str,
    split: str,
) -> dict:
    """Compute top-K metrics for absolute K (not percentage)."""
    k = min(k, len(y_true))  # cap at available rows
    order = np.argsort(y_score)[::-1]
    topk_idx = order[:k]
    topk_pos = y_true[topk_idx].sum()
    total_pos = y_true.sum()
    precision = topk_pos / k if k > 0 else 0.0
    recall = topk_pos / total_pos if total_pos > 0 else 0.0
    prevalence = total_pos / len(y_true)
    lift = (precision / prevalence) if prevalence > 0 else 0.0
    total_churned_acv = acv[y_true == 1].sum()
    topk_churned_acv = acv[topk_idx][y_true[topk_idx] == 1].sum()
    lost_acv_capture = topk_churned_acv / total_churned_acv if total_churned_acv > 0 else 0.0
    return {
        "model": model,
        "split": split,
        "k": k,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "lift_at_k": lift,
        "lost_acv_capture_at_k": lost_acv_capture,
    }


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------
def plot_reliability_diagram(
    score_store: dict[str, dict[str, np.ndarray]],
    y_true: np.ndarray,
    selected_model: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.2)
    # Show selected model (LR and HGB variants if both exist)
    exp = selected_model.rsplit("_", 1)[0] if "_" in selected_model else selected_model
    model_pairs = [
        (f"{exp}_lr",  "#1f77b4", f"{exp} LR"),
        (f"{exp}_hgb", "#ff7f0e", f"{exp} HGB"),
    ]
    n_bins = 5
    for mname, color, label in model_pairs:
        if mname not in score_store or "model_selection" not in score_store[mname]:
            continue
        y_score = score_store[mname]["model_selection"]
        bin_edges = np.linspace(0, 1, n_bins + 1)
        frac_pos, mean_pred = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (y_score >= lo) & (y_score < hi)
            if mask.sum() == 0:
                continue
            frac_pos.append(y_true[mask].mean())
            mean_pred.append(y_score[mask].mean())
        ax.plot(mean_pred, frac_pos, "o-", color=color,
                label=label, linewidth=1.5, markersize=6)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Reliability Diagram — model_selection ({exp})")
    ax.legend()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "reliability_diagram.png", dpi=150)
    plt.close(fig)
    print(f"  Saved reliability diagram → {FIGURES_DIR / 'reliability_diagram.png'}")


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------
def plot_roc_curves(
    score_store: dict[str, dict[str, np.ndarray]],
    y_true: np.ndarray,
    selected_model: str,
) -> None:
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2)

    model_styles = {
        "prevalence_baseline": ("#aec7e8", ":", "Prevalence Baseline"),
        "rule_baseline": ("#ffbb78", "--", "Rule Baseline"),
    }
    # Add experiment models
    colors_ml = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, mname in enumerate(EXPERIMENT_FEATURE_SETS):
        style = "-" if mname == selected_model else "--"
        lw = 2.0 if mname == selected_model else 1.0
        model_styles[mname] = (colors_ml[i % len(colors_ml)], style,
                                f"{mname}{'  ← selected' if mname == selected_model else ''}")

    for model_name, (color, ls, label) in model_styles.items():
        if model_name not in score_store or "model_selection" not in score_store[model_name]:
            continue
        y_score = score_store[model_name]["model_selection"]
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
        lw = 2.0 if model_name == selected_model else 1.2
        ax.plot(fpr, tpr, color=color, linestyle=ls,
                label=f"{label} (AUC={auc_val:.3f})", linewidth=lw)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — model_selection split")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved ROC curves → {FIGURES_DIR / 'roc_curves.png'}")


# ---------------------------------------------------------------------------
# Ablation — retrain without each feature block (selected experiment only)
# ---------------------------------------------------------------------------
def run_ablation_retrain(
    df_train: pd.DataFrame,
    df_ms: pd.DataFrame,
    y_train: np.ndarray,
    y_ms: np.ndarray,
    baseline_ap: float,
    selected_exp: str,
    num_feats: list[str],
    cat_feats: list[str],
    selected_pipeline=None,
) -> pd.DataFrame:
    """Retrain selected HGB estimator with each feature block removed.

    Clones the actual selected HGB classifier (same hyperparameters) with a new
    preprocessor limited to the remaining features. This gives a genuine HGB
    ablation, not a differently-configured LR proxy.

    selected_pipeline: the fitted sklearn Pipeline for the selected model
        (required). If None or the classifier step cannot be extracted, returns
        an empty DataFrame with a warning.
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    if selected_pipeline is None or not hasattr(selected_pipeline, "named_steps"):
        print("  WARNING: selected_pipeline not provided — ablation skipped.")
        return pd.DataFrame()

    # train.py names the classifier step "clf"; fall back to "classifier" for safety
    base_clf = (selected_pipeline.named_steps.get("clf")
                or selected_pipeline.named_steps.get("classifier"))
    if base_clf is None:
        print(f"  WARNING: no 'clf'/'classifier' step in pipeline "
              f"(steps: {list(selected_pipeline.named_steps.keys())}) — ablation skipped.")
        return pd.DataFrame()

    model_type = "hgb"  # selected model is HGB; preprocessor skips scaler
    blocks = ABLATION_BLOCKS.get(selected_exp, ABLATION_BLOCKS_EXPERIENCE)
    rows = []
    for block_name, block_cols in blocks.items():
        remaining_num = [f for f in num_feats if f not in block_cols]
        remaining_cat = [f for f in cat_feats if f not in block_cols]
        if not remaining_num and not remaining_cat:
            continue

        pre = make_preprocessor(remaining_num, remaining_cat, model_type=model_type)
        clf = Pipeline([
            ("pre", pre),
            ("clf", clone(base_clf)),
        ])

        try:
            clf.fit(df_train[remaining_num + remaining_cat].copy(), y_train)
            y_score = clf.predict_proba(df_ms[remaining_num + remaining_cat].copy())[:, 1]
            ablated_ap = average_precision_score(y_ms, y_score) if y_ms.sum() > 0 else float("nan")
        except Exception as e:
            ablated_ap = float("nan")
            print(f"  WARNING: ablation for {block_name} failed: {e}")

        rows.append({
            "experiment": selected_exp,
            "block": block_name,
            "features_removed": ", ".join(block_cols),
            "method": "retrain_hgb_without_block",
            "baseline_ap": round(baseline_ap, 4),
            "ablated_ap": round(ablated_ap, 4) if not np.isnan(ablated_ap) else float("nan"),
            "ap_drop": round(baseline_ap - ablated_ap, 4) if not np.isnan(ablated_ap) else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Adversarial label shuffle (selected model)
# ---------------------------------------------------------------------------
def run_adversarial_shuffle(
    pipeline,
    df_ms: pd.DataFrame,
    y_true: np.ndarray,
    num_feats: list[str],
    cat_feats: list[str],
    model_name: str,
    n_shuffles: int = 5,
) -> pd.DataFrame:
    """Scores are fixed; labels are shuffled. Tests score vs permuted labels, NOT label leakage."""
    rng = np.random.default_rng(42)
    y_score = pipeline.predict_proba(df_ms[num_feats + cat_feats])[:, 1]
    aps = []
    for _ in range(n_shuffles):
        y_shuffled = rng.permutation(y_true.astype(int))
        if y_shuffled.sum() == 0:
            continue
        aps.append(average_precision_score(y_shuffled, y_score))
    mean_ap = float(np.mean(aps)) if aps else float("nan")
    std_ap  = float(np.std(aps))  if aps else float("nan")
    if mean_ap > 0.15:
        print(f"  WARNING: shuffled AP for {model_name} = {mean_ap:.4f} > 0.15 "
              "— possible data leakage!")
    return pd.DataFrame([{
        "model": model_name,
        "n_shuffles": n_shuffles,
        "mean_shuffled_ap": mean_ap,
        "std_shuffled_ap": std_ap,
        "note": (
            "Tests score sensitivity to label permutation. Does NOT test "
            "training-set label leakage (that would require re-fitting with permuted labels)."
        ),
    }])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("WP7 PANW Churn  –  Part 2 Evaluation")
    print("=" * 70)

    df = load_data()
    sel_record = load_selection_record()

    # Identify selected model from training record
    sel_model_info = sel_record.get("selected_model", {})
    selected_model_key = (
        f"{sel_model_info.get('experiment', 'experience_led')}_"
        f"{sel_model_info.get('model_type', 'lr')}"
    )
    selected_exp = sel_model_info.get("experiment", "experience_led")
    print(f"\n  Selected model from record: {selected_model_key}")
    deployment_policy = sel_record.get("deployment_policy", selected_model_key)
    print(f"  Deployment policy: {deployment_policy}")

    # Load baselines
    prev_info = load_prevalence_baseline()
    prevalence_val = prev_info["value"]
    _ = load_rule_baseline()

    # Load all experiment ML models
    exp_models = load_all_experiment_models()
    n_loaded = sum(1 for v in exp_models.values() if v is not None)
    print(f"  Loaded {n_loaded}/{len(exp_models)} experiment models")

    # Also load refitted model if available
    refit_key = f"{selected_model_key}_refitted"
    refit_info = sel_record.get("refit", {})
    refit_artifact = refit_info.get("artifact", f"{selected_model_key}_refitted.pkl")
    refitted_pipeline = load_sklearn_model(refit_artifact.replace(".pkl", ""))
    if refitted_pipeline is not None:
        exp_models[refit_key] = refitted_pipeline
        EXPERIMENT_FEATURE_SETS[refit_key] = EXPERIMENT_FEATURE_SETS[selected_model_key]
        print(f"  Loaded refitted model: {refit_key}")
    else:
        print(f"  NOTE: refitted model not found ({refit_artifact}) — using selection-time model")

    # Full model registry for evaluation
    all_models = {
        "prevalence_baseline": ("baseline", None),
        "rule_baseline": ("baseline", None),
    }
    for mname, pipe in exp_models.items():
        if pipe is not None:
            all_models[mname] = ("ml", pipe)

    # Storage
    all_metrics: dict[str, dict[str, dict]] = {m: {} for m in all_models}
    topk_rows: list[dict] = []
    score_store: dict[str, dict[str, np.ndarray]] = {m: {} for m in all_models}

    # -----------------------------------------------------------------------
    # Evaluate each model × split
    # -----------------------------------------------------------------------
    for split in EVAL_SPLITS:
        sub, y_true = get_split(df, split)
        acv_arr = sub["acv"].fillna(0.0).values

        if y_true.sum() == 0:
            warnings.warn(f"Split '{split}' has 0 positives — skipping.")
            continue

        for model_name, (mtype, pipeline) in all_models.items():
            # Refitted model was trained on train + model_selection; skip those splits
            if model_name == refit_key and split in ("train", "model_selection"):
                continue

            if mtype == "baseline" and model_name == "prevalence_baseline":
                y_score = score_prevalence(sub, prevalence_val)
            elif mtype == "baseline" and model_name == "rule_baseline":
                y_score = score_rule(sub)
            else:
                feat_key = model_name if model_name in EXPERIMENT_FEATURE_SETS else selected_model_key
                num_f, cat_f = EXPERIMENT_FEATURE_SETS[feat_key]
                y_score = score_sklearn(pipeline, sub, num_f, cat_f)

            score_store[model_name][split] = y_score
            core = compute_core_metrics(y_true, y_score, split)
            all_metrics[model_name][split] = core

            for k in K_VALS:
                row = compute_topk_metrics(y_true, y_score, acv_arr, k, model_name, split)
                topk_rows.append(row)

        # ACV-only baseline
        for k in K_VALS:
            row = compute_topk_metrics(y_true, acv_arr, acv_arr, k, "acv_only_baseline", split)
            topk_rows.append(row)

        # Random baseline (5 seeds, averaged)
        rng_base = np.random.default_rng(0)
        for k in K_VALS:
            prec_list, rec_list, lift_list, acv_list = [], [], [], []
            for _ in range(5):
                rand_scores = rng_base.random(len(y_true))
                m = compute_topk_metrics(y_true, rand_scores, acv_arr, k, "random_baseline", split)
                prec_list.append(m["precision_at_k"])
                rec_list.append(m["recall_at_k"])
                lift_list.append(m["lift_at_k"])
                acv_list.append(m["lost_acv_capture_at_k"])
            topk_rows.append({
                "model": "random_baseline", "split": split, "k": k,
                "precision_at_k": np.mean(prec_list),
                "recall_at_k": np.mean(rec_list),
                "lift_at_k": np.mean(lift_list),
                "lost_acv_capture_at_k": np.mean(acv_list),
            })

    # -----------------------------------------------------------------------
    # Summary table to stdout
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Model':<35} {'Split':<18} {'AP':>8} {'ROC-AUC':>10}")
    print("-" * 70)
    for model_name in all_models:
        for split in EVAL_SPLITS:
            mets = all_metrics[model_name].get(split, {})
            if not mets:
                continue
            ap_val = mets.get("ap", float("nan"))
            roc_val = mets.get("roc_auc", float("nan"))
            marker = " ← SELECTED" if model_name == selected_model_key else ""
            marker += " ← REFITTED" if model_name == refit_key else ""
            flag = "  *** FINAL TEST ***" if split == "final_test" else ""
            print(f"{model_name:<35} {split:<18} {ap_val:>8.4f} {roc_val:>10.4f}{flag}{marker}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Ablation — selected experiment, retrain on train split without each block
    # -----------------------------------------------------------------------
    print("\nRunning retrain-based ablation analysis ...")
    sub_ms, y_ms = get_split(df, "model_selection")
    sel_num, sel_cat = EXPERIMENT_FEATURE_SETS[selected_model_key]
    sel_baseline_ap = all_metrics[selected_model_key].get("model_selection", {}).get("ap", float("nan"))

    df_train_sub = df[(df["split"] == "train") & (df["label_matured"] == True)].copy()
    y_train_abl = df_train_sub[TARGET].astype(int).values

    ablation_df = run_ablation_retrain(
        df_train_sub, sub_ms, y_train_abl, y_ms,
        sel_baseline_ap, selected_exp, sel_num, sel_cat,
        selected_pipeline=exp_models.get(selected_model_key),
    )
    ablation_df.to_csv(OUTPUTS_DIR / "ablation.csv", index=False)
    print(f"  Saved ablation → {OUTPUTS_DIR / 'ablation.csv'}")

    # -----------------------------------------------------------------------
    # Feature importance — permutation on model_selection split
    # Uses the selected experiment model (not refitted) for honest held-out estimates.
    # -----------------------------------------------------------------------
    print("Computing feature importance (permutation, model_selection) ...")
    sel_pipe_fi = exp_models.get(selected_model_key)
    if sel_pipe_fi is not None:
        from sklearn.inspection import permutation_importance
        X_ms_fi = sub_ms[sel_num + sel_cat].copy()
        # Apply same preprocessing as scoring pipeline does internally
        rng_fi  = np.random.default_rng(42)
        perm_result = permutation_importance(
            sel_pipe_fi, X_ms_fi, y_ms,
            scoring="average_precision",
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        fi_rows = []
        for feat_name, imp_mean, imp_std in zip(
            sel_num + sel_cat,
            perm_result.importances_mean,
            perm_result.importances_std,
        ):
            fi_rows.append({
                "feature_name":    feat_name,
                "importance_mean": float(imp_mean),
                "importance_std":  float(imp_std),
                "model":           selected_model_key,
                "eval_split":      "model_selection",
                "method":          "permutation_held_out",
            })
        fi_df = (
            pd.DataFrame(fi_rows)
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        fi_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
        print(f"  Saved feature_importance → {OUTPUTS_DIR / 'feature_importance.csv'}")
        top5 = fi_df.head(5)
        for _, r in top5.iterrows():
            print(f"    {r['feature_name']:<45} {r['importance_mean']:+.5f}")
    else:
        print(f"  Skipping feature importance — selected model pipeline not available")

    # -----------------------------------------------------------------------
    # Adversarial shuffle (selected model, model_selection split)
    # -----------------------------------------------------------------------
    print("Running adversarial label shuffle (model_selection, 5x)...")
    sel_pipe = exp_models.get(selected_model_key)
    if sel_pipe is not None:
        adv_df = run_adversarial_shuffle(
            sel_pipe, sub_ms, y_ms, sel_num, sel_cat, selected_model_key,
        )
        adv_df.to_csv(OUTPUTS_DIR / "adversarial_shuffle.csv", index=False)
        print(f"  Saved adversarial shuffle → {OUTPUTS_DIR / 'adversarial_shuffle.csv'}")
    else:
        print(f"  Skipping adversarial shuffle — selected model pipeline not available")

    # -----------------------------------------------------------------------
    # Reliability diagram + ROC curves (model_selection)
    # -----------------------------------------------------------------------
    print("Plotting reliability diagram and ROC curves...")
    plot_reliability_diagram(score_store, y_ms, selected_model_key)
    plot_roc_curves(score_store, y_ms, selected_model_key)

    # -----------------------------------------------------------------------
    # Save metrics JSON
    # -----------------------------------------------------------------------
    metrics_path = OUTPUTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved metrics → {metrics_path}")

    # -----------------------------------------------------------------------
    # Save top-K CSV
    # -----------------------------------------------------------------------
    topk_df = pd.DataFrame(topk_rows)
    topk_df.to_csv(OUTPUTS_DIR / "eval_topk.csv", index=False)
    print(f"  Saved top-K metrics → {OUTPUTS_DIR / 'eval_topk.csv'}")

    # -----------------------------------------------------------------------
    # FINAL TEST RESULTS (primary reporting)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS  (2026-04-01 — exposed; exploratory)")
    print("=" * 70)
    for model_name in all_models:
        mets = all_metrics[model_name].get("final_test", {})
        if not mets:
            continue
        print(
            f"  {model_name:<35}  AP={mets['ap']:.4f}  ROC-AUC={mets['roc_auc']:.4f}"
            f"  Brier={mets['brier_score']:.4f}  LogLoss={mets['log_loss']:.4f}"
        )
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Unseen-parent sensitivity test (final_test)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("UNSEEN-PARENT SENSITIVITY TEST (final_test)")
    print("=" * 70)

    sensitivity_record: dict = {}
    if "parent_org_id" in df.columns:
        train_parents = set(df.loc[df["split"] == "train", "parent_org_id"].dropna())
        ms_parents    = set(df.loc[df["split"] == "model_selection", "parent_org_id"].dropna())
        # Refit model was trained on train + model_selection → both are "seen" parents.
        fit_parents   = train_parents | ms_parents

        ft_all = df[df["split"] == "final_test"].copy()
        ft_pids = set(ft_all["parent_org_id"].dropna())

        # Two exclusion levels:
        # • clean_train: exclude train parents only  (for pre-refit selection-stage models)
        # • clean_fit  : exclude train ∪ ms parents  (for refitted model — stricter, correct)
        n_overlap_train = len(train_parents & ft_pids)
        n_overlap_fit   = len(fit_parents   & ft_pids)

        ft_clean_train = ft_all[~ft_all["parent_org_id"].isin(train_parents)].copy()
        ft_clean_fit   = ft_all[~ft_all["parent_org_id"].isin(fit_parents)].copy()

        y_ft_clean_train = ft_clean_train[TARGET].values.astype(float)
        y_ft_clean_fit   = ft_clean_fit[TARGET].values.astype(float)

        print(f"  Train parents:                                   {len(train_parents):,}")
        print(f"  Model-selection parents:                         {len(ms_parents):,}")
        print(f"  Fit parents (train ∪ ms):                        {len(fit_parents):,}")
        print(f"  Overlap train ∩ final_test:                      {n_overlap_train}")
        print(f"  Overlap (train ∪ ms) ∩ final_test:               {n_overlap_fit}")
        print(f"  final_test rows (all):                           {len(ft_all)}")
        print(f"  final_test rows (train excluded, n_clean_train): {len(ft_clean_train)}")
        print(f"  final_test rows (train+ms excluded, n_clean_fit):{len(ft_clean_fit)}")
        n_seen_by_refit = n_overlap_fit - n_overlap_train
        print(f"  Additional rows seen by refit (via ms parents):  {n_seen_by_refit}")
        print(f"  NOTE: 'truly unseen' relative to refit = {len(ft_clean_fit)} rows, "
              f"{int(y_ft_clean_fit.sum())} positives "
              f"(prevalence {y_ft_clean_fit.mean():.4f})")

        sensitivity_record = {
            "n_train_parents": len(train_parents),
            "n_ms_parents": len(ms_parents),
            "n_fit_parents": len(fit_parents),
            "n_overlap_train_parents": n_overlap_train,
            "n_overlap_fit_parents": n_overlap_fit,
            "n_final_test_all": len(ft_all),
            "n_final_test_clean_train": len(ft_clean_train),
            "n_final_test_clean_fit": len(ft_clean_fit),
            "n_positives_clean_train": int(y_ft_clean_train.sum()),
            "n_positives_clean_fit": int(y_ft_clean_fit.sum()),
            "prevalence_clean_train": round(float(y_ft_clean_train.mean()), 6),
            "prevalence_clean_fit": round(float(y_ft_clean_fit.mean()), 6),
            "note": (
                "clean_train excludes parents seen in train only (for pre-refit selection models). "
                "clean_fit excludes parents seen in train OR model_selection (correct for the "
                "refitted model, which was fitted on both splits). The 'unseen_parent' sensitivity "
                "claim is only valid for n_final_test_clean_fit rows relative to the refitted model."
            ),
        }

        # --- Evaluate on train-excluded set (selection-stage models, not refitted) ---
        if len(ft_clean_train) > 0 and y_ft_clean_train.sum() > 0:
            print(f"\n  Selection-stage models (train parents excluded, n={len(ft_clean_train)}):")
            print(f"  {'Model':<35} {'AP':>8} {'ROC-AUC':>10} {'Prev':>8}")
            print("  " + "-" * 65)
            sens_metrics_train: dict = {}
            for model_name, (mtype, pipeline) in all_models.items():
                if model_name == refit_key:
                    continue  # refitted model uses stricter exclusion below
                if mtype == "baseline" and model_name == "prevalence_baseline":
                    y_score = score_prevalence(ft_clean_train, prevalence_val)
                elif mtype == "baseline":
                    y_score = score_rule(ft_clean_train)
                else:
                    feat_key = model_name if model_name in EXPERIMENT_FEATURE_SETS else selected_model_key
                    num_f, cat_f = EXPERIMENT_FEATURE_SETS[feat_key]
                    y_score = score_sklearn(pipeline, ft_clean_train, num_f, cat_f)
                core = compute_core_metrics(y_ft_clean_train, y_score, "ft_unseen_train")
                if core:
                    sens_metrics_train[model_name] = core
                    print(f"  {model_name:<35} {core['ap']:>8.4f} {core['roc_auc']:>10.4f} "
                          f"{y_ft_clean_train.mean():>8.4f}")
            sensitivity_record["metrics_unseen_train"] = sens_metrics_train

        # --- Evaluate refitted model on stricter (train+ms) exclusion set ---
        if len(ft_clean_fit) > 0 and y_ft_clean_fit.sum() > 0:
            print(f"\n  Refitted model (train+ms parents excluded, n={len(ft_clean_fit)}):")
            print(f"  {'Model':<35} {'AP':>8} {'ROC-AUC':>10} {'Prev':>8}")
            print("  " + "-" * 65)
            sens_metrics_fit: dict = {}
            for model_name, (mtype, pipeline) in all_models.items():
                if mtype == "baseline" and model_name == "prevalence_baseline":
                    y_score = score_prevalence(ft_clean_fit, prevalence_val)
                elif mtype == "baseline":
                    y_score = score_rule(ft_clean_fit)
                else:
                    feat_key = model_name if model_name in EXPERIMENT_FEATURE_SETS else selected_model_key
                    num_f, cat_f = EXPERIMENT_FEATURE_SETS[feat_key]
                    y_score = score_sklearn(pipeline, ft_clean_fit, num_f, cat_f)
                core = compute_core_metrics(y_ft_clean_fit, y_score, "ft_unseen_fit")
                if core:
                    sens_metrics_fit[model_name] = core
                    print(f"  {model_name:<35} {core['ap']:>8.4f} {core['roc_auc']:>10.4f} "
                          f"{y_ft_clean_fit.mean():>8.4f}")
            sensitivity_record["metrics_unseen_fit"] = sens_metrics_fit
        else:
            print(f"  WARNING: {int(y_ft_clean_fit.sum())} positives in train+ms excluded set — not computable.")
            sensitivity_record["metrics_unseen_fit"] = {}
    else:
        print("  WARNING: parent_org_id column not found — skipping.")

    with open(metrics_path, "r") as f:
        metrics_out = json.load(f)
    metrics_out["unseen_parent_sensitivity"] = sensitivity_record
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  Sensitivity record appended to {metrics_path}")

    # -----------------------------------------------------------------------
    # Account-level policy backtest (final_test cohort)
    # Uses deployment_policy model (refitted if available, else selected)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ACCOUNT-LEVEL POLICY BACKTEST (final_test cohort)")
    print("=" * 70)
    print("NOTE: Exploratory only — final_test cohort was inspected during development.")

    ft_df = df[df["split"] == "final_test"].copy()

    if ft_df[TARGET].sum() == 0:
        print("WARNING: No positives in final_test — skipping policy backtest.")
    else:
        acv_ft = ft_df["acv"].fillna(0.0).values

        # Deployment policy scores
        deploy_pipe = exp_models.get(refit_key) or exp_models.get(selected_model_key)
        dep_num, dep_cat = EXPERIMENT_FEATURE_SETS.get(
            refit_key if refit_key in EXPERIMENT_FEATURE_SETS else selected_model_key,
            (sel_num, sel_cat),
        )
        if deploy_pipe is not None:
            deploy_scores = score_sklearn(deploy_pipe, ft_df, dep_num, dep_cat)
        else:
            deploy_scores = score_rule(ft_df)

        rule_scores = score_rule(ft_df)

        ft_df["deploy_score"] = deploy_scores
        ft_df["rule_score"] = rule_scores
        ft_df["deploy_rwacv"] = deploy_scores * acv_ft
        ft_df["rule_rwacv"] = rule_scores * acv_ft
        ft_df["churned_acv"] = ft_df["acv"].fillna(0.0) * ft_df[TARGET].fillna(0)

        account_ft = ft_df.groupby("company_id").agg(
            max_deploy=("deploy_score", "max"),
            max_rule=("rule_score", "max"),
            max_deploy_rwacv=("deploy_rwacv", "max"),
            max_rule_rwacv=("rule_rwacv", "max"),
            max_acv=("acv", "max"),
            total_acv=("acv", "sum"),
            any_churned=(TARGET, "max"),
            churned_acv=("churned_acv", "sum"),
            parent_org_id=("parent_org_id", "first"),
        ).reset_index()

        n_accounts = len(account_ft)
        total_churned_acv_accounts = account_ft["churned_acv"].sum()

        policies = {
            "deploy_score": "max_deploy",
            "deploy_risk_weighted_acv": "max_deploy_rwacv",
            "rule_score": "max_rule",
            "rule_risk_weighted_acv": "max_rule_rwacv",
            "commercial_baseline": "max_acv",
        }

        # Use K_VALS for capacities; cap at available accounts
        capacities = [k for k in K_VALS if k <= n_accounts]
        if n_accounts not in capacities and n_accounts <= max(K_VALS):
            capacities.append(n_accounts)
        policy_rows = []

        for cap in capacities:
            rand_capture = cap / n_accounts if n_accounts > 0 else 0.0
            for policy_name, score_col in policies.items():
                sel = account_ft.nlargest(cap, score_col)
                acv_capture = (sel["churned_acv"].sum() / total_churned_acv_accounts
                               if total_churned_acv_accounts > 0 else 0.0)
                policy_rows.append({
                    "policy": policy_name,
                    "capacity_n_accounts": cap,
                    "n_churned_accounts_found": int(sel["any_churned"].sum()),
                    "churned_acv_capture_pct": round(acv_capture, 4),
                    "random_baseline_acv_capture_pct": round(rand_capture, 4),
                    "lift_vs_random": round(acv_capture / rand_capture, 3) if rand_capture > 0 else float("nan"),
                })

        policy_df = pd.DataFrame(policy_rows)
        policy_df.to_csv(OUTPUTS_DIR / "account_policy_backtest.csv", index=False)
        print(f"\n  {n_accounts} accounts, {int(account_ft['any_churned'].sum())} with churned contract")
        print(f"  Total churned ACV: ${total_churned_acv_accounts:,.0f}")
        display_caps = [k for k in K_VALS if k <= n_accounts]
        print(f"\n  Policy backtest (top-N accounts, churned ACV captured):")
        header = f"  {'Policy':<28}" + "".join(f"  Cap={c}".rjust(10) for c in display_caps)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for policy_name in policies:
            caps_str = ""
            for cap in display_caps:
                rows_p = policy_df[
                    (policy_df["policy"] == policy_name) & (policy_df["capacity_n_accounts"] == cap)
                ]
                if len(rows_p) > 0:
                    caps_str += f"  {rows_p.iloc[0]['churned_acv_capture_pct']:.1%}".rjust(10)
                else:
                    caps_str += "  N/A".rjust(10)
            print(f"  {policy_name:<28}{caps_str}")
        print(f"\n  Saved → {OUTPUTS_DIR / 'account_policy_backtest.csv'}")

    print("\nDone. All outputs written to part2/outputs/ and part2/figures/")


if __name__ == "__main__":
    main()

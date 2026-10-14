"""
make_splits.py — Assign chronological splits by pred_date and explicit fitting dates.

Reads the per-contract cohort produced by build_features.py and:
  1. Assigns each row to a named split using pred_date (not label_available_at).
  2. Removes dead-zone rows (pred_dates where labels would not yet be available
     at the fitting date, AND the prediction precedes the fitting date).
  3. Separates already-decided scoring contracts.
  4. Runs duplicate contract_id and entity-overlap audits.
  5. Writes split_manifest.json, split_manifest.csv, split_cohort_full.csv,
     boundary_purged.csv, scoring_already_decided.csv, entity_overlap_audit.json.

Split assignment rules
----------------------
  Fitting dates (the date at which a model would have been trained):
    TF1 = 2023-10-01
    TF2 = 2024-10-01
    TF3 = 2025-10-01

  Label maturity lag: label_available_at = pred_date + 180 days.
  Therefore, to train a model fitted at TF, all pred_dates ≤ TF − 180d
  will have labels available.

  Split boundaries (pred_date-based):
    train           pred_date ≤ TF1 − 180d = 2023-04-04
                    (labels fully available by TF1)
    dead_zone_1     pred_date in (2023-04-04, 2023-10-01)
                    (labels not yet mature at TF1; pred_date before TF1)
    model_selection pred_date in [TF1, TF2 − 180d] = [2023-10-01, 2024-04-04)
                    (evaluated after TF1 fitting; labels available by TF2)
    dead_zone_2     pred_date in (2024-04-04, 2024-10-01)
    calibration     pred_date in [TF2, TF3 − 180d] = [2024-10-01, 2025-04-04)
    dead_zone_3     pred_date in (2025-04-04, 2025-10-01)
    final_test      pred_date in [TF3, OBS_END − 180d] = [2025-10-01, 2026-04-04]
                    (all labeled: label_available_at ≤ OBS_END = 2026-10-01)
    scoring         pred_date > 2026-04-04  OR  label_matured == False
                    (no observed outcome yet)

  Chronological integrity: each evaluation split's earliest pred_date is at or after
  the fitting date used to select the model being evaluated, so no evaluation
  observation was seen during training.

Hard failures (sys.exit(1)):
  - Any training row with pred_date > TF1 − 180d.
  - Any model_selection row with pred_date < TF1 or pred_date >= TF2 − 180d.
  - Any calibration row with pred_date < TF2 or pred_date >= TF3 − 180d.
  - Any final_test row with pred_date < TF3 or label_matured == False.
  - Any duplicate contract_id within a single split.

Entity overlap between labeled splits is reported, not failed.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parents[2]
INPUT_CSV  = ROOT / "data" / "analysis" / "renewal_cohort_subscriptions.csv"
OUTPUT_DIR = ROOT / "part2" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("make_splits.py — pred_date-based split assignment + manifest")
print("=" * 70)

# ---------------------------------------------------------------------------
# Fitting dates and boundaries
# ---------------------------------------------------------------------------
OBS_END = pd.Timestamp("2026-10-01")

TF1 = pd.Timestamp("2023-10-01")   # First fitting date
TF2 = pd.Timestamp("2024-10-01")   # Second fitting date
TF3 = pd.Timestamp("2025-10-01")   # Third fitting date

LAG_DAYS = 180   # label_available_at = pred_date + LAG_DAYS

# Evaluation windows: [fitting_date, next_fitting_date − LAG_DAYS)
# (last pred_date for which label matures by the next fitting date)
TRAIN_END    = TF1 - pd.Timedelta(days=LAG_DAYS)     # 2023-04-04
MS_END       = TF2 - pd.Timedelta(days=LAG_DAYS)     # 2024-04-04
CAL_END      = TF3 - pd.Timedelta(days=LAG_DAYS)     # 2025-04-04
FT_END       = OBS_END - pd.Timedelta(days=LAG_DAYS) # 2026-04-04

print(f"\n  Fitting dates and boundaries:")
print(f"    TF1 (first fit)   = {TF1.date()}  |  train cutoff     = pred_date ≤ {TRAIN_END.date()}")
print(f"    TF2 (second fit)  = {TF2.date()}  |  model_sel cutoff = pred_date in [{TF1.date()}, {MS_END.date()})")
print(f"    TF3 (third fit)   = {TF3.date()}  |  calib cutoff     = pred_date in [{TF2.date()}, {CAL_END.date()})")
print(f"    OBS_END           = {OBS_END.date()} |  final_test range = pred_date in [{TF3.date()}, {FT_END.date()}]")

ALL_SPLITS     = ["train", "model_selection", "calibration", "final_test", "scoring"]
LABELED_SPLITS = ["train", "model_selection", "calibration", "final_test"]
DEAD_ZONES     = ["dead_zone_1", "dead_zone_2", "dead_zone_3"]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"\n[1] Loading per-contract cohort from {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
df["pred_date"]          = pd.to_datetime(df["pred_date"],          errors="coerce")
df["label_available_at"] = pd.to_datetime(df["label_available_at"], errors="coerce")
df["label_matured"]      = df["label_matured"].astype(bool)

if "decision_known_at_pred_date" in df.columns:
    df["decision_known_at_pred_date"] = df["decision_known_at_pred_date"].astype(bool)
else:
    df["decision_known_at_pred_date"] = False

print(f"    Loaded {len(df):,} rows  ({df['label_matured'].sum():,} labeled, "
      f"{(~df['label_matured']).sum():,} scoring-eligible)")

# ---------------------------------------------------------------------------
# Step A: Assign splits by pred_date
# ---------------------------------------------------------------------------
print("\n[2] Assigning splits by pred_date ...")

def _assign_split(row) -> str:
    pd_ts = row["pred_date"]
    lm    = row["label_matured"]
    if pd.isna(pd_ts):
        return "unknown"

    # Labeled contracts
    if lm:
        if pd_ts <= TRAIN_END:
            return "train"
        elif pd_ts < TF1:
            return "dead_zone_1"
        elif pd_ts < MS_END:
            return "model_selection"
        elif pd_ts < TF2:
            return "dead_zone_2"
        elif pd_ts < CAL_END:
            return "calibration"
        elif pd_ts < TF3:
            return "dead_zone_3"
        elif pd_ts <= FT_END:
            return "final_test"
        else:
            # pred_date > FT_END but label_matured=True: only possible for late label arrivals
            return "scoring"
    else:
        return "scoring"

df["split"] = df.apply(_assign_split, axis=1)

all_display = ALL_SPLITS + DEAD_ZONES
for s in all_display:
    print(f"    {s:<20} : {(df['split']==s).sum():>7,} rows")

n_unk = (df["split"] == "unknown").sum()
if n_unk > 0:
    print(f"  WARNING: {n_unk} rows could not be assigned a split.", file=sys.stderr)

# ---------------------------------------------------------------------------
# Step B: Separate dead zones
# ---------------------------------------------------------------------------
print("\n[3] Separating dead-zone rows ...")

df_dead = df[df["split"].isin(DEAD_ZONES)].copy()
df_live = df[~df["split"].isin(DEAD_ZONES) & (df["split"] != "unknown")].copy()

for dz in DEAD_ZONES:
    n = (df_dead["split"] == dz).sum()
    if n > 0:
        _sub = df_dead[df_dead["split"] == dz]
        print(f"    {dz}: {n:,} rows  pred_date [{_sub['pred_date'].min().date()}, {_sub['pred_date'].max().date()}]")

print(f"    Total purged to boundary_purged.csv: {len(df_dead):,}")
print(f"    Retained for splits: {len(df_live):,}")

# ---------------------------------------------------------------------------
# Step C: Separate already-decided scoring contracts
# ---------------------------------------------------------------------------
print("\n[4] Separating already-decided scoring contracts ...")

df_labeled     = df_live[df_live["split"] != "scoring"].copy()
df_scoring_all = df_live[df_live["split"] == "scoring"].copy()

df_scoring         = df_scoring_all[~df_scoring_all["decision_known_at_pred_date"]].copy()
df_scoring_decided = df_scoring_all[ df_scoring_all["decision_known_at_pred_date"]].copy()

print(f"    Scoring (actionable)     : {len(df_scoring):>8,}")
print(f"    Scoring (already decided): {len(df_scoring_decided):>8,}")

# ---------------------------------------------------------------------------
# Step D: Integrity gates
# ---------------------------------------------------------------------------
print("\n[5] Integrity gates ...")

errors_found = False

# Train: pred_date <= TRAIN_END
train_rows = df_labeled[df_labeled["split"] == "train"]
viol = train_rows[train_rows["pred_date"] > TRAIN_END]
if len(viol):
    print(f"  GATE FAIL: {len(viol)} train rows have pred_date > {TRAIN_END.date()}", file=sys.stderr)
    errors_found = True
else:
    print(f"  ✓ All train rows: pred_date <= {TRAIN_END.date()}")

# Verify all train labels available by TF1
train_laa_max = train_rows["label_available_at"].max()
if pd.notna(train_laa_max) and train_laa_max > TF1:
    print(f"  WARNING: max train label_available_at={train_laa_max.date()} > TF1={TF1.date()}", file=sys.stderr)
else:
    print(f"  ✓ Max train label_available_at={train_laa_max.date() if pd.notna(train_laa_max) else 'N/A'} "
          f"<= TF1={TF1.date()}")

# Model selection: pred_date in [TF1, MS_END)
ms_rows = df_labeled[df_labeled["split"] == "model_selection"]
ms_viol = ms_rows[~((ms_rows["pred_date"] >= TF1) & (ms_rows["pred_date"] < MS_END))]
if len(ms_viol):
    print(f"  GATE FAIL: {len(ms_viol)} model_selection rows outside [{TF1.date()}, {MS_END.date()})",
          file=sys.stderr)
    errors_found = True
else:
    print(f"  ✓ All model_selection rows: pred_date in [{TF1.date()}, {MS_END.date()})")

# Calibration: pred_date in [TF2, CAL_END)
cal_rows = df_labeled[df_labeled["split"] == "calibration"]
cal_viol = cal_rows[~((cal_rows["pred_date"] >= TF2) & (cal_rows["pred_date"] < CAL_END))]
if len(cal_viol):
    print(f"  GATE FAIL: {len(cal_viol)} calibration rows outside [{TF2.date()}, {CAL_END.date()})",
          file=sys.stderr)
    errors_found = True
else:
    print(f"  ✓ All calibration rows: pred_date in [{TF2.date()}, {CAL_END.date()})")

# Final test: pred_date in [TF3, FT_END], all labeled
ft_rows = df_labeled[df_labeled["split"] == "final_test"]
ft_viol_pd = ft_rows[~((ft_rows["pred_date"] >= TF3) & (ft_rows["pred_date"] <= FT_END))]
ft_viol_lm = ft_rows[~ft_rows["label_matured"]]
if len(ft_viol_pd):
    print(f"  GATE FAIL: {len(ft_viol_pd)} final_test rows have pred_date outside [{TF3.date()}, {FT_END.date()}]",
          file=sys.stderr)
    errors_found = True
elif len(ft_viol_lm):
    print(f"  GATE FAIL: {len(ft_viol_lm)} final_test rows are not labeled (label_matured=False)",
          file=sys.stderr)
    errors_found = True
else:
    print(f"  ✓ All final_test rows: pred_date in [{TF3.date()}, {FT_END.date()}], all labeled")

if errors_found:
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step E: Duplicate contract_id check
# ---------------------------------------------------------------------------
print("\n[6] Duplicate contract_id check ...")

dup_found = False
for s in ALL_SPLITS:
    if s == "scoring":
        check = df_scoring
    else:
        check = df_labeled[df_labeled["split"] == s]
    if len(check) == 0:
        continue
    n_dup = int(check["contract_id"].duplicated(keep=False).sum())
    if n_dup > 0:
        print(f"  GATE FAIL: {n_dup} duplicate contract_ids in '{s}'.", file=sys.stderr)
        dup_found = True
    else:
        print(f"  ✓ {s:<20}: no duplicate contract_ids")

if dup_found:
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step F: Entity-overlap audit
# ---------------------------------------------------------------------------
print("\n[7] Entity-overlap audit ...")

company_splits: dict = {}
for s in LABELED_SPLITS:
    for co in df_labeled.loc[df_labeled["split"] == s, "company_id"].unique():
        company_splits.setdefault(co, set()).add(s)

multi = {co: ss for co, ss in company_splits.items() if len(ss) > 1}
n_total_co = len(company_splits)

parent_overlap: dict = {}
if "parent_org_id" in df_labeled.columns:
    parent_splits: dict = {}
    for s in LABELED_SPLITS:
        for par in df_labeled.loc[df_labeled["split"] == s, "parent_org_id"].dropna().unique():
            parent_splits.setdefault(par, set()).add(s)
    multi_par = {p: ss for p, ss in parent_splits.items() if len(ss) > 1}
    parent_overlap = {
        "n_parents_total":        len(parent_splits),
        "n_parents_multi_split":  len(multi_par),
        "pct_parents_multi_split": round(len(multi_par) / max(len(parent_splits), 1), 4),
    }

train_cos = set(df_labeled.loc[df_labeled["split"] == "train", "company_id"])
ft_cos    = set(df_labeled.loc[df_labeled["split"] == "final_test", "company_id"])
new_ft    = ft_cos - train_cos
overlap_tf = ft_cos & train_cos

overlap_audit = {
    "note": (
        "Company recurrence across splits is legitimate for existing-customer "
        "renewal prediction. Performance on never-seen-in-train companies is a "
        "sensitivity check."
    ),
    "n_companies_total":         n_total_co,
    "n_companies_multi_split":   len(multi),
    "pct_companies_multi_split": round(len(multi) / max(n_total_co, 1), 4),
    "train_final_test_overlap": {
        "n_train_companies":             len(train_cos),
        "n_final_test_companies":        len(ft_cos),
        "n_in_both":                     len(overlap_tf),
        "n_final_test_never_in_train":   len(new_ft),
        "pct_final_test_never_in_train": round(len(new_ft) / max(len(ft_cos), 1), 4),
    },
    "parent_overlap": parent_overlap,
}

print(f"  Companies in labeled cohort              : {n_total_co:,}")
print(f"  Companies in multiple labeled splits     : {len(multi):,} "
      f"({overlap_audit['pct_companies_multi_split']:.1%})")
print(f"  Train–FinalTest company overlap          : {len(overlap_tf):,}")
print(f"  Final-test never seen in train           : {len(new_ft):,} "
      f"({overlap_audit['train_final_test_overlap']['pct_final_test_never_in_train']:.1%})")
if parent_overlap:
    print(f"  Parent orgs in multiple splits           : "
          f"{parent_overlap['n_parents_multi_split']:,} / {parent_overlap['n_parents_total']:,}")

# ---------------------------------------------------------------------------
# Step G: Build manifest
# ---------------------------------------------------------------------------
df_all = pd.concat([df_labeled, df_scoring], ignore_index=True)

manifest = {}
manifest_rows = []

for s in ALL_SPLITS:
    subset = df_all[df_all["split"] == s]
    n_rows = len(subset)
    if s != "scoring":
        lm    = subset[subset["label_matured"].astype(bool)]
        n_pos = int(lm["churned"].fillna(0).astype(int).sum()) if "churned" in lm.columns else 0
        n_neg = int((lm["churned"].fillna(0).astype(int) == 0).sum()) if "churned" in lm.columns else 0
        n_mat = len(lm)
        prev  = round(n_pos / n_mat, 4) if n_mat > 0 else None
    else:
        n_pos, n_neg, n_mat, prev = 0, 0, 0, None

    total_acv = round(float(subset["acv"].sum()), 2) if "acv" in subset.columns else 0.0
    manifest[s] = {
        "n_rows": n_rows, "n_positives": n_pos, "n_negatives": n_neg,
        "n_labeled": n_mat, "prevalence": prev, "total_acv": total_acv,
    }
    manifest_rows.append({
        "split": s, "n_rows": n_rows, "n_positives": n_pos,
        "n_negatives": n_neg, "n_labeled": n_mat, "prevalence": prev,
        "total_acv": total_acv,
    })

# Warnings
for s in LABELED_SPLITS:
    if manifest[s]["n_positives"] == 0:
        print(f"  WARNING: split '{s}' has 0 positives.")
ft_m = manifest["final_test"]
if ft_m["n_positives"] < 20:
    print(f"  WARNING: final_test has only {ft_m['n_positives']} positives.")

print(
    f"\n  FINAL TEST SET SEALED: {ft_m['n_rows']} rows, {ft_m['n_positives']} positives. "
    "Do not evaluate until final model is selected."
)

# ---------------------------------------------------------------------------
# Timing audit (updated with fitting dates)
# ---------------------------------------------------------------------------
timing_audit = {
    "fitting_dates": {
        "TF1": str(TF1.date()),
        "TF2": str(TF2.date()),
        "TF3": str(TF3.date()),
    },
    "label_maturity_lag_days": LAG_DAYS,
    "split_boundaries_by_pred_date": {
        "train":           f"pred_date <= {TRAIN_END.date()}",
        "dead_zone_1":     f"pred_date in ({TRAIN_END.date()}, {TF1.date()})",
        "model_selection": f"pred_date in [{TF1.date()}, {MS_END.date()})",
        "dead_zone_2":     f"pred_date in ({MS_END.date()}, {TF2.date()})",
        "calibration":     f"pred_date in [{TF2.date()}, {CAL_END.date()})",
        "dead_zone_3":     f"pred_date in ({CAL_END.date()}, {TF3.date()})",
        "final_test":      f"pred_date in [{TF3.date()}, {FT_END.date()}]",
        "scoring":         f"pred_date > {FT_END.date()} or label_matured=False",
    },
    "chronological_property": (
        "Each evaluation split's minimum pred_date >= its corresponding fitting date. "
        "No evaluation observation was seen during training."
    ),
    "dead_zones_purged": len(df_dead),
    "n_by_split": {s: manifest[s]["n_rows"] for s in ALL_SPLITS},
}
for s in LABELED_SPLITS:
    if len(df_labeled[df_labeled["split"] == s]) > 0:
        _sub = df_labeled[df_labeled["split"] == s]
        timing_audit["n_by_split"][s + "_pred_date_min"] = str(_sub["pred_date"].min().date())
        timing_audit["n_by_split"][s + "_pred_date_max"] = str(_sub["pred_date"].max().date())

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
print("\n[8] Saving outputs ...")

with open(OUTPUT_DIR / "split_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("    part2/outputs/split_manifest.json")

pd.DataFrame(manifest_rows).to_csv(OUTPUT_DIR / "split_manifest.csv", index=False)
print("    part2/outputs/split_manifest.csv")

with open(OUTPUT_DIR / "entity_overlap_audit.json", "w") as f:
    json.dump(overlap_audit, f, indent=2)
print("    part2/outputs/entity_overlap_audit.json")

with open(OUTPUT_DIR / "timing_audit.json", "w") as f:
    json.dump(timing_audit, f, indent=2)
print("    part2/outputs/timing_audit.json")

# Dead zones
df_dead.to_csv(OUTPUT_DIR / "boundary_purged.csv", index=False)
print(f"    part2/outputs/boundary_purged.csv          — {len(df_dead):,} rows")

# split_cohort_full.csv is the canonical split-annotated cohort.
# renewal_cohort_subscriptions.csv is kept immutable (as produced by build_features.py).
# Downstream scripts (train.py, evaluate.py) read from split_cohort_full.csv.
df_all.to_csv(OUTPUT_DIR / "split_cohort_full.csv", index=False)
print(f"    part2/outputs/split_cohort_full.csv        — {len(df_all):,} rows  ← canonical split output")

if len(df_scoring_decided) > 0:
    df_scoring_decided.to_csv(OUTPUT_DIR / "scoring_already_decided.csv", index=False)
    print(f"    part2/outputs/scoring_already_decided.csv  — {len(df_scoring_decided):,} rows")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n[9] Split summary")
header = f"  {'split':<20} {'n_rows':>7} {'n_labeled':>10} {'n_pos':>7} {'prevalence':>11} {'total_acv':>14}"
print(header)
print("  " + "-" * (len(header) - 2))
for s in ALL_SPLITS:
    m    = manifest[s]
    prev = f"{m['prevalence']:.4f}" if m["prevalence"] is not None else "   N/A"
    print(
        f"  {s:<20} {m['n_rows']:>7,} {m['n_labeled']:>10,} "
        f"{m['n_positives']:>7,} {prev:>11} {m['total_acv']:>14,.2f}"
    )

print(f"\n  Dead zones purged to boundary_purged.csv: {len(df_dead):,} rows")
print(f"  Scoring (already decided, excluded):      {len(df_scoring_decided):,} rows")

print("\n" + "=" * 70)
print("make_splits.py — COMPLETE")
print("=" * 70)

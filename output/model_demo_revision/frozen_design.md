# Frozen design: customer-experience model demonstration

Working folder: panw-case-study-model-demo. The original project is the fallback and must not be modified.

## Purpose and limits
Demonstrate recovery of declared synthetic customer-behavior mechanisms, not estimate Palo Alto Networks behavior. This revision is motivated by exposed prior results. No real-world performance claim is permitted. No label flips, test-based coefficient selection, weakened rules or selective seed reporting.

## Minimal scope
Keep source tables, identifiers, cleaning, observation windows, per-contract expiry-minus-90-day features, outcome maturity, chronological splits, three feature experiments and existing LR/HGB grids. Keep the existing health-rule formula unchanged. All accounts are directly managed. Support products remain outside subscription churn modeling.

## Revised mechanisms (freeze before new outcome generation)
1. A stable, noisy company experience state is a common cause of recorded operational behavior and cancellation. Operating measurements do not depend on sampled cancellation outcomes. Existing product/deployment trajectories remain, but coverage and workflow success more directly reflect the shared experience state.
2. Nonlinear combinations increase risk: deployment failure with unsuccessful use; substantial value deterioration with incomplete deployment; commercial pressure with weak value; unresolved service problems with weak deployment. Coefficients are synthetic design assumptions, not learned PANW effects.
3. Preserve annual segment anchors: SMB 8%, midmarket 5%, enterprise 2.5%, large enterprise 1%. A separate seeded Monte Carlo reference population sets logistic intercepts before outcomes. These anchor average reference annual probabilities, not exact realized cancellation rates. Actual populations, budget adjustments, switching costs, term lengths and survival selection can change observed rates.
4. Preserve stochastic outcomes, geometric annual-to-term conversion and existing segment caps. Healthy customers can cancel and struggling customers can renew. Company size affects the anchor, not an automatic cancellation rule.
5. Preserve the existing health-rule weights. Do not supply hidden probability, latent state, archetype or outcome to model features.

## Evaluation fixed in advance
Main dataset seed: 137, 10,000 purchasing accounts. Original seed 42 is exposed development evidence.
Independent replication seeds: 991 and 2027, each 10,000 purchasing accounts, with all results retained. Freeze the primary selected model before predicting replication accounts. Report these as synthetic-mechanism replications, not new real customer evidence.
Primary operational comparison: precision among the top 30 reviewed purchasing accounts, with policy-specific lead contract. Also report 25, 50 and 100-account capacities, recall and cancelled ACV capture. Report overall AP and ROC-AUC separately. The existing historical AP-based candidate-selection rule stays in place to minimize changes. Do not choose a different winner after reviewing the final cohort.
Keep the existing temporal purges, training-only preprocessing and parent-level paired bootstrap. No success threshold is guaranteed. A favorable aggregate metric alone cannot justify a claim of superior top-30 precision.

## Required deliverables
Changes list; before-and-after synthetic evidence with population counts; all-seed comparison; current assumptions and model card; source hashes proving the original remained intact. Old slides are retained as a fallback but marked stale for the regenerated data. Presentation editing is separate from this data/model revision.

## Pre-evaluation reproducibility correction
Before fitting any models or inspecting new performance, renewal-chain random streams were corrected to include the master seed as well as the lineage ID. Preliminary generation-only runs were stopped and discarded. All three final populations use this correction. The declared experience coefficients and evaluation seeds are unchanged.

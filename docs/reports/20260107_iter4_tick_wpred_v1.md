# Iteration 4: DG tick sampling = w_pred (Full LOSO)

- Date: 2026-01-07
- Git commit: aca80ac
- Primary lever: **DG loss tick sampling** (uniform → `w_pred`-weighted)
- Config: `configs/syncdg_ctm_v1_tick_wpred.yaml`
- Command: `conda run --no-capture-output -n eeg python scripts/run_loso.py --config configs/syncdg_ctm_v1_tick_wpred.yaml --device cuda --amp --exp-name tick_wpred_v1`
- Outputs: `outputs/experiments/20260107_103321_tick_wpred_v1/summary_all.json`

## Aggregate (9 folds)
- acc: **0.3569 ± 0.0962**
- kappa: **0.1425 ± 0.1283**
- macro-f1: **0.3048 ± 0.1285**
- worst target subject: **5** (acc=0.2517)

## Failure-first diagnosis
- **Mean accuracy regressed** vs Iter3 (`valnll_ckpt_v1`: acc_mean=0.3764).
- Subject deltas (acc):
  - improved: target=4 (**+0.1163**), target=6 (**+0.0243**), target=7 (**+0.0069**)
  - regressed: target=8 (**-0.1997**), target=1 (**-0.0799**)
- **Early-exit still not triggering** (`exit_tau_mean=8.0` everywhere).
- Checkpointing by val NLL still weakly correlated with test acc in this run (corr≈+0.15).

## Decision
- Treat as **negative** for the mainline (keep `dg.tick_sampling` option for future ablations).

## Next (one lever)
- Improve DG model selection by adding a **burn-in epoch** before val-NLL checkpointing (`train.min_ckpt_epoch`, e.g. 20).


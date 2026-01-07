# Iteration 5: Val-NLL checkpointing + burn-in (Full LOSO)

- Date: 2026-01-07
- Git commit: aa68b1b
- Primary lever: **model selection** — add **burn-in** before val-NLL checkpointing (`train.min_ckpt_epoch=20`)
- Config: `configs/syncdg_ctm_v1_minckpt20.yaml`
- Command: `conda run --no-capture-output -n eeg python scripts/run_loso.py --config configs/syncdg_ctm_v1_minckpt20.yaml --device cuda --amp --exp-name valnll_minckpt20`
- Outputs: `outputs/experiments/20260107_122724_valnll_minckpt20/summary_all.json`

## Aggregate (9 folds)
- acc: **0.3686 ± 0.1245**
- kappa: **0.1582 ± 0.1660**
- macro-f1: **0.3047 ± 0.1522**
- worst target subject: **4** (acc=0.2500)

## Failure-first diagnosis
- Mean acc slightly **below** Iter3 (`valnll_ckpt_v1`: 0.3764) and **above** Iter4 (`tick_wpred_v1`: 0.3569).
- Burn-in succeeded mechanically: all folds selected `best_epoch >= 22`.
- **Pathology found:** for target=4, source-val metrics drift to **uniform predictions**:
  - val acc stays at **0.25**, while val NLL decreases toward **log(4)=1.38629**; best checkpoint becomes a chance-level model.
  - This makes **min val-NLL** an unsafe single criterion when the val subject is hard and accuracy does not improve.
- Early-exit still not triggering (`exit_tau_mean=8.0` everywhere).

## Decision
- Keep `train.min_ckpt_epoch` (prevents trivial too-early checkpoints), but **val-NLL alone is insufficient** as a DG checkpoint criterion.

## Next (one lever)
- Replace checkpoint criterion with a **lexicographic** rule:
  - maximize source-val **accuracy**, tie-break by **min source-val NLL** (after burn-in).


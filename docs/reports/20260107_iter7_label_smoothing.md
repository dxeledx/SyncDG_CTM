# Iteration 7: Label smoothing (planned)

- Date: 2026-01-07
- Git commit: 0bcd2d0
- Primary lever: **loss regularization** — add **label smoothing** to the main classification loss.
- Base: Iter6 (`ckpt_metric=acc_then_nll`, `min_ckpt_epoch=20`)

## Motivation (failure-first from Iter6)
- Iter6 improved mean acc/kappa but showed **high NLL/ECE** on difficult targets (5/7/2/6), indicating over-confidence and unstable calibration under strict DG.

## Code change
- `scripts/train_loso_fold.py`: `train.label_smoothing` (default 0.0) applied to the aggregated prediction loss.

## Proposed run
```bash
conda run --no-capture-output -n eeg python scripts/run_loso.py --config configs/syncdg_ctm_v1_accthen_nll_minckpt20_ls01.yaml --device cuda --amp --exp-name ls01_accthen_nll_minckpt20
```

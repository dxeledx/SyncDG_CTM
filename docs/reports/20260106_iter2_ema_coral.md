# Iteration: EMA-CORAL stabilization

- Date: 2026-01-06
- Git commit: 43c8431
- Primary lever: **CORAL** (switch from per-batch covariance to **EMA covariance memory**, optional shrinkage)
- Motivation (failure-first):
  - Per-fold source-val acc was often high but test acc low; several folds had high NLL/ECE.
  - Per-batch CORAL covariance estimates are very noisy under small per-domain batch sizes (can spike), likely causing negative transfer / instability.

## Code change
- Implemented `CovarianceEMAMemory` + `coral_ema_loss` in `syncdg_ctm/losses/coral.py`.
- Training now uses EMA-CORAL by default; parameters live in config:
  - `dg.coral.momentum` (zeta), `dg.coral.ridge` (lambda), `dg.coral.shrinkage`.

## L0 sanity
```bash
conda run -n eeg python scripts/run_loso.py --config configs/syncdg_ctm_debug.yaml --subjects 1,2 --device cuda --amp --exp-name ema_coral_smoke
```

## L1 direction check (example)
Baseline reference (previous): `outputs/experiments/l1_cdanE_t07_v01/summary.json`

Run:
```bash
conda run -n eeg python scripts/train_loso_fold.py --config configs/syncdg_ctm_v1.yaml --target-subject 7 --source-val-subject 1 --device cuda --amp --run-dir outputs/experiments/l1_ema_coral_t07_v01
```

Observed:
- target=7 test acc improved **0.2778 → 0.2934**

# Iteration 6: Checkpoint metric = acc_then_nll (planned)

- Date: 2026-01-07
- Git commit: 88a2b9c
- Primary lever: **model selection** — change checkpoint criterion to **lexicographic**:
  - maximize source-val **accuracy**, tie-break by **min source-val NLL** (after `train.min_ckpt_epoch`)

## Motivation (failure-first from Iter5)
- Some folds can minimize val-NLL by drifting toward **uniform predictions** (NLL→log(4)) while val accuracy stays at chance.
- Therefore, **val-NLL alone** is unsafe as a checkpoint criterion when the val subject is hard.

## Code change
- `scripts/train_loso_fold.py`: added `train.ckpt_metric` with options `nll|acc|acc_then_nll` (default `nll`).

## Proposed run
```bash
conda run --no-capture-output -n eeg python scripts/run_loso.py --config configs/syncdg_ctm_v1_accthen_nll_minckpt20.yaml --device cuda --amp --exp-name accthen_nll_minckpt20
```

# Iteration 3: Checkpoint by source-val NLL

- Date: 2026-01-07
- Git commit: a280fdf
- Primary lever: **model selection / early stopping** (DG) — choose checkpoint by **min source-val NLL**
- Protocol note: **source-val = next subject after target (cyclic)** (target=1 → val=2, target=9 → val=1)

## Motivation (failure-first)
- In the previous full LOSO run, max source-val accuracy was **anti-correlated** with test accuracy, while min source-val NLL was **positively correlated** with test accuracy.

## Code change
- `scripts/train_loso_fold.py`: checkpoint + early-stopping now use **val NLL**, while still logging val acc.
- `scripts/run_loso.py`: default source-val selection is now **next subject after target** (cyclic).

## Run command
```bash
conda run -n eeg python scripts/run_loso.py --config configs/syncdg_ctm_v1.yaml --device cuda --amp --exp-name valnll_ckpt_v1
```

## Results (full LOSO)
- Outputs: `outputs/experiments/20260107_081014_valnll_ckpt_v1/summary_all.json`
- acc: **0.3764 ± 0.1418**
- kappa: **0.1685 ± 0.1890**
- macro-f1: **0.3189 ± 0.1731**
- worst target subject: **5** (acc=0.2483)

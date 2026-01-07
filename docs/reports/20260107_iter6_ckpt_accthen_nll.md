# Iteration 6: Checkpoint metric = acc_then_nll (Full LOSO)

- Date: 2026-01-07
- Git commit: 88a2b9c
- Primary lever: **model selection** — change checkpoint criterion to **lexicographic**:
  - maximize source-val **accuracy**, tie-break by **min source-val NLL** (after `train.min_ckpt_epoch`)

## Motivation (failure-first from Iter5)
- Some folds can minimize val-NLL by drifting toward **uniform predictions** (NLL→log(4)) while val accuracy stays at chance.
- Therefore, **val-NLL alone** is unsafe as a checkpoint criterion when the val subject is hard.

## Code change
- `scripts/train_loso_fold.py`: added `train.ckpt_metric` with options `nll|acc|acc_then_nll` (default `nll`).

## Run
```bash
conda run --no-capture-output -n eeg python scripts/run_loso.py --config configs/syncdg_ctm_v1_accthen_nll_minckpt20.yaml --device cuda --amp --exp-name accthen_nll_minckpt20
```

## Results (full LOSO)
- Outputs: `outputs/experiments/20260107_161706_accthen_nll_minckpt20/summary_all.json`
- acc: **0.3860 ± 0.1285**
- kappa: **0.1813 ± 0.1714**
- macro-f1: **0.3398 ± 0.1367**
- worst target subject: **5** (acc=0.2396)

## Failure-first diagnosis
- Mean acc/kappa improved vs Iter3 (`valnll_ckpt_v1`: acc=0.3764, kappa=0.1685), mainly from targets 4/6/9.
- Tail remains weak: targets 5/7 stay near chance (and target 5 slightly below chance).
- Calibration is unstable on hard subjects (some folds show high NLL/ECE).
- Early-exit still not triggering (`exit_tau_mean=8.0` everywhere).

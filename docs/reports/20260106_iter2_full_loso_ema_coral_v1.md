# Iteration 2: EMA-CORAL v1 (Full LOSO evaluation)

- Date: 2026-01-06
- Git commit: 309050a
- Config: `configs/syncdg_ctm_v1.yaml`
- Command: `conda run -n eeg python scripts/run_loso.py --config configs/syncdg_ctm_v1.yaml --device cuda --amp --exp-name ema_coral_v1`
- Outputs: `outputs/experiments/20260106_213852_ema_coral_v1/summary_all.json`

## Aggregate (9 folds)
- acc: **0.3571 ± 0.1058**
- kappa: **0.1427 ± 0.1411**
- macro-f1: **0.3037 ± 0.1271**
- worst target subject: **5** (acc=0.2483)

## Per-target test metrics
| target | acc | kappa | macro-f1 | nll | ece |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4236 | 0.2315 | 0.3958 | 1.224 | 0.072 |
| 2 | 0.2795 | 0.0394 | 0.2290 | 1.475 | 0.154 |
| 3 | 0.5226 | 0.3634 | 0.5046 | 1.151 | 0.069 |
| 4 | 0.3090 | 0.0787 | 0.2425 | 2.096 | 0.468 |
| 5 | 0.2483 | -0.0023 | 0.1262 | 2.015 | 0.454 |
| 6 | 0.2830 | 0.0440 | 0.2334 | 2.000 | 0.420 |
| 7 | 0.2934 | 0.0579 | 0.2889 | 1.828 | 0.369 |
| 8 | 0.5226 | 0.3634 | 0.4764 | 1.288 | 0.137 |
| 9 | 0.3316 | 0.1088 | 0.2365 | 1.545 | 0.307 |

## Failure-first diagnosis (high confidence)
- **Early-exit not working:** `exit_tau_mean=8.0` for all folds (never exits early).
- **Calibration instability:** several targets have very high NLL/ECE (e.g., target 4/5/6).
- **Model selection mismatch in DG:** across folds,
  - corr(best source-val acc, test acc) ≈ **-0.59** (anti-correlated)
  - corr(min source-val NLL, test acc) ≈ **+0.58** (positively correlated)

## Next iteration (one primary lever)
- Switch checkpointing / early stopping criterion from **max source-val accuracy** to **min source-val NLL**.


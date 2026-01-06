# SyncDG-CTM

SyncDG-CTM is a **strict Domain Generalization (DG)** framework for **cross-subject motor imagery EEG** on **BCI Competition IV-2a (MOABB: `BNCI2014_001`)**.

This repo implements the model described in `SyncDG_CTM_design_final.md`:
- CTM-style internal ticks (`T_thought`)
- Online Welford normalization + shrinkage
- Low-rank synchronization projection energy (log-space DG alignment, **no LN**)
- Multi-source DG losses (CDAN(one-hot), Proto-EMA, CORAL, SupCon)
- Reliability-based aggregation + early-exit (teacher distillation for `g`)

## Environment

You said you already prepared a conda env named `eeg` with GPU + deps.

## Quick start (one fold)

```bash
conda run -n eeg python scripts/train_loso_fold.py --help
```

## Protocol

See `docs/PROTOCOL.md`.

## Repo layout

- `syncdg_ctm/`: core library (data, models, losses, training utils)
- `configs/`: yaml configs
- `scripts/`: runnable entrypoints (train/eval)
- `docs/`: protocol, reports, design docs copy

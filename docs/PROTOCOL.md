# Protocol (Strict DG, LOSO)

## Dataset
- **BCI Competition IV-2a** via **MOABB**: `BNCI2014_001`
- 9 subjects, 4 MI classes, 22 channels, 250 Hz

## Main setting (Strict Domain Generalization)
- **Outer split**: LOSO (leave one subject out as *target test*).
- **Training**: only source subjects (no target data, no target statistics).
- **Inner split for early-stopping / model selection**: hold out **one source subject** as *source-val* using a deterministic rule:
  - **source-val = next subject after target (cyclic)**, e.g. target=1 → val=2, target=9 → val=1.
- **Checkpoint criterion**: choose the best checkpoint by **min source-val NLL** (not max source-val accuracy).

## Preprocessing
- window: **2–6s MI** (BCIIV-2a canonical)  
  - MOABB `MotorImagery` uses `tmin/tmax` **relative to the dataset task interval** (`dataset.interval[0]`).  
  - For `BNCI2014_001`, `dataset.interval == [2, 6]` seconds, so **use `tmin=0, tmax=4`** to obtain the **2–6s** window.
- band-pass: 4–38 Hz
- CAR (common average reference)
- channel-wise z-score: **fit on training set only**, applied to val/test

## Commands
- One outer fold:
  - `conda run --no-capture-output -n eeg python scripts/train_loso_fold.py --config configs/syncdg_ctm_v1.yaml --target-subject 1 --device cuda --amp`
- Full LOSO (one experiment folder contains all folds):
  - `conda run --no-capture-output -n eeg python scripts/run_loso.py --config configs/syncdg_ctm_v1.yaml --device cuda --amp`
- Debug run (fast sanity):
  - `conda run --no-capture-output -n eeg python scripts/train_loso_fold.py --config configs/syncdg_ctm_debug.yaml --target-subject 1 --device cuda --amp`

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/syncdg_ctm_v1.yaml")
    parser.add_argument("--outdir", type=str, default="outputs/runs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--source-val-subject", type=int, default=None)
    parser.add_argument("--subjects", type=str, default="1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()

    targets = [int(s.strip()) for s in args.subjects.split(",") if s.strip()]
    for target in targets:
        if args.source_val_subject is None:
            val = next(s for s in range(1, 10) if s != target)
        else:
            val = int(args.source_val_subject)

        cmd = [
            sys.executable,
            "scripts/train_loso_fold.py",
            "--config",
            args.config,
            "--target-subject",
            str(target),
            "--source-val-subject",
            str(val),
            "--outdir",
            args.outdir,
            "--num-workers",
            str(args.num_workers),
        ]
        if args.device is not None:
            cmd += ["--device", args.device]
        if args.amp:
            cmd += ["--amp"]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

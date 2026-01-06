from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import time
import statistics


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/syncdg_ctm_v1.yaml")
    parser.add_argument("--outdir", type=str, default="outputs/experiments")
    parser.add_argument("--exp-name", type=str, default=None, help="Optional experiment name (folder will include timestamp).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--source-val-subject", type=int, default=None)
    parser.add_argument("--subjects", type=str, default="1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()

    targets = [int(s.strip()) for s in args.subjects.split(",") if s.strip()]
    cfg_stem = Path(args.config).stem
    exp_name = args.exp_name or cfg_stem
    exp_dir = Path(args.outdir) / f"{_now_tag()}_{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot config + command.
    (exp_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    try:
        (exp_dir / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    print(f"[exp] dir={exp_dir} folds={len(targets)}", flush=True)

    fold_summaries = []
    for target in targets:
        if args.source_val_subject is None:
            val = next(s for s in range(1, 10) if s != target)
        else:
            val = int(args.source_val_subject)

        fold_dir = exp_dir / f"fold_t{target:02d}_v{val:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-u",
            "scripts/train_loso_fold.py",
            "--config",
            args.config,
            "--target-subject",
            str(target),
            "--source-val-subject",
            str(val),
            "--run-dir",
            str(fold_dir),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.device is not None:
            cmd += ["--device", args.device]
        if args.amp:
            cmd += ["--amp"]

        print(f"[exp] running fold target={target} val={val}: {' '.join(cmd)}", flush=True)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        subprocess.run(cmd, check=True, env=env)

        summary_path = fold_dir / "summary.json"
        if summary_path.exists():
            fold_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    if fold_summaries:
        fold_summaries = sorted(fold_summaries, key=lambda x: int(x["target_subject"]))
        accs = [float(s["test"]["acc"]) for s in fold_summaries]
        kappas = [float(s["test"]["kappa"]) for s in fold_summaries]
        f1s = [float(s["test"]["macro_f1"]) for s in fold_summaries]

        agg = {
            "acc_mean": float(statistics.mean(accs)),
            "acc_std": float(statistics.stdev(accs)) if len(accs) > 1 else 0.0,
            "kappa_mean": float(statistics.mean(kappas)),
            "kappa_std": float(statistics.stdev(kappas)) if len(kappas) > 1 else 0.0,
            "macro_f1_mean": float(statistics.mean(f1s)),
            "macro_f1_std": float(statistics.stdev(f1s)) if len(f1s) > 1 else 0.0,
            "worst_target_subject": int(min(fold_summaries, key=lambda s: float(s["test"]["acc"]))["target_subject"]),
            "worst_acc": float(min(accs)),
        }
        out = {"exp_dir": str(exp_dir), "config": args.config, "folds": fold_summaries, "aggregate": agg}
        (exp_dir / "summary_all.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[exp] aggregate:", json.dumps(agg, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

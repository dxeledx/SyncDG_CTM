from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from syncdg_ctm.data.dataset import EEGTrialDataset
from syncdg_ctm.data.moabb import MoabbBCIIV2aConfig, get_loso_split_indices, load_bnci2014_001
from syncdg_ctm.data.samplers import BalancedDomainClassBatchSampler, seed_worker
from syncdg_ctm.data.transforms import ChannelZScore, compute_channel_stats
from syncdg_ctm.losses import (
    CovarianceEMAMemory,
    PrototypeEMAMemory,
    cdan_onehot_loss,
    coral_ema_loss,
    orth_loss,
    proto_alignment_loss,
    supervised_contrastive_loss,
    tick_diversity_loss,
)
from syncdg_ctm.models import (
    DomainDiscriminator,
    DomainDiscriminatorConfig,
    ProjectionHead,
    ProjectionHeadConfig,
    SyncDGCTM,
    SyncDGCTMConfig,
)
from syncdg_ctm.models.ctm_core import CTMCoreConfig
from syncdg_ctm.models.reliability import ReliabilityConfig
from syncdg_ctm.models.sync_head import SyncHeadConfig
from syncdg_ctm.models.tokenizer import TokenizerConfig
from syncdg_ctm.training.metrics import compute_metrics
from syncdg_ctm.training.schedule import dann_ramp
from syncdg_ctm.utils.config import dump_yaml, load_yaml
from syncdg_ctm.utils.seed import seed_everything


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _device_from_arg(device: str | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@torch.no_grad()
def evaluate(
    model: SyncDGCTM,
    loader: DataLoader,
    device: torch.device,
    *,
    early_exit: bool = True,
    exit_q: float = 0.7,
    exit_margin: float = 0.2,
    exit_kl: float = 0.05,
) -> dict[str, float]:
    model.eval()
    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []

    tau_exit_all: list[np.ndarray] = []
    y_prob_exit_all: list[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        out = model(x)
        p_ticks = out["p_ticks"]
        w = out["w_pred"]
        p_agg = (w.unsqueeze(-1) * p_ticks).sum(dim=1)

        y_true_all.append(y.cpu().numpy())
        y_prob_all.append(p_agg.cpu().numpy())

        if early_exit:
            q = out["q_ticks"]
            margin = out["u_ticks"][..., 1]
            kl = out["u_ticks"][..., 2]
            b, t = q.shape
            tau_exit = torch.full((b,), t - 1, device=device, dtype=torch.long)
            for tau in range(t):
                ok = (q[:, tau] > exit_q) & (margin[:, tau] > exit_margin) & (kl[:, tau] < exit_kl)
                tau_exit = torch.where((tau_exit == (t - 1)) & ok, torch.full_like(tau_exit, tau), tau_exit)

            p_exit = p_ticks[torch.arange(b, device=device), tau_exit]
            tau_exit_all.append((tau_exit + 1).cpu().numpy())  # 1-indexed
            y_prob_exit_all.append(p_exit.cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    metrics = compute_metrics(y_true, y_prob)

    if early_exit and tau_exit_all:
        tau_exit = np.concatenate(tau_exit_all, axis=0)
        y_prob_exit = np.concatenate(y_prob_exit_all, axis=0)
        metrics_exit = compute_metrics(y_true, y_prob_exit)
        metrics["exit_acc"] = metrics_exit["acc"]
        metrics["exit_kappa"] = metrics_exit["kappa"]
        metrics["exit_tau_mean"] = float(tau_exit.mean())
        metrics["exit_tau_p95"] = float(np.percentile(tau_exit, 95))

    return metrics


def build_model(cfg: dict) -> tuple[SyncDGCTM, dict]:
    m = cfg["model"]
    tok = TokenizerConfig(
        n_channels=22,
        d_token=int(m["tokenizer"]["d_token"]),
        n_tokens=int(m["tokenizer"]["n_tokens"]),
        temporal_kernels=tuple(int(k) for k in m["tokenizer"]["temporal_kernels"]),
        dropout=float(m["tokenizer"].get("dropout", 0.1)),
    )
    ctm = CTMCoreConfig(
        d_model=int(m["ctm"]["d_model"]),
        d_token=int(m["tokenizer"]["d_token"]),
        n_ticks=int(m["ctm"]["n_ticks"]),
        memory_len=int(m["ctm"]["memory_len"]),
        attn_heads=int(m["ctm"]["attn_heads"]),
        synapse_hidden=int(m["ctm"]["synapse_hidden"]),
        nlm_hidden=int(m["ctm"]["nlm_hidden"]),
        neuron_embed_dim=int(m["ctm"]["neuron_embed_dim"]),
        dropout=float(m["ctm"].get("dropout", 0.1)),
    )
    head = SyncHeadConfig(
        d_model=int(m["ctm"]["d_model"]),
        d_out=int(m["sync_head"]["d_out"]),
        num_classes=int(m["num_classes"]),
        gamma=float(m["sync_head"]["gamma"]),
        welford_shrinkage=float(m["sync_head"]["welford_shrinkage"]),
        eps=float(m["sync_head"]["eps"]),
        eps_h=float(m["sync_head"]["eps_h"]),
    )
    rel = ReliabilityConfig(
        in_dim=5,
        hidden=int(cfg.get("reliability", {}).get("hidden", 32)),
        dropout=float(cfg.get("reliability", {}).get("dropout", 0.1)),
        tau_g=float(cfg["early_exit"]["tau_g"]),
    )

    model_cfg = SyncDGCTMConfig(num_classes=int(m["num_classes"]), tokenizer=tok, ctm=ctm, sync_head=head, reliability=rel)
    model = SyncDGCTM(model_cfg)
    resolved = {
        "model": {
            "num_classes": int(m["num_classes"]),
            "tokenizer": asdict(tok),
            "ctm": asdict(ctm),
            "sync_head": asdict(head),
            "reliability": asdict(rel),
        }
    }
    return model, resolved


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/syncdg_ctm_v1.yaml")
    parser.add_argument("--target-subject", type=int, required=True, help="Held-out test subject id (1..9)")
    parser.add_argument("--source-val-subject", type=int, default=None, help="One source subject used as val (LOSO-in-source)")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="If set, write all outputs into this directory (recommended when running full LOSO).",
    )
    parser.add_argument("--outdir", type=str, default="outputs/runs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    # Validation subject: the *next* subject after target (cyclic) unless specified.
    # For BCIIV-2a (BNCI2014_001), subjects are 1..9.
    source_val_subject = (
        int(args.source_val_subject) if args.source_val_subject is not None else (1 if args.target_subject == 9 else args.target_subject + 1)
    )

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("train", {}).get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)

    device = _device_from_arg(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(args.outdir) / f"{_now_tag()}_t{args.target_subject}_v{source_val_subject:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, run_dir / "config.yaml")
    print(f"[fold] target={args.target_subject} source_val={source_val_subject} device={device} run_dir={run_dir}", flush=True)

    # Data
    data_cfg = cfg["data"]
    moabb_cfg = MoabbBCIIV2aConfig(
        tmin=float(data_cfg["tmin"]),
        tmax=float(data_cfg["tmax"]),
        fmin=float(data_cfg["fmin"]),
        fmax=float(data_cfg["fmax"]),
        resample=None if data_cfg.get("resample", None) in (None, "null") else int(data_cfg["resample"]),
        apply_car=bool(data_cfg.get("apply_car", True)),
        cache_dir=str(data_cfg.get("cache_dir", "outputs/cache")),
    )
    x_all, y_all, subj_all = load_bnci2014_001(moabb_cfg)
    split = get_loso_split_indices(subj_all, target_subject=args.target_subject, source_val_subject=source_val_subject)

    x_train = x_all[split["train"]]
    y_train = y_all[split["train"]]
    subj_train = subj_all[split["train"]]

    x_val = x_all[split["val"]]
    y_val = y_all[split["val"]]

    x_test = x_all[split["test"]]
    y_test = y_all[split["test"]]

    # Map training subjects to contiguous domain ids.
    train_subjects = sorted(set(subj_train.tolist()))
    subject_to_domain = {s: i for i, s in enumerate(train_subjects)}
    d_train = np.asarray([subject_to_domain[int(s)] for s in subj_train], dtype=np.int64)
    n_domains = len(train_subjects)

    # Z-score: fit on TRAIN only.
    mean, std = compute_channel_stats(x_train)
    zscore = ChannelZScore(mean=mean, std=std)

    train_ds = EEGTrialDataset(x_train, y_train, d_train, transform=zscore if data_cfg.get("zscore", True) else None)
    val_ds = EEGTrialDataset(x_val, y_val, np.full_like(y_val, -1), transform=zscore if data_cfg.get("zscore", True) else None)
    test_ds = EEGTrialDataset(x_test, y_test, np.full_like(y_test, -1), transform=zscore if data_cfg.get("zscore", True) else None)

    batch_cfg = cfg["train"]["batch"]
    sampler = BalancedDomainClassBatchSampler(
        labels=y_train,
        domains=d_train,
        n_classes=int(cfg["model"]["num_classes"]),
        domains_per_batch=int(batch_cfg["domains_per_batch"]),
        samples_per_class=int(batch_cfg["samples_per_class"]),
        steps_per_epoch=cfg["train"].get("steps_per_epoch", None),
        seed=seed,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model, resolved_model_cfg = build_model(cfg)
    model = model.to(device)

    d_out = int(cfg["model"]["sync_head"]["d_out"])
    num_classes = int(cfg["model"]["num_classes"])

    disc = DomainDiscriminator(
        DomainDiscriminatorConfig(in_dim=d_out * num_classes, n_domains=n_domains, hidden=256, dropout=0.2)
    ).to(device)
    proj_head = ProjectionHead(ProjectionHeadConfig(in_dim=d_out, hidden=256, out_dim=128)).to(device)
    proto_mem = PrototypeEMAMemory(n_domains=n_domains, n_classes=num_classes, feat_dim=d_out, momentum=0.1).to(
        device=device, dtype=torch.float32
    )
    coral_cfg = cfg.get("dg", {}).get("coral", {}) or {}
    cov_mem = CovarianceEMAMemory(
        n_domains=n_domains,
        feat_dim=d_out,
        momentum=float(coral_cfg.get("momentum", 0.01)),
        ridge=float(coral_cfg.get("ridge", 1e-3)),
        shrinkage=float(coral_cfg.get("shrinkage", 0.0)),
    ).to(device=device, dtype=torch.float32)

    lr = float(cfg["train"]["optimizer"]["lr"])
    wd = float(cfg["train"]["optimizer"]["weight_decay"])

    main_params = list(model.tokenizer.parameters()) + list(model.ctm.parameters()) + list(model.head.parameters())
    main_params += list(disc.parameters()) + list(proj_head.parameters())

    opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=wd)
    opt_g = torch.optim.AdamW(model.g.parameters(), lr=2e-3, weight_decay=wd)

    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    epochs = int(cfg["train"]["epochs"])
    warmup_epochs = int(cfg["dg"]["warmup_epochs"])
    patience = int(cfg["train"]["early_stop_patience"])
    min_ckpt_epoch = int(cfg.get("train", {}).get("min_ckpt_epoch", 1))
    ckpt_metric = str(cfg.get("train", {}).get("ckpt_metric", "nll")).lower()
    if ckpt_metric in ("nll", "val_nll", "loss"):
        ckpt_metric = "nll"
    elif ckpt_metric in ("acc", "val_acc", "accuracy"):
        ckpt_metric = "acc"
    elif ckpt_metric in ("acc_then_nll", "acc_then_loss", "acc+nll"):
        ckpt_metric = "acc_then_nll"
    else:
        raise ValueError(f"Unknown train.ckpt_metric={ckpt_metric!r} (expected: nll|acc|acc_then_nll)")

    lmb = cfg["dg"]["lambda"]
    lambda_div = float(lmb["div"])
    lambda_orth = float(lmb["orth"])
    lambda_adv_max = float(lmb["adv"])
    lambda_proto_max = float(lmb["proto"])
    lambda_coral_max = float(lmb["coral"])
    lambda_supcon_max = float(lmb["supcon"])
    lambda_teach = float(lmb["teach"])

    tau_L = float(cfg["early_exit"]["tau_L"])

    best_val_nll = math.inf
    best_val_acc = -math.inf
    max_val_acc = -math.inf
    max_val_acc_epoch = 0
    best_path = run_dir / "best.pt"
    bad_epochs = 0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        disc.train()
        proj_head.train()
        sampler.set_epoch(epoch)

        if epoch < warmup_epochs:
            ramp = 0.0
        else:
            denom = max(epochs - warmup_epochs, 1)
            ramp = dann_ramp((epoch - warmup_epochs + 1) / denom)

        lambda_adv = lambda_adv_max * ramp
        lambda_proto = lambda_proto_max * ramp
        lambda_coral = lambda_coral_max * ramp
        lambda_supcon = lambda_supcon_max * ramp

        use_tqdm = os.isatty(1)
        log_every = int(cfg.get("train", {}).get("log_every", 50))

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch+1}/{epochs}",
            ncols=120,
            dynamic_ncols=True,
            leave=False,
            disable=not use_tqdm,
        )
        for step, batch in enumerate(pbar, start=1):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            d = batch["domain"].to(device)

            opt_main.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(x)
                z_ticks = out["z_ticks"]
                x_ticks = out["x_ticks"]
                logits_ticks = out["logits_ticks"]
                p_ticks = out["p_ticks"]

                w_pred = out["w_pred"].detach()  # no grad to g from cls
                p_agg = (w_pred.unsqueeze(-1) * p_ticks).sum(dim=1)
                log_p_agg = torch.log(p_agg + 1e-8)
                label_smoothing = float(cfg.get("train", {}).get("label_smoothing", 0.0))
                if label_smoothing > 0.0:
                    # Treat log-probabilities as logits (log_softmax(log_p)=log_p because sum(p)=1).
                    loss_cls = F.cross_entropy(log_p_agg, y, label_smoothing=label_smoothing)
                else:
                    loss_cls = F.nll_loss(log_p_agg, y)

                t = x_ticks.shape[1]
                tick_sampling = str(cfg.get("dg", {}).get("tick_sampling", "uniform")).lower()
                if tick_sampling == "uniform":
                    tau_star = torch.randint(0, t, (x.shape[0],), device=device)
                elif tick_sampling in ("w_pred", "wpred"):
                    # Sample the tick using the (detached) reliability weights.
                    w = w_pred.clamp_min(1e-8)
                    tau_star = torch.multinomial(w, num_samples=1, replacement=True).squeeze(1)
                elif tick_sampling in ("argmax", "max"):
                    tau_star = w_pred.argmax(dim=1)
                elif tick_sampling == "last":
                    tau_star = torch.full((x.shape[0],), t - 1, device=device, dtype=torch.long)
                else:
                    raise ValueError(f"Unknown dg.tick_sampling={tick_sampling!r} (expected: uniform|w_pred|argmax|last)")
                x_star = x_ticks[torch.arange(x.shape[0], device=device), tau_star]

                loss_div = tick_diversity_loss(z_ticks)
                loss_orth = orth_loss(model.head.proj)

                if ramp > 0.0:
                    # CDAN-E style entropy conditioning (stop-grad) to down-weight uncertain samples.
                    p_star = p_ticks[torch.arange(x.shape[0], device=device), tau_star]
                    ent = -(p_star * torch.log(p_star + 1e-8)).sum(dim=1)
                    ent_max = math.log(num_classes)
                    w_ent = (1.0 - ent / ent_max).clamp(0.0, 1.0).detach()

                    loss_adv = cdan_onehot_loss(
                        x_star, y, d, disc, n_classes=num_classes, grl_lambda=1.0, sample_weight=w_ent
                    )
                    loss_proto, _ = proto_alignment_loss(x_star, y, d, memory=proto_mem)
                    loss_coral = coral_ema_loss(x_star, d, memory=cov_mem)
                    feats_supcon = proj_head(x_star)
                    loss_supcon = supervised_contrastive_loss(feats_supcon, y, temperature=0.1)
                else:
                    loss_adv = x_star.new_tensor(0.0)
                    loss_proto = x_star.new_tensor(0.0)
                    loss_coral = x_star.new_tensor(0.0)
                    loss_supcon = x_star.new_tensor(0.0)

                loss_main = (
                    loss_cls
                    + lambda_div * loss_div
                    + lambda_orth * loss_orth
                    + lambda_adv * loss_adv
                    + lambda_proto * loss_proto
                    + lambda_coral * loss_coral
                    + lambda_supcon * loss_supcon
                )

            scaler.scale(loss_main).backward()
            scaler.unscale_(opt_main)
            torch.nn.utils.clip_grad_norm_(main_params, float(cfg["train"]["grad_clip"]))
            scaler.step(opt_main)
            scaler.update()

            # g update (teacher distillation only)
            opt_g.zero_grad(set_to_none=True)
            with torch.no_grad():
                # per-tick CE for teacher weights
                ce = []
                for tt in range(logits_ticks.shape[1]):
                    ce.append(F.cross_entropy(logits_ticks[:, tt, :], y, reduction="none"))
                ce = torch.stack(ce, dim=1)  # (B,T)
                w_teach = F.softmax(-ce / tau_L, dim=1)

            w_pred_g = out["w_pred"]  # with grad w.r.t g
            w_teach = w_teach.clamp_min(1e-8)
            w_pred_g = w_pred_g.clamp_min(1e-8)
            loss_teach = (w_teach * (torch.log(w_teach) - torch.log(w_pred_g))).sum(dim=1).mean()
            (lambda_teach * loss_teach).backward()
            opt_g.step()

            metrics_str = (
                f"cls={float(loss_cls.detach().cpu()):.4f} "
                f"adv={float(loss_adv.detach().cpu()):.4f} "
                f"proto={float(loss_proto.detach().cpu()):.4f} "
                f"coral={float(loss_coral.detach().cpu()):.4f} "
                f"supcon={float(loss_supcon.detach().cpu()):.4f} "
                f"div={float(loss_div.detach().cpu()):.4f} "
                f"ramp={float(ramp):.3f}"
            )

            if use_tqdm:
                pbar.set_postfix(
                    {
                        "cls": float(loss_cls.detach().cpu()),
                        "adv": float(loss_adv.detach().cpu()),
                        "proto": float(loss_proto.detach().cpu()),
                        "coral": float(loss_coral.detach().cpu()),
                        "supcon": float(loss_supcon.detach().cpu()),
                        "div": float(loss_div.detach().cpu()),
                        "ramp": float(ramp),
                    }
                )
            elif step == 1 or step % log_every == 0:
                print(f"[epoch {epoch+1} step {step}/{len(train_loader)}] {metrics_str}", flush=True)

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            early_exit=True,
            exit_q=float(cfg.get("exit", {}).get("q", 0.7)),
            exit_margin=float(cfg.get("exit", {}).get("margin", 0.2)),
            exit_kl=float(cfg.get("exit", {}).get("kl", 0.05)),
        )
        with (run_dir / "val_metrics.json").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch + 1, **val_metrics}, ensure_ascii=False) + "\n")
        print(
            f"[epoch {epoch+1}] val_acc={val_metrics['acc']:.4f} val_nll={val_metrics['nll']:.4f} val_kappa={val_metrics['kappa']:.4f}",
            flush=True,
        )

        val_acc = float(val_metrics["acc"])
        val_nll = float(val_metrics["nll"])

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            max_val_acc_epoch = epoch + 1

        # Burn-in: avoid selecting trivial early checkpoints (e.g., near-uniform predictions minimizing NLL).
        if epoch + 1 < min_ckpt_epoch:
            bad_epochs = 0
            continue

        # DG model selection (configurable):
        # - nll: minimize source-val NLL
        # - acc: maximize source-val accuracy
        # - acc_then_nll: maximize acc, tie-break by min NLL
        improved = False
        if ckpt_metric == "nll":
            improved = val_nll < best_val_nll
        elif ckpt_metric == "acc":
            improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_nll < best_val_nll)
        elif ckpt_metric == "acc_then_nll":
            improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_nll < best_val_nll)

        if improved:
            best_val_nll = val_nll
            best_val_acc = val_acc
            bad_epochs = 0
            best_epoch = epoch + 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "disc": disc.state_dict(),
                    "proj_head": proj_head.state_dict(),
                    "seed": seed,
                    "target_subject": int(args.target_subject),
                    "source_val_subject": int(source_val_subject),
                    "train_subjects": train_subjects,
                    "resolved_model_cfg": resolved_model_cfg,
                    "moabb_cfg": asdict(moabb_cfg),
                    "best_epoch": int(best_epoch),
                    "best_val_nll": float(best_val_nll),
                    "best_val_acc": float(best_val_acc),
                    "ckpt_metric": ckpt_metric,
                    "coral_mem": {
                        "cov": cov_mem.cov.detach().cpu(),
                        "initialized": cov_mem.initialized.detach().cpu(),
                        "momentum": float(cov_mem.momentum),
                        "ridge": float(cov_mem.ridge),
                        "shrinkage": float(cov_mem.shrinkage),
                    },
                },
                best_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if not best_path.exists():
        raise RuntimeError(
            f"No checkpoint was saved (best_path={best_path}). "
            f"Check that train.min_ckpt_epoch <= train.epochs (min_ckpt_epoch={min_ckpt_epoch}, epochs={epochs})."
        )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        early_exit=True,
        exit_q=float(cfg.get("exit", {}).get("q", 0.7)),
        exit_margin=float(cfg.get("exit", {}).get("margin", 0.2)),
        exit_kl=float(cfg.get("exit", {}).get("kl", 0.05)),
    )
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "target_subject": int(args.target_subject),
        "source_val_subject": int(source_val_subject),
        "train_subjects": train_subjects,
        "best_val_acc": float(best_val_acc),
        "best_val_nll": float(best_val_nll),
        "best_epoch": int(best_epoch),
        "ckpt_metric": ckpt_metric,
        "max_val_acc": float(max_val_acc),
        "max_val_acc_epoch": int(max_val_acc_epoch),
        "test": test_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()

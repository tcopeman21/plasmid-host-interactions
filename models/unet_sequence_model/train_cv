from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from models.unet_sequence_model.data import ArrayDataset, AugmentConfig, augment_batch, load_pkl_sequences, rc_onehot
from models.unet_sequence_model.model import GCNUNet


@dataclass(frozen=True)
class TrainConfig:
    pkl_path: str
    out_dir: str

    k_folds: int = 5
    seed: int = 42

    epochs: int = 25
    batch_train: int = 512
    batch_eval: int = 1024
    lr: float = 1e-3
    weight_decay: float = 3e-4

    lam_rank: float = 0.40
    lam_rc: float = 0.30
    grad_clip: float = 0.5

    es_patience: int = 8
    num_workers: int = 2

    use_bf16_amp: bool = True

    aug_shift_max: int = 3
    aug_mut_rate: float = 0.02


@torch.no_grad()
def eval_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds, truth = [], []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_amp and device.type == "cuda")
        else torch.cuda.amp.autocast(enabled=False)
    )

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()
        with amp_ctx:
            risk = model(xb)
            yhat = risk.mean(dim=1)
        preds.append(yhat.detach().cpu().float().numpy())
        truth.append(yb.detach().cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(truth)
    r = pearsonr(y_pred, y_true)[0]
    return float(r), y_pred, y_true


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    aug_cfg: AugmentConfig,
    cfg: TrainConfig,
) -> float:
    model.train()
    huber = torch.nn.HuberLoss()
    total, n = 0.0, 0

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_amp and device.type == "cuda")
        else torch.cuda.amp.autocast(enabled=False)
    )

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()

        xb_aug = augment_batch(xb, aug_cfg)

        opt.zero_grad(set_to_none=True)

        with amp_ctx:
            risk = model(xb_aug)
            yhat = risk.mean(dim=1)
            loss_sup = huber(yhat, yb)

            risk_rc = model(rc_onehot(xb_aug))
            yhat_rc = risk_rc.mean(dim=1)
            loss_rc = F.mse_loss(yhat, yhat_rc)

            qlo = torch.quantile(yb, 0.35)
            qhi = torch.quantile(yb, 0.65)
            lo_idx = torch.where(yb <= qlo)[0]
            hi_idx = torch.where(yb >= qhi)[0]

            if len(lo_idx) > 1 and len(hi_idx) > 1:
                k = min(len(lo_idx), len(hi_idx), 64)
                ii = hi_idx[torch.randint(len(hi_idx), (k,), device=yb.device)]
                jj = lo_idx[torch.randint(len(lo_idx), (k,), device=yb.device)]
                loss_rank = F.softplus(-(yhat[ii] - yhat[jj])).mean()
            else:
                loss_rank = torch.tensor(0.0, device=device)

            loss = loss_sup + cfg.lam_rank * loss_rank + cfg.lam_rc * loss_rc

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        total += float(loss_sup.item()) * len(yb)
        n += len(yb)

    return total / max(1, n)


def run(cfg: TrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "train_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.use_bf16_amp)

    X, y = load_pkl_sequences(cfg.pkl_path)
    ds = ArrayDataset(X, y)

    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    aug_cfg = AugmentConfig(shift_max=cfg.aug_shift_max, mut_rate=cfg.aug_mut_rate)

    fold_summary = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(range(len(ds))), start=1):
        print(f"\n=== Fold {fold}/{cfg.k_folds} ===")
        tr_idx = np.array(tr_idx)
        te_idx = np.array(te_idx)

        v_n = int(len(tr_idx) * 0.10)
        val_idx = tr_idx[:v_n]
        train_idx = tr_idx[v_n:]

        tr_loader = DataLoader(Subset(ds, train_idx), batch_size=cfg.batch_train, shuffle=True, num_workers=cfg.num_workers)
        va_loader = DataLoader(Subset(ds, val_idx), batch_size=cfg.batch_eval, shuffle=False, num_workers=cfg.num_workers)
        te_loader = DataLoader(Subset(ds, te_idx), batch_size=cfg.batch_eval, shuffle=False, num_workers=cfg.num_workers)

        model = GCNUNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))

        best_val_r = -1.0
        stale = 0
        ckpt_path = os.path.join(cfg.out_dir, f"fold{fold}_unet.pt")

        for ep in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, tr_loader, opt, device, use_amp, scaler, aug_cfg, cfg)
            val_r, _, _ = eval_loader(model, va_loader, device, use_amp)
            sch.step()

            print(f"Ep {ep:02d} | train_huber={train_loss:.4f} | val_pearson={val_r:.4f}")

            if val_r > best_val_r:
                best_val_r = val_r
                stale = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                stale += 1
                if stale >= cfg.es_patience:
                    print("Early stopping.")
                    break

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_r, _, _ = eval_loader(model, te_loader, device, use_amp)

        fold_summary.append({"fold": fold, "best_val_pearson": best_val_r, "test_pearson": test_r, "checkpoint": ckpt_path})
        print(f"Fold {fold} test_pearson={test_r:.4f}")

    out_csv = os.path.join(cfg.out_dir, "cv_summary.json")
    with open(out_csv, "w") as f:
        json.dump(fold_summary, f, indent=2)

    print(f"\nSaved: {out_csv}")
    print("Done.")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", dest="pkl_path", required=True)
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--k", dest="k_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-train", type=int, default=512)
    p.add_argument("--batch-eval", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", dest="weight_decay", type=float, default=3e-4)
    p.add_argument("--lam-rank", type=float, default=0.40)
    p.add_argument("--lam-rc", type=float, default=0.30)
    p.add_argument("--amp-bf16", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--aug-shift", type=int, default=3)
    p.add_argument("--aug-mut", type=float, default=0.02)
    args = p.parse_args()

    use_amp = bool(args.amp_bf16) and not bool(args.no_amp)

    return TrainConfig(
        pkl_path=args.pkl_path,
        out_dir=args.out_dir,
        k_folds=args.k_folds,
        seed=args.seed,
        epochs=args.epochs,
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lam_rank=args.lam_rank,
        lam_rc=args.lam_rc,
        use_bf16_amp=use_amp,
        aug_shift_max=args.aug_shift,
        aug_mut_rate=args.aug_mut,
    )


if __name__ == "__main__":
    run(parse_args())

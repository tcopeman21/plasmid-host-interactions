from __future__ import annotations

import argparse
import gc
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.unet_sequence_model.model import GCNUNet
from models.unet_sequence_model.data import rc_onehot, augment_batch, AugmentConfig
from trace.dna import clean_dna

# The 4-part plasmid CSV columns used in your notebook:
SEQ_KEYS = ["Position 1 Sequence", "Position 2 Sequence", "Position 3 Sequence", "Position 4 Sequence"]


@dataclass(frozen=True)
class FinetuneConfig:
    window: int = 192
    stride: int = 20

    epochs: int = 10
    batch_size: int = 512
    lr: float = 5e-4
    weight_decay: float = 3e-4
    patience: int = 4
    grad_clip: float = 0.5

    # weak supervision augmentation
    aug_shift_max: int = 3
    aug_mut_rate: float = 0.02

    # RC consistency weight (your notebook used 0.3)
    lam_rc: float = 0.30

    # deterministic-ish
    seed: int = 42


def build_sequence(row: dict) -> str:
    return "".join(clean_dna(str(row.get(k, ""))) for k in SEQ_KEYS)


def onehot_encode(seq: str) -> np.ndarray:
    seq = clean_dna(seq)
    x = np.zeros((len(seq), 4), dtype=np.float32)
    m = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, ch in enumerate(seq):
        j = m.get(ch, None)
        if j is not None:
            x[i, j] = 1.0
    return x


def make_window_dataset(rows: list[dict], y: np.ndarray, cfg: FinetuneConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Weak supervision: each 192-bp window inherits the plasmid-level label.
    """
    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    for r, yi in zip(rows, y):
        if not np.isfinite(yi):
            continue
        seq = build_sequence(r)
        if len(seq) < cfg.window:
            continue
        for j in range(0, len(seq) - cfg.window + 1, cfg.stride):
            w = seq[j : j + cfg.window]
            X_list.append(onehot_encode(w))
            y_list.append(float(yi))

    if not X_list:
        raise RuntimeError("No windows produced. Check sequences/columns/window/stride.")

    X = torch.from_numpy(np.stack(X_list).astype(np.float32))  # (Nw, L, 4)
    yt = torch.from_numpy(np.asarray(y_list, dtype=np.float32))  # (Nw,)
    return X, yt


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> float:
    model.eval()
    huber = nn.HuberLoss()
    total, n = 0.0, 0

    amp_ctx = torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda"))
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()
        with amp_ctx:
            pred = model(xb).mean(dim=1)
            loss = huber(pred, yb)
        total += float(loss.item()) * len(yb)
        n += len(yb)
    return total / max(1, n)


def finetune_one(
    base_ckpt: str,
    out_ckpt: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: FinetuneConfig,
    device: torch.device,
) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    model = GCNUNet().to(device)
    model.load_state_dict(torch.load(base_ckpt, map_location=device), strict=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    huber = nn.HuberLoss()
    mse = nn.MSELoss()

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    aug_cfg = AugmentConfig(shift_max=cfg.aug_shift_max, mut_rate=cfg.aug_mut_rate)

    best_val = float("inf")
    stale = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot, n = 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float()

            xb_aug = augment_batch(xb, aug_cfg)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xb_aug).mean(dim=1)
                loss_sup = huber(pred, yb)

                pred_rc = model(rc_onehot(xb_aug)).mean(dim=1)
                loss_rc = mse(pred, pred_rc)

                loss = loss_sup + cfg.lam_rc * loss_rc

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            tot += float(loss_sup.item()) * len(yb)
            n += len(yb)

        train_loss = tot / max(1, n)
        val_loss = eval_loss(model, val_loader, device=device, use_amp=use_amp)

        print(f"ep {ep:02d} | train_huber={train_loss:.4f} | val_huber={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            stale = 0
            os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
            torch.save(model.state_dict(), out_ckpt)
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    # load best for final val loss
    model.load_state_dict(torch.load(out_ckpt, map_location=device), strict=True)
    final_val = eval_loss(model, val_loader, device=device, use_amp=use_amp)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"best_val_huber": float(best_val), "final_val_huber": float(final_val), "out_ckpt": out_ckpt}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--csv", required=True, help="Plasmid CSV containing Position 1..4 Sequence columns.")
    p.add_argument("--lfc-col", default="Average", help="Column containing plasmid-level LFC.")
    p.add_argument("--out-dir", required=True, help="Directory to save finetuned checkpoints and logs.")
    p.add_argument("--base-ckpts", nargs="+", required=True, help="Baseline UNet checkpoints to finetune (ensemble folds).")

    p.add_argument("--window", type=int, default=192)
    p.add_argument("--stride", type=int, default=20)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", dest="weight_decay", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=4)

    p.add_argument("--aug-shift", type=int, default=3)
    p.add_argument("--aug-mut", type=float, default=0.02)
    p.add_argument("--lam-rc", type=float, default=0.30)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.25, help="Fraction of plasmids in train_val used for val.")
    p.add_argument("--test-ratio", type=float, default=0.0, help="Set >0 if you want to reserve a test set here (usually 0; validation happens elsewhere).")
    return p.parse_args()


def main():
    a = parse_args()

    cfg = FinetuneConfig(
        window=a.window,
        stride=a.stride,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        patience=a.patience,
        aug_shift_max=a.aug_shift,
        aug_mut_rate=a.aug_mut,
        lam_rc=a.lam_rc,
        seed=a.seed,
    )

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "finetune_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    df = pd.read_csv(a.csv).dropna(subset=[a.lfc_col])
    rows = df.to_dict(orient="records")
    y = df[a.lfc_col].astype(np.float32).values

    # simple train/val split at plasmid level (fine: validation script should handle the strict split logic)
    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    val_n = int(len(idx) * a.val_frac)
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | plasmids train={len(train_rows)} val={len(val_rows)}")

    # build window datasets
    Xtr, ytr = make_window_dataset(train_rows, y_train, cfg)
    Xva, yva = make_window_dataset(val_rows, y_val, cfg)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    summary = []
    for base_ckpt in a.base_ckpts:
        base_name = os.path.splitext(os.path.basename(base_ckpt))[0]
        out_ckpt = os.path.join(a.out_dir, f"{base_name}_ft.pt")

        print(f"\n=== finetune {base_name} ===")
        res = finetune_one(base_ckpt, out_ckpt, train_loader, val_loader, cfg, device)
        res["base_ckpt"] = base_ckpt
        summary.append(res)

    with open(os.path.join(a.out_dir, "finetune_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Saved to: {a.out_dir}")


if __name__ == "__main__":
    main()

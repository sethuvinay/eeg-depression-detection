"""Training entry point: self-supervised pretraining + supervised fine-tuning.

Two-phase protocol (mirrors the original research):

1. ``--phase ssl`` — contrastive pretraining (NT-Xent) of the encoders + GNN
   on augmented views of each window. No labels needed.
2. ``--phase supervised`` — fine-tune the whole network with cross-entropy,
   optionally warm-started from an SSL checkpoint.

Splits are strictly subject-wise: no subject appears in both train and test.
"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data_loader import (CHEST_FS, LABEL_TO_INDEX, USABLE_LABELS,
                          WESADLoader)
from .models import MultimodalAffectNet, nt_xent_loss
from .preprocessing import (build_window_index, preprocess_chest,
                            window_labels)

#: Modalities fed to the model (chest signals with the richest affect signal).
MODALITIES = ["ECG", "EDA", "EMG", "Resp"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class WindowedWESADDataset(Dataset):
    """Sliding-window samples ``(modalities, time) -> class`` for subjects."""

    def __init__(self, data_dir: str, subjects, window_s: float = 60.0,
                 stride_s: float = 30.0, target_fs: float = 64.0):
        self.target_fs = target_fs
        self.window_n = int(window_s * target_fs)
        self.samples = []  # list of (window_array, class_index)

        loader = WESADLoader(data_dir)
        for subject in tqdm(subjects, desc="Loading subjects"):
            signals, labels = loader.load(subject)
            conditioned = preprocess_chest(signals, fs=CHEST_FS,
                                           target_fs=target_fs)
            # Keep only the modalities the model consumes.
            conditioned = {m: conditioned[m] for m in MODALITIES}
            # Align labels (chest rate) with the resampled signals.
            index = build_window_index(len(labels), CHEST_FS, target_fs,
                                       window_s, stride_s, labels)
            if not index:
                continue
            starts_chest = [s for s, _ in index]
            starts_rs = [s for _, s in index]
            win_chest = int(window_s * CHEST_FS)
            _, win_labels = window_labels(labels, win_chest,
                                          int(stride_s * CHEST_FS))
            for s_rs, raw_label in zip(starts_rs, win_labels):
                if raw_label not in USABLE_LABELS:
                    continue
                window = np.stack(
                    [conditioned[m][s_rs:s_rs + self.window_n]
                     for m in MODALITIES]
                ).astype(np.float32)
                self.samples.append((window, LABEL_TO_INDEX[int(raw_label)]))

        if not self.samples:
            raise RuntimeError("No training windows built — check data dir.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Stochastic augmentation for contrastive views.

    Applies the *same* transform to every modality of a window so that
    cross-modal correspondences the GNN should learn are preserved.
    """
    if random.random() < 0.8:  # jitter
        x = x + torch.randn_like(x) * 0.03
    if random.random() < 0.5:  # random scaling
        x = x * random.uniform(0.9, 1.1)
    if random.random() < 0.5:  # temporal shift
        shift = random.randint(-x.size(-1) // 20, x.size(-1) // 20)
        x = torch.roll(x, shifts=shift, dims=-1)
    return x


# --------------------------------------------------------------------------- #
# Training phases
# --------------------------------------------------------------------------- #
def pretrain_ssl(model, loader, device, epochs, lr, ckpt_path):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state = float("inf"), None
    for epoch in range(epochs):
        total, n = 0.0, 0
        for x, _ in tqdm(loader, desc=f"SSL epoch {epoch + 1}/{epochs}",
                         leave=False):
            x = x.to(device)
            z1 = model.project(augment(x))
            z2 = model.project(augment(x))
            loss = nt_xent_loss(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        avg = total / n
        print(f"[ssl] epoch {epoch + 1}: loss={avg:.4f}")
        if avg < best:
            best, best_state = avg, {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(best_state, ckpt_path)
    print(f"Saved best SSL checkpoint (loss={best:.4f}) -> {ckpt_path}")


def finetune_supervised(model, train_loader, val_loader, device, epochs, lr,
                        ckpt_path):
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"train {epoch + 1}/{epochs}",
                         leave=False):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate_accuracy(model, val_loader, device)
        sched.step(1 - acc)
        print(f"[sup] epoch {epoch + 1}: val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(best_state, ckpt_path)
    print(f"Saved best supervised checkpoint (val_acc={best_acc:.4f}) -> {ckpt_path}")


@torch.no_grad()
def evaluate_accuracy(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Train MultimodalAffectNet on WESAD")
    p.add_argument("--data-dir", required=True, help="directory with WESAD .pkl files")
    p.add_argument("--phase", choices=["ssl", "supervised"], required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--window-s", type=float, default=60.0)
    p.add_argument("--stride-s", type=float, default=30.0)
    p.add_argument("--target-fs", type=float, default=64.0)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--test-subjects", default="S16,S17",
                   help="comma-separated subject ids held out for validation/test")
    p.add_argument("--ssl-checkpoint", default=None,
                   help="warm-start supervised training from an SSL checkpoint")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = WESADLoader(args.data_dir)
    test_subjects = [s.strip() for s in args.test_subjects.split(",") if s.strip()]
    train_subjects = [s for s in loader.subjects if s not in test_subjects]
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Held-out subjects ({len(test_subjects)}): {test_subjects}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_ds = WindowedWESADDataset(args.data_dir, train_subjects,
                                    args.window_s, args.stride_s, args.target_fs)
    val_ds = WindowedWESADDataset(args.data_dir, test_subjects,
                                  args.window_s, args.stride_s, args.target_fs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    print(f"Windows: train={len(train_ds)}, val={len(val_ds)}")

    model = MultimodalAffectNet(modalities=MODALITIES,
                                emb_dim=args.emb_dim).to(device)
    if args.ssl_checkpoint:
        state = torch.load(args.ssl_checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Warm-started from {args.ssl_checkpoint}")

    if args.phase == "ssl":
        ckpt = os.path.join(args.checkpoint_dir, "ssl_best.pt")
        pretrain_ssl(model, train_loader, device, args.epochs, args.lr, ckpt)
    else:
        ckpt = os.path.join(args.checkpoint_dir, "supervised_best.pt")
        finetune_supervised(model, train_loader, val_loader, device,
                            args.epochs, args.lr, ckpt)


if __name__ == "__main__":
    main()

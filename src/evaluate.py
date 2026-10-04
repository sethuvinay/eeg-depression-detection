"""Evaluation on held-out subjects: accuracy, macro-F1, confusion matrix.

Example:
    python -m src.evaluate --data-dir data/WESAD \\
        --checkpoint checkpoints/supervised_best.pt \\
        --plot outputs/confusion_matrix.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from torch.utils.data import DataLoader

from .data_loader import INDEX_TO_LABEL, LABEL_NAMES, WESADLoader
from .models import MultimodalAffectNet
from .train import MODALITIES, WindowedWESADDataset


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MultimodalAffectNet")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-subjects", default="S16,S17")
    p.add_argument("--window-s", type=float, default=60.0)
    p.add_argument("--stride-s", type=float, default=30.0)
    p.add_argument("--target-fs", type=float, default=64.0)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--plot", default=None, help="path to save confusion matrix PNG")
    return p.parse_args()


def plot_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (held-out subjects)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Confusion matrix saved -> {path}")


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = WESADLoader(args.data_dir)
    test_subjects = [s.strip() for s in args.test_subjects.split(",") if s.strip()]
    print(f"Evaluating on held-out subjects: {test_subjects}")

    test_ds = WindowedWESADDataset(args.data_dir, test_subjects,
                                   args.window_s, args.stride_s, args.target_fs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=2)

    model = MultimodalAffectNet(modalities=MODALITIES, emb_dim=args.emb_dim).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_true, all_pred = [], []
    for x, y in test_loader:
        x = x.to(device)
        all_pred.extend(model(x).argmax(1).cpu().tolist())
        all_true.extend(y.tolist())

    class_names = [LABEL_NAMES[INDEX_TO_LABEL[i]] for i in range(len(INDEX_TO_LABEL))]
    acc = accuracy_score(all_true, all_pred)
    print(f"\nAccuracy (held-out subjects): {acc:.4f}")
    print(classification_report(all_true, all_pred, target_names=class_names,
                                digits=4))
    if args.plot:
        cm = confusion_matrix(all_true, all_pred)
        plot_confusion_matrix(cm, class_names, args.plot)


if __name__ == "__main__":
    main()

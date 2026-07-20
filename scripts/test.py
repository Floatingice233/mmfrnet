"""MMFRNet evaluation script."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn

from utils.config import get_config
from utils.dataset import get_dataset, get_num_classes
from models.mmfrnet import MMFRNet
from utils.metrics import compute_metrics


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_outputs = []
    all_targets = []

    for view_a, view_b, targets in loader:
        view_a = view_a.to(device)
        view_b = view_b.to(device)
        targets = targets.to(device)

        outputs = model(view_a, view_b)
        all_outputs.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return all_outputs, all_targets


def main():
    args = get_config()

    if args.num_classes is None:
        args.num_classes = get_num_classes(args.dataset)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    test_set = get_dataset(args.dataset, args.data_dir, "test", args.img_size)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Test samples: {len(test_set)}")

    # Load model
    model = MMFRNet(
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        neighborhood_size=args.neighborhood_size,
        gc_reduction=args.gc_reduction,
    ).to(device)

    checkpoint_path = os.path.join(args.save_dir, "best.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val ACC={checkpoint['acc']:.4f}")

    # Evaluate
    outputs, targets = evaluate(model, test_loader, device)
    acc, auc = compute_metrics(outputs, targets, args.num_classes)
    print(f"\nTest Results on {args.dataset}:")
    print(f"  ACC: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")


if __name__ == "__main__":
    main()

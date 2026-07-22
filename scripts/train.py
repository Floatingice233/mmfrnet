"""MMFRNet training script."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import time
from datetime import datetime
import shutil
import numpy as np
import torch
import torch.nn as nn

from utils.config import get_config
from utils.dataset import get_dataset, get_num_classes
from models.mmfrnet import MMFRNet
from utils.metrics import compute_metrics

# Force unbuffered output for background execution
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, "reconfigure") else None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, acc, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "acc": acc,
    }, path)


def train_epoch(model, loader, optimizer, criterion, device, epoch, args):
    model.train()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    for batch_idx, (view_a, view_b, targets) in enumerate(loader):
        view_a = view_a.to(device)
        view_b = view_b.to(device)
        targets = targets.to(device)

        outputs = model(view_a, view_b)
        raw_loss = criterion(outputs, targets)
        loss = raw_loss / args.grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % args.grad_accum_steps == 0 or \
           batch_idx == len(loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += raw_loss.item()
        all_outputs.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

        if batch_idx % args.log_interval == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {raw_loss.item():.4f}",
                  flush=True)

    avg_loss = total_loss / len(loader)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    train_acc, train_auc = compute_metrics(all_outputs, all_targets,
                                           args.num_classes)

    return avg_loss, train_acc, train_auc


@torch.no_grad()
def evaluate(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    for view_a, view_b, targets in loader:
        view_a = view_a.to(device)
        view_b = view_b.to(device)
        targets = targets.to(device)

        outputs = model(view_a, view_b)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        all_outputs.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(loader)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    acc, auc = compute_metrics(all_outputs, all_targets, args.num_classes)

    return avg_loss, acc, auc


def main():
    args = get_config()

    if args.num_classes is None:
        args.num_classes = get_num_classes(args.dataset)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    micro_batch = args.batch_size // args.grad_accum_steps
    print(f"Loading dataset: {args.dataset}")
    print(f"Effective batch size: {args.batch_size} "
          f"(micro={micro_batch} x accum={args.grad_accum_steps})")
    train_set = get_dataset(args.dataset, args.data_dir, "train", args.img_size)
    val_set = get_dataset(args.dataset, args.data_dir, "val", args.img_size)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=micro_batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    print(f"Number of classes: {args.num_classes}")

    # Model
    model = MMFRNet(
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        neighborhood_size=args.neighborhood_size,
        gc_reduction=args.gc_reduction,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # Optimizer (SGD as per paper)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        t0 = time.time()

        train_loss, train_acc, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args,
        )
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device, args,
        )

        elapsed = time.time() - t0
        if elapsed < 1:
            elapsed_str = f"{elapsed*1000:.0f}ms"
        else:
            elapsed_str = f"{elapsed:.1f}s"

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts} | {elapsed_str:>6s} | "
              f"Epoch {epoch:3d}/{args.epochs} | lr={current_lr:.5f} | "
              f"Train Loss: {train_loss:.4f} ACC: {train_acc:.4f} "
              f"AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} ACC: {val_acc:.4f} "
              f"AUC: {val_auc:.4f}", flush=True)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc,
                           os.path.join(args.save_dir, "best.pth"))
            print(f"  -> New best saved (ACC={val_acc:.4f})")

        save_checkpoint(model, optimizer, epoch, val_acc,
                       os.path.join(args.save_dir, "last.pth"))

    print(f"\nTraining complete. Best val ACC: {best_val_acc:.4f}")

    # Final test evaluation
    print(f"\nLoading test set...")
    test_set = get_dataset(args.dataset, args.data_dir, "test", args.img_size)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    checkpoint = torch.load(os.path.join(args.save_dir, "best.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_auc = evaluate(
        model, test_loader, criterion, device, args,
    )
    print(f"Test  Loss: {test_loss:.4f}, ACC: {test_acc:.4f}, "
          f"AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()

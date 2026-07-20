"""Continue training each dataset for 30 more epochs, then test eval."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from models.mmfrnet import MMFRNet
from utils.dataset import get_dataset, get_num_classes
from utils.metrics import compute_metrics


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for view_a, view_b, labels in loader:
            view_a, view_b = view_a.to(device), view_b.to(device)
            labels = labels.to(device)
            outputs = model(view_a, view_b)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * view_a.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / len(loader.dataset)
    acc, auc = compute_metrics(preds, labels, num_classes)
    return avg_loss, acc, auc


def train_epoch(model, loader, optimizer, criterion, device, num_classes):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for view_a, view_b, labels in loader:
        view_a, view_b = view_a.to(device), view_b.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(view_a, view_b)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * view_a.size(0)
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / len(loader.dataset)
    acc, auc = compute_metrics(preds, labels, num_classes)
    return avg_loss, acc, auc


def continue_training(dataset_name, data_dir, ckpt_dir, device, extra_epochs=30,
                      log_f=None):
    if log_f is None:
        log_f = sys.stdout

    def tprint(*args, **kwargs):
        print(*args, **kwargs, file=log_f, flush=True)

    tprint(f"Continue training {dataset_name} for {extra_epochs} more epochs")
    tprint(f"{'='*60}")

    num_classes = get_num_classes(dataset_name)

    train_set = get_dataset(dataset_name, data_dir, "train", 224)
    val_set = get_dataset(dataset_name, data_dir, "val", 224)
    test_set = get_dataset(dataset_name, data_dir, "test", 224)
    tprint(f"Train: {len(train_set)}, Val: {len(val_set)}, "
           f"Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False,
                             num_workers=0, pin_memory=True)

    model = MMFRNet(
        num_classes=num_classes, embed_dim=64, gc_reduction=8,
        window_size=8, neighborhood_size=7,
    ).to(device)

    ckpt_path = os.path.join(ckpt_dir, dataset_name, "best.pth")
    last_path = os.path.join(ckpt_dir, dataset_name, "last.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    prev_val_acc = ckpt.get("acc", ckpt.get("val_acc", 0))
    prev_val_auc = ckpt.get("val_auc", 0)
    tprint(f"Loaded checkpoint: epoch={start_epoch}, "
           f"val_acc={prev_val_acc:.4f}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    best_val_acc = prev_val_acc
    best_val_auc = prev_val_auc
    best_epoch = start_epoch

    for ep in range(1, extra_epochs + 1):
        current_epoch = start_epoch + ep
        t0 = time.time()
        train_loss, train_acc, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device, num_classes)
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device, num_classes)
        elapsed = time.time() - t0
        if elapsed < 1:
            elapsed_str = f"{elapsed*1000:.0f}ms"
        else:
            elapsed_str = f"{elapsed:.1f}s"

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tprint(f"{ts} | {elapsed_str:>6s} | "
               f"Epoch {current_epoch:3d} | "
               f"Train Loss: {train_loss:.4f} ACC: {train_acc:.4f} "
               f"AUC: {train_auc:.4f} | "
               f"Val Loss: {val_loss:.4f} ACC: {val_acc:.4f} "
               f"AUC: {val_auc:.4f}")

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": current_epoch,
            "val_acc": val_acc,
            "val_auc": val_auc,
        }, last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_auc = val_auc
            best_epoch = current_epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": current_epoch,
                "val_acc": val_acc,
                "val_auc": val_auc,
            }, ckpt_path)
            tprint(f"  -> New best saved (ACC={val_acc:.4f})")

    tprint(f"\nBest Val: epoch={best_epoch}, ACC={best_val_acc:.4f}, "
           f"AUC={best_val_auc:.4f}")

    # Test evaluation
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc, test_auc = evaluate(
        model, test_loader, criterion, device, num_classes)
    tprint(f"TEST  Loss: {test_loss:.4f}, ACC: {test_acc:.4f}, "
           f"AUC: {test_auc:.4f}")

    return best_val_acc, best_val_auc, test_acc, test_auc


def main():
    datasets = ["pneumoniamnist", "breastmnist", "retinamnist",
                "dermamnist", "organcmnist"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data"
    ckpt_dir = "./checkpoints"
    extra = 30

    results = {}
    for ds in datasets:
        log_dir = f"logs/{ds}"
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"{log_dir}/resume_{ts}.log"
        log_f = open(log_path, 'w', buffering=1)
        try:
            va, v_auc, ta, t_auc = continue_training(
                ds, data_dir, ckpt_dir, device, extra, log_f)
            results[ds] = (va, v_auc, ta, t_auc)
        finally:
            log_f.close()
        print(f"{ds}: Val ACC={va:.4f} AUC={v_auc:.4f} | "
              f"Test ACC={ta:.4f} AUC={t_auc:.4f}", flush=True)

    # Final comparison
    paper = {
        "pneumoniamnist": (0.979, 0.928),
        "breastmnist": (0.902, 0.896),
        "retinamnist": (0.753, 0.583),
        "dermamnist": (0.893, 0.774),
        "organcmnist": (0.996, 0.951),
    }
    print(f"\n{'='*70}")
    print("FINAL COMPARISON (after +30 epochs)")
    print(f"{'Dataset':<22s} {'Paper AUC':>8s} {'Paper ACC':>8s} "
          f"{'Test AUC':>8s} {'Test ACC':>8s} {'AUC Gap':>8s} {'ACC Gap':>8s}")
    print("-" * 70)
    for ds, (p_auc, p_acc) in paper.items():
        _, _, ta, t_auc = results[ds]
        print(f"{ds:<22s} {p_auc:8.4f} {p_acc:8.4f} "
              f"{t_auc:8.4f} {ta:8.4f} "
              f"{t_auc-p_auc:+8.4f} {ta-p_acc:+8.4f}")


if __name__ == "__main__":
    main()

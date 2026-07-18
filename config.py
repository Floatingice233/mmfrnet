"""Configuration for MMFRNet training and model."""

import argparse


def get_config():
    parser = argparse.ArgumentParser(description="MMFRNet Training")

    # Dataset
    parser.add_argument("--dataset", type=str, default="pneumoniamnist",
                        choices=["dermamnist", "pneumoniamnist", "retinamnist",
                                 "breastmnist", "organmnist_coronal"],
                        help="MedMNIST dataset name")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to store datasets")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes (auto-detected if None)")

    # Input
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size after resize")
    parser.add_argument("--crop_ratio", type=float, default=0.75,
                        help="Center crop ratio for view B (0.7~0.8)")

    # Model
    parser.add_argument("--embed_dim", type=int, default=64,
                        help="Embedding dimension C for patch stem")
    parser.add_argument("--gc_reduction", type=int, default=8,
                        help="Reduction ratio r for Enhanced Global Context")
    parser.add_argument("--window_size", type=int, default=8,
                        help="Window size for WNA")
    parser.add_argument("--neighborhood_size", type=int, default=7,
                        help="Neighborhood size k for WNA")

    # Training
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Total training epochs")
    parser.add_argument("--lr_init", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--lr_step", type=int, default=100,
                        help="Epoch to decay learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.1,
                        help="Learning rate decay factor")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")

    # System
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Logging
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log every N steps")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Model checkpoint directory")

    return parser.parse_args()

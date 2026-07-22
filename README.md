# MMFRNet / MMFRNet 多视图多粒度特征正则化网络

PyTorch implementation of **MMFRNet: Quaternion Gated Convolution-based Multi-view and Multi-granularity Feature Regularization Network for Undefined Medical Image Classification**.

**MMFRNet**（基于四元数门控卷积的多视图多粒度特征正则化网络）的 PyTorch 复现实现，用于医学图像分类任务。

## Architecture / 网络架构

MMFRNet extends single-view medical images into multi-view data through data augmentation (center crop + CLAHE), then extracts multi-granularity features using four core modules:

MMFRNet 通过数据增强（中心裁剪+CLAHE）将单视图数据扩展为多视图，利用四个核心模块提取多粒度特征：

- **GIR** (Global Information Regularization / 全局信息正则化): Depthwise Conv + Enhanced Global Context for global features
- **LIR** (Local Information Regularization / 局部信息正则化): Window Neighborhood Attention (WNA) for local features
- **FFTConv** (Fast Fourier Transform Convolution / 傅里叶变换卷积): Frequency-domain convolution via FFT for complementary feature extraction
- **QGConv** (Quaternion Gated Convolution / 四元数门控卷积): Quaternion convolution with gating for dynamic multi-view/multi-granularity fusion

## Environment / 环境配置

### Conda (recommended / 推荐)

```bash
conda create -n mmfrnet python=3.10 -y
conda activate mmfrnet
pip install -r requirements.txt
```

### Pip

```bash
pip install -r requirements.txt
```

## Datasets / 数据集

Uses [MedMNIST](https://medmnist.com/) benchmark datasets / 使用 MedMNIST 公开基准数据集:

| 数据集 | 模态 | 类别数 | 任务 |
|--------|------|--------|------|
| DermaMNIST | 皮肤镜 / Dermatoscope | 7 | 皮肤病变分类 |
| PneumoniaMNIST | 胸部X光 / Chest X-ray | 2 | 肺炎二分类 |
| RetinaMNIST | 眼底相机 / Fundus Camera | 5 | 糖尿病视网膜病变分级 |
| BreastMNIST | 乳腺超声 / Breast Ultrasound | 2 | 乳腺癌二分类 |
| OrganMNIST_Coronal | 腹部CT / Abdominal CT | 11 | 器官多分类 |

Datasets are downloaded automatically on first use / 数据集首次使用时自动下载。

## Quick Start / 快速开始

```bash
# Train all 5 datasets sequentially
bash scripts/run_all.sh

# Or train individual datasets
python scripts/train.py --dataset pneumoniamnist
```

## Training / 训练

Training matches the paper's strategy:
- **Optimizer**: SGD, momentum=0.9, weight_decay=1e-4
- **LR schedule**: 0.01 initial → 0.001 at epoch 100 (StepLR, gamma=0.1)
- **Effective batch size**: 128 (via gradient accumulation: micro=32 × 4)

```bash
# Train all 5 datasets sequentially (150 epochs each)
bash scripts/run_all.sh

# Train a single dataset
python scripts/train.py --dataset pneumoniamnist

# Full options
python scripts/train.py --dataset dermamnist --batch_size 128 --epochs 150
```

### Resume from checkpoint / 从检查点续训

Resumes from `last.pth`, restores optimizer/scheduler state, and continues for N more epochs.

```bash
# Via command line (edit resume.py main() to adjust dataset/epochs)
python scripts/resume.py
```

## Testing / 测试

```bash
python scripts/test.py --dataset pneumoniamnist --save_dir checkpoints/pneumoniamnist
```

## Gradient Accumulation / 梯度累积

To match the paper's batch_size=128 on 8GB VRAM, gradient accumulation is used:
- **micro batch**: 32 samples per forward pass
- **accumulation steps**: 4
- **effective batch**: 128 (equivalent to paper)

This keeps BN statistics slightly noisier (32-sample batches vs 128), but optimizer updates are mathematically identical to the paper's setting.

## Key Parameters / 关键参数

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 128 | Effective batch size (micro=batch_size/grad_accum_steps) |
| `--grad_accum_steps` | 4 | Gradient accumulation steps |
| `--epochs` | 150 | Total training epochs |
| `--lr_init` | 0.01 | Initial learning rate (SGD) |
| `--lr_step` | 100 | Epoch to decay lr (×0.1) |
| `--lr_decay` | 0.1 | LR decay factor |
| `--momentum` | 0.9 | SGD momentum |
| `--weight_decay` | 1e-4 | Weight decay |
| `--embed_dim` | 64 | Base embedding dimension C |
| `--crop_ratio` | 0.75 | Center crop ratio for view B |
| `--window_size` | 8 | Window size for WNA |
| `--neighborhood_size` | 7 | Neighborhood k for WNA |
| `--log_interval` | 50 | Log every N batches |
| `--num_workers` | 0 | DataLoader workers (0 for Windows) |

## Model Variants / 模型变体

- `MMFRNet(num_classes, embed_dim=64)` — Base / 基础版 (22.8M 参数)
- `MMFRNet(num_classes, embed1_dim=32)` — Small / 轻量版 (更少参数)

## Project Structure / 项目结构

```
project/
├── data/                        # Datasets
├── scripts/                     # Entry-point scripts
│   ├── train.py                 # Train a single dataset
│   ├── test.py                  # Evaluate a trained model
│   ├── resume.py                # Resume training from checkpoint
│   └── run_all.sh               # Train all 5 datasets sequentially
├── models/                      # Model modules
│   ├── gir.py                   # Global Information Regularization
│   ├── lir.py                   # Local Information Regularization (WNA)
│   ├── fftconv.py               # FFT frequency-domain convolution
│   ├── qgconv.py                # Quaternion Gated Convolution
│   └── mmfrnet.py               # Full MMFRNet model
├── utils/                       # Utilities
│   ├── config.py                # Hyperparameter configuration
│   ├── dataset.py               # MedMNIST data loading
│   ├── transforms.py            # Multi-view data augmentation
│   └── metrics.py               # ACC/AUC evaluation metrics
├── checkpoints/<dataset>/       # Model checkpoints (best.pth, last.pth)
├── logs/<dataset>/              # Training logs (timestamped)
└── requirements.txt
```

## Reference / 参考文献

```
@article{xie2026mmfrnet,
    title={MMFRNet: Quaternion Gated Convolution-based Multi-view 
           and Multi-granularity Feature Regularization Network 
           for Undefined Medical Image Classification},
    author={Siyue Xie and Guoheng Huang},
    journal={Artificial Intelligence and Emerging Technologies},
    year={2026}
}
```

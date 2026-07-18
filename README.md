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

## Training / 训练

```bash
# Train on PneumoniaMNIST / 在肺炎数据集上训练
python train.py --dataset pneumoniamnist

# Train on other datasets / 训练其他数据集
python train.py --dataset dermamnist --batch_size 64 --epochs 150

# Full options / 完整参数
python train.py --dataset pneumoniamnist --embed_dim 64 --batch_size 128 --epochs 150 --lr_init 0.01
```

## Testing / 测试

```bash
python test.py --dataset pneumoniamnist --save_dir ./checkpoints
```

## Key Parameters / 关键参数

| Parameter / 参数 | Default / 默认值 | Description / 说明 |
|-----------|---------|-------------|
| `--embed_dim` | 64 | Base embedding dimension C / 基础嵌入维度 |
| `--batch_size` | 128 | Training batch size / 训练批次大小 |
| `--epochs` | 150 | Total training epochs / 总训练轮数 |
| `--lr_init` | 0.01 | Initial learning rate (SGD) / 初始学习率 |
| `--crop_ratio` | 0.75 | Center crop ratio for view B / 视图B的中心裁剪比例 |
| `--window_size` | 8 | Window size for WNA / WNA的窗口大小 |
| `--neighborhood_size` | 7 | Neighborhood k for WNA / WNA的邻域大小k |

## Model Variants / 模型变体

- `MMFRNet(num_classes, embed_dim=64)` — Base / 基础版 (22.8M 参数)
- `MMFRNet(num_classes, embed_dim=32)` — Small / 轻量版 (更少参数)

## Project Structure / 项目结构

```
project/
├── data/                        # 数据集存放
├── models/
│   ├── gir.py                   # 全局信息正则化模块
│   ├── lir.py                   # 局部信息正则化模块 (WNA)
│   ├── fftconv.py               # 傅里叶频域卷积模块
│   ├── qgconv.py                # 四元数门控卷积模块
│   └── mmfrnet.py               # 完整MMFRNet模型
├── utils/
│   ├── transforms.py            # 多视图数据增强
│   └── metrics.py               # ACC/AUC评估指标
├── config.py                    # 超参数配置
├── dataset.py                   # MedMNIST数据加载
├── train.py                     # 训练主脚本
├── test.py                      # 测试评估脚本
└── requirements.txt             # 依赖列表
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

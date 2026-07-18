"""Global Information Regularization (GIR) Module.

Depthwise Conv + Enhanced Global Context with Competitive Info Bottleneck.
"""

import torch
import torch.nn as nn


class EnhancedGlobalContext(nn.Module):
    """Enhanced Global Context mechanism with Competitive Info Bottleneck.

    gc = W_r * maxout(W_s1 * X_avg, W_s2 * X_avg)
    where X_avg = (1/n) * sum(x_i) is global average pooled feature.
    """

    def __init__(self, dim, reduction=8):
        super().__init__()
        reduced_dim = dim // reduction
        self.compress1 = nn.Linear(dim, reduced_dim)
        self.compress2 = nn.Linear(dim, reduced_dim)
        self.expand = nn.Linear(reduced_dim, dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_avg = x.mean(dim=[2, 3])  # (B, C)

        # Competitive compression
        z1 = self.compress1(x_avg)  # (B, C/r)
        z2 = self.compress2(x_avg)  # (B, C/r)
        z = torch.max(z1, z2)  # maxout

        # Expand and normalize
        gc = self.expand(z)  # (B, C)
        gc = torch.softmax(gc, dim=1)  # (B, C)
        return gc


class GIR(nn.Module):
    """Global Information Regularization Module.

    Y = softmax(gc) * DW(x)
    where DW is a Depthwise Conv with BN and GELU.
    """

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.gc = EnhancedGlobalContext(dim, reduction)

    def forward(self, x):
        # x: (B, C, H, W)
        dw_out = self.dw(x)  # (B, C, H, W)
        gc = self.gc(x)  # (B, C)

        # Element-wise weighting: softmax(gc) * dw_out
        gc = gc.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        out = gc * dw_out  # (B, C, H, W)
        return out

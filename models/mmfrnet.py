"""MMFRNet: Multi-view and Multi-granularity Feature Regularization Network.

Quaternion Gated Convolution-based architecture for medical image classification.
"""

import torch
import torch.nn as nn

from models.gir import GIR
from models.lir import LIR
from models.fftconv import FFTConv
from models.qgconv import QGConv


class PatchEmbed(nn.Module):
    """Patch embedding: 4x4 conv stride 4 → H/4 x W/4 x C."""

    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        return self.norm(self.conv(x))


class BranchStage(nn.Module):
    """Feature extraction: GIR + LIR + FFTConv in parallel."""

    def __init__(self, dim, window_size=8, neighborhood_size=7, num_heads=4,
                 gc_reduction=8):
        super().__init__()
        self.gir = GIR(dim, gc_reduction)
        self.lir = LIR(dim, window_size, neighborhood_size, num_heads)
        self.fftconv = FFTConv(dim)

    def forward(self, x):
        g = self.gir(x)
        l = self.lir(x, global_feat=g)
        f = self.fftconv(x)
        return g, l, f


def _ch4(x):
    """Round up to nearest multiple of 4 (for quaternion compatibility)."""
    return ((x + 3) // 4) * 4


class MMFRNet(nn.Module):
    """MMFRNet for medical image classification.

    Args:
        num_classes: number of output classes
        embed_dim: base channel dimension C (must be multiple of 4)
        window_size: WNA window size
        neighborhood_size: WNA neighborhood size k
        gc_reduction: reduction ratio for Enhanced Global Context
    """

    def __init__(self, num_classes, embed_dim=64, window_size=8,
                 neighborhood_size=7, gc_reduction=8):
        super().__init__()
        C = _ch4(embed_dim)
        num_heads = max(1, C // 16)

        # Stem
        self.stem_a = PatchEmbed(3, C)
        self.stem_b = PatchEmbed(3, C)

        # ── Round 1 (H/4 → H/8) ──
        self.stage1 = BranchStage(C, window_size, neighborhood_size,
                                  num_heads, gc_reduction)
        # 3 feature types → QGConv → 4 components, then keep C channels
        self.qgconv1_a = QGConv(C * 3, C, downsample=True)
        self.qgconv1_b = QGConv(C * 3, C, downsample=True)
        self.qgconv1_c = QGConv(C * 6, C, downsample=True)  # A+B fused

        # ── Round 2 (H/8 → H/16) ──
        ws2 = max(2, window_size // 2)
        self.stage2 = BranchStage(C, ws2, neighborhood_size,
                                  num_heads, gc_reduction)
        self.qgconv2_a = QGConv(C * 3, C, downsample=True)
        self.qgconv2_b = QGConv(C * 3, C, downsample=True)
        self.qgconv2_c = QGConv(C * 6, C, downsample=True)

        # ── Round 3 (H/16 → feature extraction only, no downsample) ──
        ws3 = max(2, window_size // 4)
        self.stage3 = BranchStage(C, ws3, neighborhood_size,
                                  num_heads, gc_reduction)
        self.qgconv3_a = QGConv(C * 3, C * 4, downsample=False)
        self.qgconv3_b = QGConv(C * 3, C * 4, downsample=False)
        self.qgconv3_c = QGConv(C * 3, C * 4, downsample=False)

        # ── Granularity alignment ──
        # Per-component dim after split: C*4/4 = C
        # 3 sources → C*3 input, C output
        self.qgconv_align_i = QGConv(C * 3, C, downsample=False)
        self.qgconv_align_j = QGConv(C * 3, C, downsample=False)
        self.qgconv_align_k = QGConv(C * 3, C, downsample=False)
        self.qgconv_align_r = QGConv(C * 3, C, downsample=False)

        # ── Final aggregation ──
        # 4 components × C = 4C → downsample → H/32
        final_out = _ch4(512)
        self.down_final = nn.Conv2d(C * 4, C * 8, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(C * 8, final_out, kernel_size=1),
            nn.BatchNorm2d(final_out),
            nn.GELU(),
        )

        # Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_out, 1280)
        self.classifier = nn.Linear(1280, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, view_a, view_b):
        C = self.stem_a.conv.out_channels

        # Stem
        a = self.stem_a(view_a)  # (B, C, H/4, W/4)
        b = self.stem_b(view_b)
        c = a + b  # C-branch starts as sum of A and B

        # ═══ Round 1 ═══
        a_g, a_l, a_f = self.stage1(a)
        b_g, b_l, b_f = self.stage1(b)
        c_g, c_l, c_f = self.stage1(c)

        a_cat = torch.cat([a_g, a_l, a_f], dim=1)  # 3C
        b_cat = torch.cat([b_g, b_l, b_f], dim=1)  # 3C
        c_cat = torch.cat([c_g, c_l, c_f], dim=1)  # 3C

        # QGConv + downsample: 3C → C @ H/8
        a1 = self.qgconv1_a(a_cat)
        b1 = self.qgconv1_b(b_cat)
        # C-branch fuses pooled A+B info
        c1 = self.qgconv1_c(torch.cat([a_cat, b_cat], dim=1))

        # Cross-branch exchange: add real-part information across branches
        _tmp_a1 = a1
        a1 = a1 + b1
        b1 = b1 + _tmp_a1

        # ═══ Round 2 ═══
        a_g2, a_l2, a_f2 = self.stage2(a1)
        b_g2, b_l2, b_f2 = self.stage2(b1)
        c_g2, c_l2, c_f2 = self.stage2(c1)

        a_cat2 = torch.cat([a_g2, a_l2, a_f2], dim=1)
        b_cat2 = torch.cat([b_g2, b_l2, b_f2], dim=1)
        c_cat2 = torch.cat([c_g2, c_l2, c_f2], dim=1)

        # QGConv + downsample: 3C → C @ H/16
        a2 = self.qgconv2_a(a_cat2)
        b2 = self.qgconv2_b(b_cat2)
        c2 = self.qgconv2_c(torch.cat([a_cat2, b_cat2], dim=1))

        _tmp_a2 = a2
        a2 = a2 + b2
        b2 = b2 + _tmp_a2

        # ═══ Round 3 (no downsample) ═══
        a_g3, a_l3, a_f3 = self.stage3(a2)
        b_g3, b_l3, b_f3 = self.stage3(b2)
        c_g3, c_l3, c_f3 = self.stage3(c2)

        a_cat3 = torch.cat([a_g3, a_l3, a_f3], dim=1)  # 3C
        b_cat3 = torch.cat([b_g3, b_l3, b_f3], dim=1)  # 3C
        c_cat3 = torch.cat([c_g3, c_l3, c_f3], dim=1)  # 3C

        # QGConv: 3C → 4C (quaternion structure)
        a_q3 = self.qgconv3_a(a_cat3)  # 4C
        b_q3 = self.qgconv3_b(b_cat3)  # 4C
        c_q3 = self.qgconv3_c(c_cat3)  # 4C

        # Split into 4 quaternion components (each C channels)
        cp = a_q3.shape[1] // 4  # = C
        a_r, a_i, a_j, a_k = a_q3.split(cp, dim=1)
        b_r, b_i, b_j, b_k = b_q3.split(cp, dim=1)
        c_r, c_i, c_j, c_k = c_q3.split(cp, dim=1)

        # Granularity alignment: group same component across 3 branches
        i_f = self.qgconv_align_i(torch.cat([a_i, b_i, c_i], dim=1))  # 3C→C
        j_f = self.qgconv_align_j(torch.cat([a_j, b_j, c_j], dim=1))
        k_f = self.qgconv_align_k(torch.cat([a_k, b_k, c_k], dim=1))
        r_f = self.qgconv_align_r(torch.cat([a_r, b_r, c_r], dim=1))

        out = torch.cat([i_f, j_f, k_f, r_f], dim=1)  # 4C

        # Final processing: H/16 → H/32
        out = self.down_final(out)  # 8C @ H/32
        out = self.final_conv(out)
        out = self.gap(out).flatten(1)  # final_out
        out = self.fc(out)  # 1280
        out = self.dropout(out)
        out = self.classifier(out)  # num_classes

        return out


def mmfrnet_small(num_classes, **kwargs):
    return MMFRNet(num_classes, embed_dim=32, **kwargs)


def mmfrnet_base(num_classes, **kwargs):
    return MMFRNet(num_classes, embed_dim=64, **kwargs)

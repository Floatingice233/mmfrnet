"""Quaternion Gated Convolution (QGConv) Module.

Quaternion convolution with gating mechanism for dynamic feature fusion.
Output = BatchNorm(LeakyReLU(X) * Sigmoid(M))
where X and M are both quaternion convolution outputs.
"""

import torch
import torch.nn as nn


class QuatConv2d(nn.Module):
    """Quaternion Convolution.

    Represents data as quaternion with 4 components (r, i, j, k).
    Uses Hamilton product to perform convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, \
            "Channels must be multiples of 4 for quaternion convolution"

        self.in_comp = in_channels // 4
        self.out_comp = out_channels // 4

        # Each conv operates on one quaternion component (in_comp → out_comp)
        self.r_conv = nn.Conv2d(self.in_comp, self.out_comp, kernel_size,
                                stride, padding)
        self.i_conv = nn.Conv2d(self.in_comp, self.out_comp, kernel_size,
                                stride, padding)
        self.j_conv = nn.Conv2d(self.in_comp, self.out_comp, kernel_size,
                                stride, padding)
        self.k_conv = nn.Conv2d(self.in_comp, self.out_comp, kernel_size,
                                stride, padding)

    def forward(self, x):
        # x: (B, C_in, H, W) where C_in is multiple of 4
        B, C_in, H, W = x.shape
        assert C_in % 4 == 0

        # Split into 4 components: r, i, j, k
        comp_dim = C_in // 4
        r, i, j, k = x.split(comp_dim, dim=1)

        # Hamilton product: W ⊗ X
        # Apply separate convs to each component and combine
        r_out = (self.r_conv(r) - self.i_conv(i) -
                 self.j_conv(j) - self.k_conv(k))
        i_out = (self.r_conv(i) + self.i_conv(r) +
                 self.j_conv(k) - self.k_conv(j))
        j_out = (self.r_conv(j) - self.i_conv(k) +
                 self.j_conv(r) + self.k_conv(i))
        k_out = (self.r_conv(k) + self.i_conv(j) -
                 self.j_conv(i) + self.k_conv(r))

        out = torch.cat([r_out, i_out, j_out, k_out], dim=1)
        return out


class QGConv(nn.Module):
    """Quaternion Gated Convolution.

    Uses two quaternion convolutions: one for feature extraction (main)
    and one for gating mask. The gating dynamically controls information flow.

    Output = BatchNorm(LeakyReLU(main_conv(x)) * Sigmoid(mask_conv(x)))

    Optionally followed by a 1×1 or 2×2 conv for dimension/downsampling adjustment.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, downsample=False, use_bn=True):
        super().__init__()
        # Ensure quaternion-compatible channels
        mid_channels = in_channels  # keep same for gating

        self.main_conv = QuatConv2d(in_channels, out_channels, kernel_size,
                                    stride, padding)
        self.mask_conv = QuatConv2d(in_channels, out_channels, kernel_size,
                                    stride, padding)

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # Optional adjustment convolution
        if downsample:
            self.adjust = nn.Conv2d(out_channels, out_channels, kernel_size=2,
                                    stride=2)
        else:
            self.adjust = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        main = self.main_conv(x)
        mask = self.mask_conv(x)

        main = self.act(main)
        mask = torch.sigmoid(mask)

        out = main * mask
        out = self.bn(out)
        out = self.adjust(out)
        return out

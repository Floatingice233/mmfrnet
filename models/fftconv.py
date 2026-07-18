"""Fast Fourier Transform Convolution (FFTConv) Module.

FFT → 1×1 Conv + ReLU → 1×1 Conv → IFFT → + residual
"""

import torch
import torch.nn as nn


class FFTConv(nn.Module):
    """FFT-based convolution for frequency-domain feature extraction.

    Operates in frequency domain to separate high-frequency details
    and low-frequency structures.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, H, W) - real-valued input
        B, C, H, W = x.shape

        # FFT: convert to frequency domain
        xf = torch.fft.rfft2(x, norm="ortho")
        # xf is complex: (B, C, H, W//2+1)
        xf = torch.view_as_real(xf)  # (B, C, H, W//2+1, 2)
        xf = xf.permute(0, 1, 4, 2, 3).contiguous()  # (B, C, 2, H, W//2+1)
        xf = xf.view(B, C * 2, xf.shape[-2], xf.shape[-1])  # (B, 2C, H, W//2+1)

        # Two 1×1 convs in frequency domain
        xf = self.conv1(xf)
        xf = self.relu(xf)
        xf = self.conv2(xf)

        # Convert back to complex
        xf = xf.view(B, C, 2, xf.shape[-2], xf.shape[-1])
        xf = xf.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, H, W//2+1, 2)
        xf = torch.view_as_complex(xf)

        # Inverse FFT
        out = torch.fft.irfft2(xf, s=(H, W), norm="ortho")

        # Residual connection
        out = out + x
        return out

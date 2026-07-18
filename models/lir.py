"""Local Information Regularization (LIR) Module.

Window Neighborhood Attention (WNA) with channel & spatial interaction
to fuse global and local features.
"""

import torch
import torch.nn as nn


class WindowNeighborhoodAttention(nn.Module):
    """Window Neighborhood Attention (WNA).

    Partitions input into windows and performs neighborhood attention
    within each window. Each token attends to its k nearest neighbors
    within the same window.
    """

    def __init__(self, dim, window_size=8, neighborhood_size=7, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.neighborhood_size = neighborhood_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def _window_partition(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.window_size, self.window_size,
                   W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = x.view(-1, self.window_size * self.window_size, C)
        return windows

    def _window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / ((H // self.window_size) *
                                    (W // self.window_size)))
        x = windows.view(B, H // self.window_size, W // self.window_size,
                         self.window_size, self.window_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, -1, H, W)
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        # Pad to multiple of window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))

        _, _, Hp, Wp = x.shape

        # Window partition
        windows = self._window_partition(x)  # (n_windows, ws^2, C)
        n_windows, n_tokens, _ = windows.shape

        # QKV projection
        qkv = self.qkv(windows).reshape(
            n_windows, n_tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)  # (3, n_windows, n_heads, n_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # (n_windows, n_heads, n_tokens, n_tokens)

        # Create neighborhood mask: keep only top-k nearest neighbors per query
        # For efficiency, approximate by taking softmax and zeroing distant entries
        # For each query position, select k nearest tokens
        n_tokens_sqrt = int(n_tokens ** 0.5)
        k = min(self.neighborhood_size, n_tokens)

        # Build neighborhood mask based on spatial positions
        device = attn.device
        coords = torch.stack(torch.meshgrid(
            torch.arange(n_tokens_sqrt, device=device),
            torch.arange(n_tokens_sqrt, device=device),
            indexing="ij",
        )).reshape(2, -1).float()  # (2, n_tokens)

        dist = torch.cdist(coords.t(), coords.t(), p=2)  # (n_tokens, n_tokens)
        _, topk_indices = torch.topk(dist, k=k, dim=-1, largest=False)
        mask = torch.zeros(n_tokens, n_tokens, device=device)
        mask.scatter_(1, topk_indices, 1.0)

        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn)

        out = torch.matmul(attn, v)  # (n_windows, n_heads, n_tokens, head_dim)
        out = out.transpose(1, 2).reshape(n_windows, n_tokens, C)
        out = self.proj(out)

        # Reverse window partition
        out = self._window_reverse(out, Hp, Wp)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out


class ChannelSpatialInteraction(nn.Module):
    """Channel and spatial interaction to fuse global (G) and local (L) features.

    Channel: GAP → Conv1×1+BN+GELU → Conv1×1+BN+GELU → Sigmoid
    Spatial: Conv1×1+BN+GELU → Conv1×1+BN+GELU → Sigmoid → single channel
    """

    def __init__(self, dim):
        super().__init__()
        mid_dim = dim // 4

        self.channel_gap = nn.AdaptiveAvgPool2d(1)
        self.channel_conv1 = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1), nn.BatchNorm2d(mid_dim), nn.GELU()
        )
        self.channel_conv2 = nn.Sequential(
            nn.Conv2d(mid_dim, dim, 1), nn.BatchNorm2d(dim), nn.Sigmoid()
        )

        self.spatial_conv1 = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1), nn.BatchNorm2d(mid_dim), nn.GELU()
        )
        self.spatial_conv2 = nn.Sequential(
            nn.Conv2d(mid_dim, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid()
        )

    def forward(self, global_feat, local_feat):
        # Channel attention
        fused = global_feat + local_feat
        ch_attn = self.channel_gap(fused)
        ch_attn = self.channel_conv1(ch_attn)
        ch_attn = self.channel_conv2(ch_attn)

        # Spatial attention
        sp_attn = self.spatial_conv1(fused)
        sp_attn = self.spatial_conv2(sp_attn)

        # Apply both attentions
        out = fused * ch_attn * sp_attn
        return out


class LIR(nn.Module):
    """Local Information Regularization Module.

    Combines WNA-based local feature extraction with channel-spatial
    interaction to fuse global and local features.
    """

    def __init__(self, dim, window_size=8, neighborhood_size=7, num_heads=4):
        super().__init__()
        self.wna = WindowNeighborhoodAttention(dim, window_size, neighborhood_size,
                                               num_heads)
        self.norm = nn.LayerNorm(dim)
        self.interaction = ChannelSpatialInteraction(dim)

    def forward(self, x, global_feat=None):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # WNA with LayerNorm
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_norm = self.norm(x_flat)
        x_norm = x_norm.transpose(1, 2).view(B, C, H, W)
        local_feat = self.wna(x_norm)  # (B, C, H, W)

        # Fuse with global features if provided, otherwise self-fuse
        if global_feat is None:
            global_feat = x

        out = self.interaction(global_feat, local_feat)
        return out

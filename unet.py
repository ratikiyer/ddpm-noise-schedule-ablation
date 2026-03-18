"""
U-Net score estimator for DDPM.

Architecture follows the original DDPM paper (Ho et al. 2020) with:
  - Sinusoidal timestep embeddings
  - ResNet blocks with GroupNorm
  - Self-attention at specified resolutions
  - Downsampling / upsampling with skip connections
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal embeddings to encode diffusion timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16,),
        dropout: float = 0.1,
        time_emb_dim: int = 512,
        image_size: int = 32,
    ):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        ch = base_channels
        current_res = image_size

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_emb_dim, dropout)]
                if current_res in attention_resolutions:
                    layers.append(SelfAttention(out_ch))
                self.down_blocks.append(nn.ModuleList(layers))
                channels.append(out_ch)
                ch = out_ch
            if level != len(channel_mults) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                channels.append(ch)
                current_res //= 2

        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout)]
                if current_res in attention_resolutions:
                    layers.append(SelfAttention(out_ch))
                if level != 0 and i == num_res_blocks:
                    layers.append(Upsample(out_ch))
                    current_res *= 2
                self.up_blocks.append(nn.ModuleList(layers))
                ch = out_ch

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.input_conv(x)

        # Downsampling
        skips = [h]
        for block in self.down_blocks:
            if isinstance(block[0], Downsample):
                h = block[0](h)
            else:
                for layer in block:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
            skips.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling
        for block in self.up_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, Upsample):
                    h = layer(h)
                else:
                    h = layer(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)

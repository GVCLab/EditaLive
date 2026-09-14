import logging
import os

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..compile_config import is_compile_enabled

COMPILE = is_compile_enabled()


def conditional_compile(func):
    if COMPILE:
        return torch.compile(mode=None, backend="inductor", dynamic=None)(func)
    else:
        return func

__all__ = [
    'WanVAE',
]

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < CACHE_T and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                            dim=2)
                    if cache_x.shape[2] < CACHE_T and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class CausalDepthwiseConv3d(nn.Module):
    """Depthwise-separable replacement for CausalConv3d.

    Keeps the same left-only temporal padding, so the causal / feat_cache contract
    is identical to CausalConv3d -- only the spatial factorisation differs.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1):
        super().__init__()
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        assert len(self.padding) == 3
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size,
                                        stride, 0, dilation, groups=in_channels)
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, 1)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x.to(x.device), x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)


class ResidualBlockFast(nn.Module):
    """ResidualBlock with CausalConv3d swapped for the depthwise-separable form."""

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalDepthwiseConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalDepthwiseConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalDepthwiseConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        res = x
        for layer in self.residual:
            if isinstance(layer, (CausalDepthwiseConv3d, CausalConv3d)) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = res[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None \
                        and feat_cache[idx] != 'Rep':
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1:, :, :].to(res.device), cache_x
                    ], dim=2)
                res = layer(res, feat_cache[idx] if feat_cache[idx] != 'Rep' else None)
                feat_cache[idx] = cache_x.detach() if feat_cache[idx] != 'Rep' else 'Rep'
                feat_idx[0] += 1
            else:
                res = layer(res)
        return h + res


class ResidualBlock2D(nn.Module):
    """Purely spatial residual block: carries no temporal state at all."""

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=True), nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=True), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1))
        self.shortcut = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        assert x.dim() == 4, f"ResidualBlock2D expects 4D input, got {x.dim()}D"
        return self.residual(x) + self.shortcut(x)


class ResampleFast(nn.Module):
    """Upsample block for the lightweight decoder.

    mode='3d' keeps the causal time_conv (2x temporal upsample); mode='2d' is
    spatial only. Same feat_cache / 'Rep' protocol as Resample.
    """

    def __init__(self, in_channels, out_channels, mode='3d'):
        super().__init__()
        assert mode in ('2d', '3d')
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resample = nn.Sequential(
            Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.time_conv = CausalConv3d(
            in_channels, in_channels * 2, kernel_size=(3, 1, 1),
            padding=(1, 0, 0)) if mode == '3d' else None

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if self.mode == '2d':
            assert x.dim() == 4
            return self.resample(x)

        assert x.dim() == 5
        b, c, t, h, w = x.shape

        if self.time_conv is not None and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # first chunk: no temporal upsample yet, just mark the slot
                feat_cache[idx] = 'Rep'
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] != 'Rep':
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1:, :, :].to(x.device), cache_x
                    ], dim=2)
                x = self.time_conv(x, feat_cache[idx] if feat_cache[idx] != 'Rep' else None)
                feat_cache[idx] = cache_x.detach()
                feat_idx[0] += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), dim=3)
                x = x.reshape(b, c, t * 2, h, w)

        t_new = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        return rearrange(x, '(b t) c h w -> b c t h w', t=t_new, b=b)


class AttentionBlock2D(nn.Module):
    """Spatial self-attention over a 4D (b*t, c, h, w) tensor.

    Same parameters and same math as AttentionBlock, but it takes the time-folded
    4D tensor directly and feeds SDPA a 3D (b*t, h*w, c) layout. AttentionBlock
    instead passes a 4D (b*t, 1, h*w, c) view built from a permuted reshape, which
    dispatches to a different SDPA kernel and accumulates in a different order --
    a ~1e-4 float32 difference that the following RMS_norm amplifies. Matching the
    reference layout here keeps the fast decoder bit-identical to upstream
    Flash-VAED. Parameter names match AttentionBlock so checkpoint keys still map.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        assert x.dim() == 4, f"AttentionBlock2D expects 4D input, got {x.dim()}D"
        identity = x
        _, c, h, w = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return self.proj(out) + identity


class FlashDecoder3d(nn.Module):
    """Flash-VAED lightweight decoder for the Wan 2.1 latent space (z_dim=16).

    ~5.9M params vs 73.3M for Decoder3d. Two structural changes, both of which
    leave causality intact: CausalConv3d -> CausalDepthwiseConv3d in the deep
    stages, and the last two upsample stages replaced by purely spatial 2D blocks
    (which cannot see across frames at all). The feat_cache / 'Rep' protocol is
    unchanged, so this streams chunk-by-chunk exactly like Decoder3d.
    """

    def __init__(self, z_dim=16):
        super().__init__()
        self.conv1 = CausalConv3d(z_dim, 384, 3, padding=1)
        self.middle = nn.ModuleList([
            ResidualBlockFast(384, 384),
            AttentionBlock2D(384),
            ResidualBlockFast(384, 384)])

        # index 8 is where the decoder drops to purely spatial blocks
        self.first_2d_idx = 8
        self.upsamples = nn.ModuleList([
            ResidualBlockFast(384, 384),
            ResidualBlockFast(384, 384),
            ResidualBlockFast(384, 384),
            ResampleFast(384, 192, mode='3d'),
            ResidualBlockFast(192, 384),
            ResidualBlockFast(384, 384),
            ResidualBlockFast(384, 384),
            ResampleFast(384, 24, mode='3d'),
            ResidualBlock2D(24, 24),
            ResidualBlock2D(24, 24),
            ResidualBlock2D(24, 24),
            ResampleFast(24, 12, mode='2d'),
            ResidualBlock2D(12, 12),
            ResidualBlock2D(12, 12),
            ResidualBlock2D(12, 12),
        ])

        self.head = nn.ModuleList([
            RMS_norm(12, images=False),
            nn.SiLU(),
            CausalConv3d(12, 3, 3, padding=1)])

    @staticmethod
    def _run_layer(layer, x, feat_cache, feat_idx):
        if isinstance(layer, (ResidualBlockFast, ResampleFast)) and feat_cache is not None:
            return layer(x, feat_cache, feat_idx)
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cached = feat_cache[idx]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and cached is not None and cached != 'Rep':
                cache_x = torch.cat([
                    cached[:, :, -1:, :, :].to(x.device), cache_x
                ], dim=2)
            out = layer(x, cached if cached != 'Rep' else None)
            feat_cache[idx] = cache_x.detach() if cached != 'Rep' else 'Rep'
            feat_idx[0] += 1
            return out
        return layer(x)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, t_in = x.shape[0], x.shape[2]

        x = self._run_layer(self.conv1, x, feat_cache, feat_idx)
        x = self._run_layer(self.middle[0], x, feat_cache, feat_idx)
        # spatial-only attention: fold time into batch, no temporal state
        t_mid = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.middle[1](x)
        x = rearrange(x, '(b t) c h w -> b c t h w', b=b, t=t_mid)
        x = self._run_layer(self.middle[2], x, feat_cache, feat_idx)

        # deep stages run per-frame in 2D; fold time into the batch dim once
        t_before_2d = x.shape[2]
        is_2d = False
        for i, layer in enumerate(self.upsamples):
            if i == self.first_2d_idx and not is_2d and x.dim() == 5:
                t_before_2d = x.shape[2]
                x = rearrange(x, 'b c t h w -> (b t) c h w')
                is_2d = True
            x = layer(x) if is_2d else self._run_layer(layer, x, feat_cache, feat_idx)

        x = rearrange(x, '(b t) c h w -> b c t h w', b=b, t=t_before_2d)
        x = self.head[0](x)
        x = self.head[1](x)
        return self._run_layer(self.head[2], x, feat_cache, feat_idx)


class FlashVAEDecoder(nn.Module):
    """conv2 + FlashDecoder3d, matching the Flash-VAED student checkpoint layout.

    The student ships its own conv2 (the pre-decoder latent projection), so the
    fast path uses that rather than the teacher's.
    """

    def __init__(self, z_dim=16):
        super().__init__()
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = FlashDecoder3d(z_dim=z_dim)


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


def count_conv3d_fast(model):
    """Cache slots for the lightweight decoder: both conv flavours take a cache.

    This over-counts by one: ResidualBlockFast's 1x1x1 shortcut conv is counted but
    never consumes a slot, because forward() calls it as self.shortcut(x) outside the
    feat_cache path (a 1x1x1 conv has zero temporal padding, so it needs no history).
    The trailing slot just stays None. Kept as-is to match upstream Flash-VAED's
    count_conv3d, which allocates the same 21 slots and uses 20.
    """
    count = 0
    for m in model.modules():
        if isinstance(m, (CausalConv3d, CausalDepthwiseConv3d)):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0,
                 build_decoder=True):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # The encoder half is always needed -- Flash-VAED ships no encoder.
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)

        # A run decodes through exactly one decoder, so only that one is built.
        # The original is 73.3M params against Flash-VAED's 5.9M, so building both
        # would waste ~280 MB of fp32 weights. build_decoder=False leaves these
        # None permanently -- the choice is fixed for the lifetime of the object.
        self.conv2 = None
        self.decoder = None
        if build_decoder:
            self.conv2 = CausalConv3d(z_dim, z_dim, 1)
            self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                     attn_scales, self.temperal_upsample, dropout)

        self.first_encode = True
        self.first_decode = True

        # Flash-VAED lightweight decoder (populated by load_fast_decoder).
        self.fast = None
        self.fast_decode = False

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        # cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        # 对encode输入的x，按时间拆分为1、4、4、4....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def stream_encode(self, x, scale):
        # cache
        t = x.shape[2]
        if self.first_encode:
            self.first_encode = False
            self.clear_cache_encode()
            self._enc_conv_idx = [0]
            out = self.encoder(
                x[:, :, :1, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
                )
            if x.shape[2] > 1:
                self._enc_conv_idx = [0]
                out_ = self.encoder(
                    x[:, :, 1:, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                    )
                out = torch.cat([out, out_], 2)
        else:
            out=[]
            for i in range(t//4):
                self._enc_conv_idx = [0]
                out.append(self.encoder(
                    x[:, :, i*4:(i+1)*4, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                    ))
            out = torch.cat(out, 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
        # self.clear_cache()
        return mu

    def load_fast_decoder(self, ckpt_path, device='cuda', dtype=torch.float32):
        """Attach the Flash-VAED lightweight decoder from its student checkpoint.

        Only loads the weights; does not flip fast_decode -- WanVAE.set_fast_decode
        owns that so the active path has a single source of truth.

        The checkpoint holds only conv1/conv2/decoder (156 keys, ~5.9M params);
        conv1 there belongs to the student's (unused) encoder half, so only
        conv2.* and decoder.* are consumed. The encoder side always stays on the
        original weights -- Flash-VAED ships no encoder.
        """
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        wanted = {k: v for k, v in sd.items()
                  if k.startswith('conv2.') or k.startswith('decoder.')}
        if not wanted:
            raise ValueError(f"No conv2./decoder. keys found in {ckpt_path}")

        fast = FlashVAEDecoder(z_dim=self.z_dim)
        missing, unexpected = fast.load_state_dict(wanted, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Flash-VAED checkpoint mismatch.\n  missing: {missing}\n  unexpected: {unexpected}")

        self.fast = fast.to(device=device, dtype=dtype).eval().requires_grad_(False)
        n = sum(p.numel() for p in self.fast.parameters())
        logging.info(f"Flash-VAED decoder loaded from {ckpt_path} ({n / 1e6:.2f}M params)")
        return self

    def _dec_modules(self):
        """(conv2, decoder, cache_slot_count) for whichever decoder is active."""
        if self.fast_decode:
            if self.fast is None:
                raise RuntimeError(
                    "fast_decode=True but no Flash-VAED weights loaded; "
                    "call load_fast_decoder(ckpt) first")
            return self.fast.conv2, self.fast.decoder, count_conv3d_fast(self.fast.decoder)
        if self.decoder is None:
            raise RuntimeError(
                "the original decoder was not built (the VAE was constructed with "
                "fast_decode=True); call WanVAE.set_fast_decode(False) to load it")
        return self.conv2, self.decoder, count_conv3d(self.decoder)

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        conv2, decoder, _ = self._dec_modules()
        x = conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def stream_decode(self, z, scale):
        # z: [b,c,t,h,w]
        t=z.shape[2]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        conv2, decoder, _ = self._dec_modules()
        x = conv2(z)
        if self.first_decode:
            self.first_decode = False
            self.clear_cache_decode()
            self.first_batch = False
            self._conv_idx = [0]
            out = decoder(
                x[:, :, :1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                )
            for i in range(t-1):
                self._conv_idx = [0]
                out_ = decoder(
                    x[:, :, i+1:i+2, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    )
                out = torch.cat([out, out_], 2)
        else:
            out = []
            for i in range(t):
                self._conv_idx = [0]
                out.append(decoder(
                    x[:, :, i:(i+1), :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    ))
            out = torch.cat(out, 2)
        # self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = self._dec_modules()[2]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def clear_cache_decode(self):
        self._conv_num = self._dec_modules()[2]
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
    
    def clear_cache_encode(self):
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
    


def _video_vae(pretrained_path=None, z_dim=None, device='cpu',
               build_decoder=True, **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.

    With ``build_decoder=False`` the original decode half is neither built nor
    loaded: its tensors are dropped straight after the torch.load, so they never
    reach the GPU. Used when the run decodes through Flash-VAED instead.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg, build_decoder=build_decoder)

    # load checkpoint
    logging.info(f'loading {pretrained_path}')
    state_dict = torch.load(pretrained_path, map_location=device)
    if not build_decoder:
        state_dict = {k: v for k, v in state_dict.items()
                      if not (k.startswith('decoder.') or k.startswith('conv2.'))}
    model.load_state_dict(state_dict, assign=True)

    return model


class WanVAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda",
                 fast_decode=False,
                 fast_decoder_pth=None):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # Checked before the model is built: with fast_decode the original decoder
        # is skipped, so a missing Flash-VAED path would leave no decoder at all.
        if fast_decode and not fast_decoder_pth:
            raise ValueError(
                "fast_decode=True requires fast_decoder_pth (the Flash-VAED "
                "checkpoint); there is no default path.")

        # Only the decode half this run will actually use gets built. The choice
        # is permanent: set_fast_decode() can no longer reach the other one.
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
            build_decoder=not fast_decode,
        ).eval().requires_grad_(False).to(device, dtype)

        # Encoding always uses the original weights; only the decode half is
        # swapped. There is no default checkpoint path: it lives wherever the
        # caller says (--fast_decoder_pth).
        self._fast_decoder_pth = fast_decoder_pth
        if fast_decode:
            self.set_fast_decode(True)

    @property
    def fast_decode(self):
        return self.model.fast_decode

    def set_fast_decode(self, enabled):
        """Switch the decode path at inference time; loads weights on first enable.

        Returns the resulting state. Must be called BEFORE warmup_stream(), since
        the two decoders compile to different guards.

        Only the decoder chosen at construction time is available, so this can
        confirm that choice but not reverse it.
        """
        enabled = bool(enabled)
        if enabled:
            if self.model.fast is None:
                if not self._fast_decoder_pth:
                    raise ValueError(
                        "fast_decode=True but no Flash-VAED checkpoint was given; "
                        "pass fast_decoder_pth (--fast_decoder_pth on the CLI).")
                if not os.path.isfile(self._fast_decoder_pth):
                    raise FileNotFoundError(
                        f"fast_decode=True but Flash-VAED checkpoint not found: "
                        f"{self._fast_decoder_pth}")
                self.model.load_fast_decoder(
                    self._fast_decoder_pth, device=self.device, dtype=self.dtype)
        elif self.model.decoder is None:
            raise RuntimeError(
                "cannot switch back to the original decoder: this WanVAE was "
                "built with fast_decode=True, so it was never constructed. "
                "Build a new WanVAE with fast_decode=False instead.")
        self.model.fast_decode = enabled
        return enabled

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]
    
    @conditional_compile
    def stream_encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.stream_encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    @conditional_compile
    def stream_decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.stream_decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]

    @torch.no_grad()
    def warmup_stream(self, pixel_chunks, latent_chunk_size, pixel_hw, latent_hw,
                      pixel_channels=3, num_decode_chunks=None, encode_dtype=None):
        """Trigger torch.compile ahead of time so the real run has no compile stalls.

        Replays the exact shape / branch / cache sequence the real loop produces, so
        every guard (input shape+stride, first_encode/first_decode branch, streaming
        cache growth) is compiled and cached now instead of mid-generation.

        IMPORTANT: the streaming feat_cache keeps evolving for a couple of steps after
        the first chunk (a cache tensor's stride only stabilises on the *second*
        steady-state chunk). So the warmup must run each non-first chunk size at least
        twice, and decode at least 3 times, or a stray recompile will still leak into
        the real run. This method handles that internally.

        With fast_decode the decode side goes through the Flash-VAED decoder, which has
        a different layer count and different cache shapes. Warmup dispatches through
        the same stream_decode(), so it compiles whichever decoder is active -- but that
        means load_fast_decoder() must run BEFORE warmup_stream(), otherwise the guards
        are compiled for the original decoder and the first real chunk recompiles.

        Args:
            pixel_chunks: the distinct per-chunk pixel frame counts, e.g. [9, 12].
            latent_chunk_size: latent frames decoded per chunk (self.chunk_size, e.g. 3).
            pixel_hw: (H, W) of the conditioning pixels fed to stream_encode.
            latent_hw: (lat_h, lat_w) of the latents fed to stream_decode.
            pixel_channels: channel count of encode input (3 for RGB).
            num_decode_chunks: decode calls to replay; >=3 covers first_decode True,
                False, and the cache-stride stabilisation on the following chunk. The
                Flash-VAED decoder has one more cache-shape transition (the 3D->2D
                handoff at upsamples[8]), so it replays 4 by default.
            encode_dtype: dtype of the real stream_encode input (e.g. bfloat16). The
                real loop slices conditioning_pixel_values which is bfloat16, not
                self.dtype, so warmup must compile for that dtype or the encode guard
                fails and recompiles on the first real chunk. Defaults to self.dtype.
        """
        if not COMPILE:
            return
        H, W = pixel_hw
        lat_h, lat_w = latent_hw
        z_dim = self.model.z_dim
        encode_dtype = encode_dtype if encode_dtype is not None else self.dtype
        if num_decode_chunks is None:
            # Flash-VAED needs one extra steady chunk: its cache spans three
            # resolutions (latent, 2x, 8x) and the 3D->2D handoff shifts strides
            # one chunk later than the original decoder does.
            num_decode_chunks = 4 if self.model.fast_decode else 3

        # The real loop runs stream_encode/stream_decode inside an outer
        # torch.autocast(bfloat16) (see EditaLiveStreamingPipeline.generate). That adds
        # AutocastCUDA
        # to every parameter's dispatch key set, which is part of the compile guard.
        # Warming up outside that context compiles a different key set, so the guard
        # fails on the first real chunk and recompiles. Replay under the same autocast.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            # --- encode warmup ---
            # Run the first chunk size once, then every size twice so the cache reaches
            # its steady-state stride (the leak we measured was _enc_feat_map stride on
            # the 2nd consecutive same-size chunk).
            self.model.first_encode = True
            seq = list(pixel_chunks)
            if len(seq) > 1:
                seq = seq + seq[1:]  # replay non-first sizes a second time
            for n in seq:
                dummy = torch.zeros(pixel_channels, n, H, W,
                                    device=self.device, dtype=encode_dtype).contiguous()
                self.stream_encode([dummy])

            # --- decode warmup: first_decode True, False, then more steady chunks ---
            self.model.first_decode = True
            for _ in range(max(3, num_decode_chunks)):
                dummy = torch.zeros(z_dim, latent_chunk_size, lat_h, lat_w,
                                    device=self.device, dtype=torch.float32).contiguous()
                self.stream_decode([dummy])

        # restore streaming state so the real run starts clean
        self.model.first_encode = True
        self.model.first_decode = True
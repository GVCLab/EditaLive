import logging as _logging
import math
import os
from typing import List

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from ..compile_config import is_compile_enabled

from .modules.attention_utils import flash_attention
from .modules.vsa_attention import video_sparse_attn_cache
from .modules.face_adapter import FaceAdapter, FaceEncoder
from .modules.motion_encoder import Generator

torch._dynamo.config.cache_size_limit = 128

_logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(_logging.ERROR)
_logging.getLogger("torch.fx.experimental.recording").setLevel(_logging.CRITICAL)
_logging.getLogger("torch._dynamo.exc").setLevel(_logging.ERROR)

COMPILE = is_compile_enabled()
COMPILE_MODE = os.getenv("COMPILE_MODE", "").strip() or None


def conditional_compile(func):
    if COMPILE:
        return torch.compile(mode=COMPILE_MODE, backend="inductor", dynamic=None)(func)
    return func


# --- Wan layers this DiT is built from, weight-compatible with Wan-Animate --- #

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class MLPProj(torch.nn.Module):
    """Projects the CLIP image embedding into the DiT's context width."""

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


@amp.autocast(enabled=False)
@conditional_compile
def rope_apply(x, grid_sizes, freqs, shift=0, cache_len=0):
    """Rotary embedding with an absolute-position offset for the rolling cache.

    ``shift`` moves the query chunk to its absolute frame position in the clip;
    ``cache_len`` extends the frame count so the cached keys keep the positions
    they were given when they were first written.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        f += cache_len
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][shift:f + shift].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class EditaLiveHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        """
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C] or [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            if e.dim() > 2:
                e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
                e = [u.squeeze(2) for u in e]
            else:
                e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class EditaLiveStreamingSelfAttention(nn.Module):
    """Self-attention over a latent chunk plus a rolling KV cache.

    ``kv_cache`` is a plain dict (``k``, ``v``, ``shift``) owned by the pipeline,
    not module state, so the pipeline can trim it between chunks without touching
    the model. ``shift`` counts the latent frames already in the cache and doubles
    as the rope offset for the incoming chunk.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 vsa_sparsity=0.0,
                 vsa_tile_size=(4, 4, 4),
                 vsa_backend="auto",
                 vsa_sink_frames=4):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        # Video Sparse Attention (VSA) config for the causal rolling-KV-cache path.
        # vsa_sparsity == 0.0 -> dense attention.
        self.vsa_sparsity = vsa_sparsity
        self.vsa_tile_size = tuple(vsa_tile_size)
        self.vsa_backend = vsa_backend
        self.vsa_sink_frames = vsa_sink_frames

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, kv_cache, is_update, is_causal):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            x = flash_attention(
                q=rope_apply(q, grid_sizes, freqs),
                k=rope_apply(k, grid_sizes, freqs),
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size)
            return self.o(x.flatten(2))

        k_cache = torch.cat([kv_cache["k"], k], dim=1)
        v_cache = torch.cat([kv_cache["v"], v], dim=1)
        shift = kv_cache["shift"]

        if is_causal:
            q_roped = rope_apply(q, grid_sizes, freqs, shift=shift)
            k_roped = rope_apply(k_cache, grid_sizes, freqs, cache_len=shift)
            v_roped = v_cache

            sparsity = getattr(self, "vsa_sparsity", 0.0)
            if sparsity > 0.0 and shift > 1:
                # Block-sparse attention over the rolling KV cache. The query chunk
                # grid is grid_sizes[0]=(F,H,W); the cache spans `shift` already
                # cached frames plus the current F, so its effective grid is
                # (shift+F, H, W) -- matching the cache_len=shift rope above.
                Fq, Hq, Wq = (int(u) for u in grid_sizes[0].tolist())
                grid_q = (Fq, Hq, Wq)
                grid_kv = (shift + Fq, Hq, Wq)
                # Attention-sink frames are the first frames of the contiguous
                # cache (the pipeline's update_kv() always keeps the head of the
                # cache). Convert sink frames -> tile blocks.
                tile = getattr(self, "vsa_tile_size", (4, 4, 4))
                sink_frames = min(getattr(self, "vsa_sink_frames", 4), 1)
                n_h = math.ceil(Hq / tile[1])
                n_w = math.ceil(Wq / tile[2])
                sink_t_tiles = math.ceil(sink_frames / tile[0]) if sink_frames > 0 else 0
                sink_blocks = sink_t_tiles * n_h * n_w
                x = video_sparse_attn_cache(
                    q_roped, k_roped, v_roped,
                    grid_q=grid_q, grid_kv=grid_kv,
                    sparsity=sparsity, tile_size=tile,
                    sink_blocks=sink_blocks,
                    backend=getattr(self, "vsa_backend", "auto"))
            else:
                x = flash_attention(
                    q=q_roped,
                    k=k_roped,
                    v=v_roped,
                    window_size=self.window_size)
        else:
            x = flash_attention(
                q=rope_apply(q, grid_sizes, freqs),
                k=rope_apply(k, grid_sizes, freqs),
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size)

        # Only the LAST denoising step of a chunk commits to the cache: earlier
        # steps read the cache but their (noisier) keys must not persist.
        if is_update:
            kv_cache["k"] = k_cache
            kv_cache["v"] = v_cache
            kv_cache["shift"] = shift + grid_sizes[0].tolist()[0]

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class EditaLiveStreamingCrossAttention(nn.Module):
    """Cross-attention to text (+ CLIP image) with a per-clip key/value cache.

    The text and reference-image conditioning are constant for a whole clip, so
    their projections are computed once on the first chunk and reused; subsequent
    forwards pass ``context=None``.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 use_img_emb=True):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.use_img_emb = use_img_emb

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        if use_img_emb:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, crossattn_cache=None):
        """
        x:              [B, L1, C].
        context:        [B, L2, C] -- the first 257 tokens are the CLIP image
                        embedding when ``use_img_emb``; None once cached.
        context_lens:   [B].
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                if self.use_img_emb:
                    context_img = context[:, :257]
                    context = context[:, 257:]
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]

            if self.use_img_emb:
                if crossattn_cache["k_img"] is None:
                    k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
                    v_img = self.v_img(context_img).view(b, -1, n, d)
                    crossattn_cache["k_img"] = k_img
                    crossattn_cache["v_img"] = v_img
                else:
                    k_img = crossattn_cache["k_img"]
                    v_img = crossattn_cache["v_img"]
        else:
            if self.use_img_emb:
                context_img = context[:, :257]
                context = context[:, 257:]

            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

            if self.use_img_emb:
                k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
                v_img = self.v_img(context_img).view(b, -1, n, d)

        if self.use_img_emb:
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)

        if self.use_img_emb:
            img_x = img_x.flatten(2)
            x = x + img_x

        x = self.o(x)
        return x


class EditaLiveStreamingAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 use_img_emb=True,
                 vsa_sparsity=0.0,
                 vsa_tile_size=(4, 4, 4),
                 vsa_backend="auto",
                 vsa_sink_frames=4):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = EditaLiveStreamingSelfAttention(
            dim, num_heads, window_size, qk_norm, eps,
            vsa_sparsity=vsa_sparsity, vsa_tile_size=vsa_tile_size,
            vsa_backend=vsa_backend, vsa_sink_frames=vsa_sink_frames)

        self.norm3 = WanLayerNorm(
            dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.cross_attn = EditaLiveStreamingCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, eps, use_img_emb=use_img_emb)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        kv_cache=None,
        crossattn_cache=None,
        is_update=False,
        is_causal=True,
    ):
        """
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C] or [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            if e.dim() > 3:
                e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
                e = [u.squeeze(2) for u in e]
            else:
                e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0],
            seq_lens, grid_sizes, freqs, kv_cache, is_update, is_causal)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class EditaLiveTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """EditaLive's DiT: chunk-by-chunk streaming inference over a rolling KV cache.

    Structurally a Wan-Animate DiT, and weight-compatible with it: same
    parameter names, same config keys, so it reads the stock Wan2.2-Animate-14B
    checkpoint and the LoRAs trained on top of it unchanged.
    """

    _no_split_modules = ['EditaLiveStreamingAttentionBlock']

    @register_to_config
    def __init__(self,
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=36,
                 dim=5120,
                 ffn_dim=13824,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=40,
                 num_layers=40,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 motion_encoder_dim=512,
                 use_context_parallel=False,
                 use_img_emb=True,
                 vsa_sparsity=0.0,
                 vsa_tile_size=(4, 4, 4),
                 vsa_backend="auto",
                 vsa_sink_frames=4):
        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.motion_encoder_dim = motion_encoder_dim
        # use_context_parallel is accepted because the Wan-Animate config.json
        # carries it, but this streaming pipeline is single-GPU and ignores it.
        self.use_img_emb = use_img_emb
        self.vsa_sparsity = vsa_sparsity
        self.vsa_tile_size = tuple(vsa_tile_size)
        self.vsa_backend = vsa_backend
        self.vsa_sink_frames = vsa_sink_frames

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.pose_patch_embedding = nn.Conv3d(
            16, dim, kernel_size=patch_size, stride=patch_size)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            EditaLiveStreamingAttentionBlock(
                dim, ffn_dim, num_heads, window_size, qk_norm,
                cross_attn_norm, eps, use_img_emb,
                vsa_sparsity=vsa_sparsity, vsa_tile_size=vsa_tile_size,
                vsa_backend=vsa_backend, vsa_sink_frames=vsa_sink_frames)
            for _ in range(num_layers)
        ])

        # head
        self.head = EditaLiveHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)
        self.d = d

        self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.motion_encoder = Generator(size=512, style_dim=512, motion_dim=20)
        self.face_adapter = FaceAdapter(
            heads_num=self.num_heads,
            hidden_dim=self.dim,
            num_adapter_layers=self.num_layers // 5,
        )

        self.face_encoder = FaceEncoder(
            in_dim=motion_encoder_dim,
            hidden_dim=self.dim,
            num_heads=4,
        )

        self.num_frame_per_block = 3

    def set_vsa(self, sparsity, backend="auto", tile_size=(1, 8, 8), sink_frames=1):
        """Push Video Sparse Attention settings onto every self-attn block.

        Applied after loading so the choice is a property of the run rather than
        of the checkpoint; ``sparsity == 0.0`` keeps the dense path.
        """
        self.vsa_sparsity = sparsity
        self.vsa_backend = backend
        self.vsa_tile_size = tuple(tile_size)
        self.vsa_sink_frames = sink_frames
        for block in self.blocks:
            block.self_attn.vsa_sparsity = sparsity
            block.self_attn.vsa_backend = backend
            block.self_attn.vsa_tile_size = tuple(tile_size)
            block.self_attn.vsa_sink_frames = sink_frames

    def get_motion_vec(self, face_pixel_values):
        b, c, T, h, w = face_pixel_values.shape
        face_pixel_values = rearrange(face_pixel_values, "b c t h w -> (b t) c h w")

        encode_bs = 8
        face_pixel_values_tmp = []
        for i in range(math.ceil(face_pixel_values.shape[0] / encode_bs)):
            face_pixel_values_tmp.append(
                self.motion_encoder.get_motion(face_pixel_values[i * encode_bs:(i + 1) * encode_bs]))

        motion_vec = torch.cat(face_pixel_values_tmp).to(face_pixel_values.dtype)
        motion_vec = rearrange(motion_vec, "(b t) c -> b t c", t=T)
        motion_vec = self.face_encoder(motion_vec)
        return motion_vec

    def after_patch_embedding(self, x: List[torch.Tensor], pose_latents):
        if pose_latents is None:
            return x
        pose_latents = [self.pose_patch_embedding(u.unsqueeze(0)) for u in pose_latents]
        for x_, pose_latents_ in zip(x, pose_latents):
            f = pose_latents_.shape[2]
            x_[:, :, -f:] += pose_latents_
        return x

    def after_transformer_block(self, block_idx, x, motion_vec, motion_masks=None):
        if block_idx % 5 == 0:
            residual_out = self.face_adapter.fuser_blocks[block_idx // 5](
                x, motion_vec, motion_masks)
            x = residual_out + x
        return x

    def _embed(self, x, t, y, pose_latents, seq_len):
        """Shared patch/time embedding for forward() and forward_ref()."""
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        x = self.after_patch_embedding(x, pose_latents)

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            if t.dim() != 1:
                if t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1].unsqueeze(1)
                    padding = last_elements.repeat(1, pad_size)
                    t = torch.cat([t, padding], dim=1)
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, ft)
                    .unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        return x, e, e0, grid_sizes, seq_lens

    def _embed_context(self, context, clip_fea):
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        if self.use_img_emb:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        return context

    @conditional_compile
    def forward(
        self,
        x,
        t,
        clip_fea,
        context,
        seq_len,
        kv_cache=None,
        crossattn_cache=None,
        is_update=False,
        y=None,
        pose_latents=None,
        motion_vec=None,
        is_causal=True,
    ):
        """Denoise one latent chunk against the rolling cache.

        Args:
            x: list of ``[C, F_chunk, H, W]`` latents (one per sample).
            t: ``[B]`` timestep.
            kv_cache / crossattn_cache: per-layer dicts owned by the pipeline.
            is_update: commit this forward's keys/values to ``kv_cache``. Set on
                the final denoising step of a chunk only.
            y: conditioning latents concatenated on the channel axis.
            pose_latents / motion_vec: source-video pose and face features for the chunk.
        """
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x, e, e0, grid_sizes, seq_lens = self._embed(x, t, y, pose_latents, seq_len)

        # context: projected once per clip, then served from crossattn_cache
        context_lens = None
        if crossattn_cache is not None and crossattn_cache[0]["is_init"]:
            context = None
        else:
            context = self._embed_context(context, clip_fea)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            is_update=is_update,
            is_causal=is_causal)

        for idx, block in enumerate(self.blocks):
            if kv_cache is not None:
                kwargs.update({"kv_cache": kv_cache[idx]})
            if crossattn_cache is not None:
                kwargs.update({"crossattn_cache": crossattn_cache[idx]})
            x = block(x, **kwargs)
            x = self.after_transformer_block(idx, x, motion_vec)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def forward_ref(
        self,
        x,
        t,
        y,
        clip_fea,
        context,
        seq_len,
        kv_cache,
        crossattn_cache,
        is_update,
    ):
        """Prime the KV cache with the clean reference frame.

        Runs the reference latent through every block at t=0 with
        ``is_update=True`` so its keys/values sit at the head of the cache and
        stay there (they are the attention sink every later chunk can see).
        Produces no output -- only the cache side effect.
        """
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x, e, e0, grid_sizes, seq_lens = self._embed(x, t, y, None, seq_len)
        # The reference frame carries no driving motion, so the face adapter is
        # fed zeros. Shape/dtype follow the model rather than being hard-coded,
        # so this also works at other widths and in fp32 (e.g. parity tests).
        motion_vec = torch.zeros(
            1, 1, 5, self.dim,
            dtype=self.patch_embedding.weight.dtype, device=device)

        # context
        context_lens = None
        context = self._embed_context(context, clip_fea)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            is_update=is_update)

        for idx, block in enumerate(self.blocks):
            kwargs.update({
                "kv_cache": kv_cache[idx],
                "crossattn_cache": crossattn_cache[idx],
            })
            x = block(x, **kwargs)
            x = self.after_transformer_block(idx, x, motion_vec)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

import functools
import logging
import math
import os

import torch

try:
    from flash_attn import flash_attn_varlen_func

    _FLASH_AVAILABLE = True
except Exception:  # pragma: no cover - flash-attn missing / CPU-only env
    flash_attn_varlen_func = None
    _FLASH_AVAILABLE = False


__all__ = ["video_sparse_attn_cache", "VSA_TILE_SIZE"]

# The official block-sparse kernel operates on 64-token blocks.
VSA_TILE_SIZE = (4, 4, 4)


@functools.lru_cache(maxsize=1)
def _load_fastvideo_kernel():
    """Return FastVideo's public sparse-only operator when it is importable."""
    try:
        from fastvideo_kernel import block_sparse_attn_from_indices

        return block_sparse_attn_from_indices
    except Exception:
        # Cached, so this warns once per process instead of silently running
        # the slower FlashAttention varlen fallback on every call.
        logging.warning(
            "VSA kernel backend unavailable (fastvideo_kernel."
            "block_sparse_attn_from_indices could not be imported; it needs "
            "fastvideo-kernel>=0.3.0). Falling back to the slower FlashAttention "
            "varlen path.")
        return None


@functools.lru_cache(maxsize=32)
def _num_tiles(grid: tuple, tile: tuple) -> tuple:
    return tuple(math.ceil(size / tile_size) for size, tile_size in zip(grid, tile))


@functools.lru_cache(maxsize=32)
def get_tile_partition_indices(
    grid: tuple,
    tile: tuple,
    device: torch.device,
) -> torch.Tensor:
    """Map flattened ``(F, H, W)`` tokens to tile-contiguous order."""
    frames, height, width = grid
    tile_f, tile_h, tile_w = tile
    indices = torch.arange(
        frames * height * width,
        device=device,
        dtype=torch.long,
    ).reshape(frames, height, width)

    partitions = []
    for frame_idx in range(math.ceil(frames / tile_f)):
        for height_idx in range(math.ceil(height / tile_h)):
            for width_idx in range(math.ceil(width / tile_w)):
                partitions.append(
                    indices[
                        frame_idx * tile_f:min((frame_idx + 1) * tile_f, frames),
                        height_idx * tile_h:min((height_idx + 1) * tile_h, height),
                        width_idx * tile_w:min((width_idx + 1) * tile_w, width),
                    ].flatten()
                )
    return torch.cat(partitions, dim=0)


@functools.lru_cache(maxsize=32)
def construct_variable_block_sizes(
    grid: tuple,
    tile: tuple,
    device: torch.device,
) -> torch.Tensor:
    """Return the valid-token count for every padded spatio-temporal tile."""
    frames, height, width = grid
    tile_f, tile_h, tile_w = tile
    count_f, count_h, count_w = _num_tiles(grid, tile)

    def _axis_sizes(axis_size: int, tile_size: int, tile_count: int) -> torch.Tensor:
        sizes = torch.full(
            (tile_count,),
            tile_size,
            dtype=torch.long,
            device=device,
        )
        sizes[-1] = axis_size - (tile_count - 1) * tile_size
        return sizes

    frame_sizes = _axis_sizes(frames, tile_f, count_f)
    height_sizes = _axis_sizes(height, tile_h, count_h)
    width_sizes = _axis_sizes(width, tile_w, count_w)
    return (
        frame_sizes[:, None, None]
        * height_sizes[None, :, None]
        * width_sizes[None, None, :]
    ).reshape(-1)


@functools.lru_cache(maxsize=32)
def get_non_pad_index(
    grid: tuple,
    tile: tuple,
    device: torch.device,
) -> torch.Tensor:
    """Return valid-token positions in the padded tile-contiguous layout."""
    variable_block_sizes = construct_variable_block_sizes(grid, tile, device)
    block_elements = math.prod(tile)
    num_blocks = variable_block_sizes.shape[0]
    block_starts = torch.arange(num_blocks, device=device) * block_elements
    padded_indices = (
        block_starts[:, None]
        + torch.arange(block_elements, device=device)[None, :]
    )
    valid_mask = (
        torch.arange(block_elements, device=device)[None, :]
        < variable_block_sizes[:, None]
    )
    return padded_indices[valid_mask]


def _tile(x: torch.Tensor, grid: tuple, tile: tuple) -> torch.Tensor:
    """Convert ``[B, H, S, D]`` to a zero-padded tile-contiguous layout."""
    batch, heads, _, dim = x.shape
    num_blocks = math.prod(_num_tiles(grid, tile))
    padded_length = num_blocks * math.prod(tile)
    partition_indices = get_tile_partition_indices(grid, tile, x.device)
    non_pad_indices = get_non_pad_index(grid, tile, x.device)

    tiled = x.new_zeros((batch, heads, padded_length, dim))
    tiled[:, :, non_pad_indices] = x[:, :, partition_indices]
    return tiled


def _untile(x: torch.Tensor, grid: tuple, tile: tuple) -> torch.Tensor:
    """Convert a padded tile-contiguous tensor back to ``[B, H, S, D]``."""
    partition_indices = get_tile_partition_indices(grid, tile, x.device)
    non_pad_indices = get_non_pad_index(grid, tile, x.device)
    batch, heads, _, dim = x.shape
    output = x.new_zeros((batch, heads, partition_indices.shape[0], dim))
    output[:, :, partition_indices] = x[:, :, non_pad_indices]
    return output


def _select_topk_blocks(
    scores: torch.Tensor,
    sparsity: float,
    sink_blocks: int,
) -> torch.Tensor:
    """Select KV blocks using EditaLive's existing attention-sink bias."""
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    if sink_blocks < 0:
        raise ValueError(f"sink_blocks must be non-negative, got {sink_blocks}")

    key_blocks = scores.shape[-1]
    topk = max(1, math.ceil((1.0 - sparsity) * key_blocks))
    topk = min(topk, key_blocks)

    if sink_blocks:
        sink = min(sink_blocks, key_blocks)
        sink_bias = torch.zeros(
            key_blocks,
            device=scores.device,
            dtype=scores.dtype,
        )
        sink_bias[sink:sink * 2] = float("inf")
        scores = scores + sink_bias.view(1, 1, 1, key_blocks)

    return torch.topk(scores, topk, dim=-1).indices


def _sparse_branch_official(
    q_tiled: torch.Tensor,
    k_tiled: torch.Tensor,
    v_tiled: torch.Tensor,
    topk_indices: torch.Tensor,
    key_block_sizes: torch.Tensor,
) -> torch.Tensor | None:
    """Run FastVideo's official sparse-only operator on preselected blocks."""
    operator = _load_fastvideo_kernel()
    if operator is None:
        return None

    batch, heads, query_blocks = topk_indices.shape[:3]
    topk = topk_indices.shape[-1]
    q2k_indices = topk_indices.to(torch.int32).contiguous()
    q2k_counts = torch.full(
        (batch, heads, query_blocks),
        topk,
        dtype=torch.int32,
        device=q_tiled.device,
    )
    output, _ = operator(
        q_tiled.contiguous(),
        k_tiled.contiguous(),
        v_tiled.contiguous(),
        q2k_indices,
        q2k_counts,
        key_block_sizes.to(torch.int32).contiguous(),
    )
    return output


def _sparse_branch_flash(
    q_blocks: torch.Tensor,
    k_blocks: torch.Tensor,
    v_blocks: torch.Tensor,
    topk_indices: torch.Tensor,
    key_block_sizes: torch.Tensor,
    query_block_sizes: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the padding-free FlashAttention varlen fallback."""
    batch, heads, num_query_blocks, block_elements, dim = q_blocks.shape
    num_key_blocks = k_blocks.shape[2]
    topk = topk_indices.shape[-1]
    device = q_blocks.device
    batch_heads = batch * heads
    num_sequences = batch_heads * num_query_blocks

    token_positions = torch.arange(block_elements, device=device)
    query_valid = (
        token_positions.view(1, block_elements)
        < query_block_sizes.view(num_query_blocks, 1)
    )
    query_lengths = query_block_sizes.to(torch.int32).repeat(batch_heads)
    query_mask = query_valid.view(1, num_query_blocks, block_elements).expand(
        batch_heads,
        num_query_blocks,
        block_elements,
    ).reshape(batch_heads, num_query_blocks * block_elements)
    query_source = q_blocks.reshape(
        batch_heads,
        num_query_blocks * block_elements,
        dim,
    )
    query_packed = query_source[query_mask]

    key_flat = k_blocks.reshape(batch_heads * num_key_blocks * block_elements, dim)
    value_flat = v_blocks.reshape(batch_heads * num_key_blocks * block_elements, dim)
    key_valid_by_block = (
        token_positions.view(1, block_elements)
        < key_block_sizes.view(num_key_blocks, 1)
    )
    selected_valid = key_valid_by_block[topk_indices].reshape(
        num_sequences,
        topk * block_elements,
    )
    key_lengths = selected_valid.sum(dim=1).to(torch.int32)

    batch_head_offsets = (
        torch.arange(batch_heads, device=device) * num_key_blocks * block_elements
    ).view(batch_heads, 1, 1, 1)
    block_offsets = (
        topk_indices.reshape(batch_heads, num_query_blocks, topk, 1).to(torch.int64)
        * block_elements
    )
    global_indices = (
        batch_head_offsets
        + block_offsets
        + token_positions.view(1, 1, 1, block_elements)
    ).reshape(num_sequences, topk * block_elements)
    packed_indices = global_indices[selected_valid]
    key_packed = key_flat.index_select(0, packed_indices)
    value_packed = value_flat.index_select(0, packed_indices)

    cumulative_query_lengths = torch.zeros(
        num_sequences + 1,
        dtype=torch.int32,
        device=device,
    )
    cumulative_query_lengths[1:] = torch.cumsum(query_lengths, dim=0)
    cumulative_key_lengths = torch.zeros(
        num_sequences + 1,
        dtype=torch.int32,
        device=device,
    )
    cumulative_key_lengths[1:] = torch.cumsum(key_lengths, dim=0)

    packed_output = flash_attn_varlen_func(
        query_packed.unsqueeze(1),
        key_packed.unsqueeze(1),
        value_packed.unsqueeze(1),
        cu_seqlens_q=cumulative_query_lengths,
        cu_seqlens_k=cumulative_key_lengths,
        max_seqlen_q=int(query_lengths.max().item()),
        max_seqlen_k=int(key_lengths.max().item()),
        softmax_scale=softmax_scale,
        causal=False,
    ).squeeze(1)

    output = query_source.new_zeros(
        batch_heads,
        num_query_blocks * block_elements,
        dim,
    )
    output[query_mask] = packed_output.to(output.dtype)
    return output.reshape(
        batch,
        heads,
        num_query_blocks * block_elements,
        dim,
    )


def _sparse_branch_sdpa(
    q_blocks: torch.Tensor,
    k_tiled: torch.Tensor,
    v_tiled: torch.Tensor,
    topk_indices: torch.Tensor,
    key_block_sizes: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the portable gather plus PyTorch SDPA fallback."""
    batch, heads, query_blocks, block_elements, dim = q_blocks.shape
    key_blocks = k_tiled.shape[2] // block_elements
    topk = topk_indices.shape[-1]

    key_by_block = k_tiled.view(batch, heads, key_blocks, block_elements, dim)
    value_by_block = v_tiled.view(batch, heads, key_blocks, block_elements, dim)
    gather_indices = topk_indices[..., None, None].expand(
        batch,
        heads,
        query_blocks,
        topk,
        block_elements,
        dim,
    )
    selected_keys = torch.gather(
        key_by_block.unsqueeze(2).expand(
            batch,
            heads,
            query_blocks,
            key_blocks,
            block_elements,
            dim,
        ),
        3,
        gather_indices,
    ).reshape(batch, heads, query_blocks, topk * block_elements, dim)
    selected_values = torch.gather(
        value_by_block.unsqueeze(2).expand(
            batch,
            heads,
            query_blocks,
            key_blocks,
            block_elements,
            dim,
        ),
        3,
        gather_indices,
    ).reshape(batch, heads, query_blocks, topk * block_elements, dim)

    token_positions = torch.arange(block_elements, device=q_blocks.device)
    valid_by_block = (
        token_positions.view(1, block_elements)
        < key_block_sizes.view(key_blocks, 1)
    )
    selected_valid = valid_by_block[topk_indices].reshape(
        batch * heads * query_blocks,
        1,
        1,
        topk * block_elements,
    )
    selected_valid = selected_valid.expand(
        batch * heads * query_blocks,
        1,
        block_elements,
        topk * block_elements,
    )

    queries = q_blocks.reshape(
        batch * heads * query_blocks,
        block_elements,
        dim,
    )
    keys = selected_keys.reshape(
        batch * heads * query_blocks,
        topk * block_elements,
        dim,
    )
    values = selected_values.reshape(
        batch * heads * query_blocks,
        topk * block_elements,
        dim,
    )
    output = torch.nn.functional.scaled_dot_product_attention(
        queries.unsqueeze(1),
        keys.unsqueeze(1),
        values.unsqueeze(1),
        attn_mask=selected_valid,
        scale=softmax_scale,
    )
    return output.squeeze(1).reshape(
        batch,
        heads,
        query_blocks * block_elements,
        dim,
    )


def video_sparse_attn_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grid_q: tuple,
    grid_kv: tuple,
    sparsity: float,
    tile_size: tuple = VSA_TILE_SIZE,
    sink_blocks: int = 0,
    softmax_scale: float | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Apply sparse attention from a query chunk to a rolling KV cache."""
    
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must all have shape [B, S, H, D]")
    if k.shape != v.shape:
        raise ValueError(f"k and v shapes must match, got {k.shape} and {v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError(
            "q and k must have matching batch, head, and head-dimension sizes"
        )
    if len(grid_q) != 3 or len(grid_kv) != 3 or len(tile_size) != 3:
        raise ValueError("grid_q, grid_kv, and tile_size must be 3-tuples")
    if any(size <= 0 for size in (*grid_q, *grid_kv, *tile_size)):
        raise ValueError("grid and tile dimensions must all be positive")

    batch, query_length, heads, dim = q.shape
    key_length = k.shape[1]
    if math.prod(grid_q) != query_length:
        raise ValueError(f"grid_q {grid_q} does not match query length {query_length}")
    if math.prod(grid_kv) != key_length:
        raise ValueError(f"grid_kv {grid_kv} does not match KV length {key_length}")

    default_scale = dim ** -0.5
    if softmax_scale is None:
        softmax_scale = default_scale
    block_elements = math.prod(tile_size)

    input_dtype = q.dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    if input_dtype not in half_dtypes:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

    q_head_major = q.transpose(1, 2)
    k_head_major = k.transpose(1, 2)
    v_head_major = v.transpose(1, 2)
    compute_dtype = q_head_major.dtype

    q_tiled = _tile(q_head_major, grid_q, tile_size)
    k_tiled = _tile(k_head_major, grid_kv, tile_size)
    v_tiled = _tile(v_head_major, grid_kv, tile_size)
    query_blocks = q_tiled.shape[2] // block_elements
    key_blocks = k_tiled.shape[2] // block_elements

    query_block_sizes = construct_variable_block_sizes(
        grid_q,
        tile_size,
        q.device,
    ).clamp(min=1)
    key_block_sizes = construct_variable_block_sizes(
        grid_kv,
        tile_size,
        k.device,
    ).clamp(min=1)

    q_by_block = q_tiled.view(
        batch,
        heads,
        query_blocks,
        block_elements,
        dim,
    )
    k_by_block = k_tiled.view(
        batch,
        heads,
        key_blocks,
        block_elements,
        dim,
    )
    v_by_block = v_tiled.view(
        batch,
        heads,
        key_blocks,
        block_elements,
        dim,
    )
    query_means = (
        q_by_block.float().sum(dim=3)
        / query_block_sizes.view(1, 1, -1, 1)
    ).to(compute_dtype)
    key_means = (
        k_by_block.float().sum(dim=3)
        / key_block_sizes.view(1, 1, -1, 1)
    ).to(compute_dtype)
    scores = (
        torch.matmul(query_means.float(), key_means.float().transpose(-2, -1))
        * softmax_scale
    )
    topk_indices = _select_topk_blocks(scores, sparsity, sink_blocks)

    backend_request = os.environ.get("VSA_BACKEND", backend).lower()
    if backend_request not in {"auto", "kernel", "sdpa"}:
        raise ValueError(
            f"unsupported VSA backend {backend_request!r}; "
            "expected 'auto', 'kernel', or 'sdpa'"
        )

    on_cuda_half = q.is_cuda and q.dtype in half_dtypes
    official_scale_supported = math.isclose(
        softmax_scale,
        default_scale,
        rel_tol=1e-6,
        abs_tol=1e-12,
    )
    output_tiled = None
    if (
        backend_request in {"auto", "kernel"}
        and on_cuda_half
        and block_elements == 64
        and official_scale_supported
    ):
        output_tiled = _sparse_branch_official(
            q_tiled,
            k_tiled,
            v_tiled,
            topk_indices,
            key_block_sizes,
        )

    use_flash = (
        output_tiled is None
        and backend_request != "sdpa"
        and on_cuda_half
        and _FLASH_AVAILABLE
    )
    if use_flash:
        output_tiled = _sparse_branch_flash(
            q_by_block,
            k_by_block,
            v_by_block,
            topk_indices,
            key_block_sizes,
            query_block_sizes,
            softmax_scale,
        )

    if output_tiled is None:
        output_tiled = _sparse_branch_sdpa(
            q_by_block,
            k_tiled,
            v_tiled,
            topk_indices,
            key_block_sizes,
            softmax_scale,
        )

    output = _untile(output_tiled, grid_q, tile_size)
    return output.transpose(1, 2).contiguous().to(input_dtype)

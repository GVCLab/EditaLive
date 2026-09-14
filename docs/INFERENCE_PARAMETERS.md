# EditaLive — Inference Parameters

This guide documents the command-line arguments accepted by
`edit_streaming.py`. Run `python edit_streaming.py --help` to view the same
options in the terminal.

Boolean-valued arguments such as `--fast_decode` accept `True` or `False`.
Action flags such as `--fp8`, `--t5_cpu`, and `--force_preprocess` are enabled
by including the flag without a value.

## Input selection

Choose exactly one of `--video`, `--folder`, or `--input_json`.

| Argument | Default | Description |
|---|---:|---|
| `--video` | `None` | Raw source video for one inference run. |
| `--ref_image` | First video frame | Optional character reference image used with `--video`. |
| `--folder` | `None` | Preprocessed root containing `src_pose.mp4`, `src_face.mp4`, and `src_ref.png`. |
| `--input_json` | `None` | JSON task list for processing multiple inputs with one model load. See [`demo/edit_causal.json`](../demo/edit_causal.json). |
| `--prompt` | Config fallback | Edit instruction for a single input, and the fallback prompt for JSON entries without one. |

`--ref_image` is only valid with `--video`. Each JSON entry must contain either
`video` (optionally with `ref_image`) or `folder`, and may also specify `id` and
`prompt`.

One run produces one output size, so every input in it must share the same
orientation: all portrait or all landscape references. A batch that mixes the
two is rejected before the model loads — run each orientation separately.

## Checkpoints and output

| Argument | Default | Description |
|---|---:|---|
| `--ckpt_dir` | `./weights/Wan-Animate` | Wan-Animate checkpoint directory. |
| `--save_dir` | `./outputs_streaming` | Generated-video directory. The default preprocessing cache is created below it. |
| `--fps` | `16` | Output video frame rate. Must be positive. |
| `--keep_audio` | `True` | Copy audio from the raw source video to the result. It is unavailable for a task that only supplies `--folder`; set it to `False` to disable audio. |

When preserving audio, keep `--preprocess_fps` equal to `--fps`; otherwise the
generated video duration and source audio can drift.

## Preprocessing

These options apply when the input uses `--video`. A `--folder` input skips
preprocessing.

| Argument | Default | Description |
|---|---:|---|
| `--preprocess_ckpt_dir` | `<ckpt_dir>/process_checkpoint` | Wan-Animate preprocessing checkpoints. |
| `--preprocess_output_dir` | `<save_dir>/preprocessed` | Root directory for reusable preprocessing results. |
| `--preprocess_long_edge` | `832` | Long edge used by preprocessing. |
| `--preprocess_short_edge` | `480` | Short edge used by preprocessing. |
| `--preprocess_fps` | `16` | Frame rate used by preprocessing. `-1` is also accepted. |
| `--vitpose_batch_size` | `16` | ViTPose preprocessing batch size. Must be positive. |
| `--smooth` | `False` | Temporally smooth pose and face crops. |
| `--retarget` | `False` | Retarget the source-video pose to the reference body proportions. |
| `--force_preprocess` | Disabled | Ignore reusable preprocessing results and run preprocessing again. |

Results are cached per task id under `--preprocess_output_dir` and reused only
when the source video, the reference image, and the preprocessing options above
(except `--vitpose_batch_size`) all match the cached run; otherwise that task is
preprocessed again.

## Sampling and resolution

| Argument | Default | Description |
|---|---:|---|
| `--frame_num` | `-1` | Number of output frames. `-1` processes the full source. |
| `--sample_solver` | `unipc` | Sampling solver: `unipc` or `dpm++`. |
| `--sample_steps` | `2` | Uniform denoising step count, used only when custom timesteps are disabled. |
| `--custom_timesteps` | `1000 250` | Explicit pre-shift timesteps for the distilled model. Pass the option without values to disable them. |
| `--sample_shift` | `5.0` | Flow-matching timestep shift. |
| `--sample_guide_scale` | `1.0` | Compatibility option for this CFG-free pipeline; values other than `1.0` are ignored. |
| `--base_seed` | `-1` | Random seed. A negative value selects a random seed for the run. |
| `--long_edge` | `832` | Long edge of the output video. |
| `--short_edge` | `480` | Short edge of the output video. |

A fixed `--frame_num` must follow `9 + 12 × n`, for example `9`, `21`, `33`,
or `81`. Output orientation follows the reference image. Output edges must be
positive multiples of 16, with `--long_edge >= --short_edge`.

`--sample_steps` and `--sample_shift` must be positive. Custom timesteps must
strictly decrease within `(0, 1000]`; the solver appends the final zero itself.
Pose and face videos in a preprocessed folder must have equal, nonzero frame counts.

To use a uniform two-step schedule instead of the default distilled
timesteps:

```bash
--custom_timesteps --sample_steps 2
```

## LoRA settings

| Argument | Default | Description |
|---|---:|---|
| `--lora_paths` | `None` | One or more LoRA checkpoint paths, fused in the supplied order. |
| `--lora_strengths` | `1.0` per LoRA | Either one shared value or one value for each entry in `--lora_paths`. |

For the three released EditaLive LoRAs, the standard ordering is:

```bash
--lora_paths \
  ./weights/EditaLive/editalive_edit.safetensors \
  ./weights/EditaLive/lightx2v.safetensors \
  ./weights/EditaLive/editalive_streaming.safetensors
```

## Compilation, decoding, and memory

| Argument | Default | Description |
|---|---:|---|
| `--enable_compile` | `False` | Enable `torch.compile` and its warm-up. Disabled mode starts faster; enabled mode benefits long or repeated runs. |
| `--compile_cache_dir` | System temporary directory | Directory for the compilation cache. Point it somewhere persistent to skip most of the warm-up on later runs; it holds about 400 MB per configuration and is only reused on the same GPU and PyTorch version. |
| `--fast_decode` | `False` | Decode with Flash-VAED instead of the original Wan VAE decoder. |
| `--fast_decoder_pth` | `./weights/Flash-VAED/Flash_VAED_Wan.pth` | Flash-VAED checkpoint used by `--fast_decode True`. |
| `--offload_model` | `True` | Move the text encoder back to CPU after encoding. |
| `--t5_cpu` | Disabled | Keep the T5 text encoder on CPU throughout inference. |
| `--fp8` | Disabled | Quantize transformer linear layers to scaled FP8. Requires a compatible GPU (SM 8.9 or newer). |
| `--low_vram` | `off` | Trade speed for VRAM. `weights` keeps the transformer blocks in CPU memory and streams them to the GPU one at a time; `weights+kv` also keeps the rolling KV cache in CPU memory. Output is identical in every mode. Cannot be combined with `--enable_compile`. See the README for measured VRAM and slowdown per resolution. |

## Video Sparse Attention

| Argument | Default | Description |
|---|---:|---|
| `--vsa_sparsity` | `0.75` | Fraction of KV blocks dropped by causal self-attention. Use `0` for dense attention. |
| `--vsa_tile_size` | `1 8 8` | Temporal-height-width tile size. |
| `--vsa_sink_frames` | `1` | Cache frames VSA always keeps. Only `0` and `1` differ; larger values behave as `1`. |
| `--vsa_backend` | `kernel` | Backend selection: `kernel`, `sdpa`, or `auto`. |

## Common configurations

```bash
# Process the complete source video.
--frame_num -1

# Run at 384 × 672, including preprocessing of a raw video.
--short_edge 384 --long_edge 672 \
--preprocess_short_edge 384 --preprocess_long_edge 672

# Enable compilation for a long or repeated run.
--enable_compile True

# Preserve source audio at matching frame rates.
--keep_audio True --preprocess_fps 16 --fps 16
```

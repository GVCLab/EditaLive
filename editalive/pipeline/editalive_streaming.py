import gc
import logging
import math
import os
from copy import deepcopy

from ..utils.validation import validate_resolution, validate_sampling

import cv2
import numpy as np
import torch
from decord import VideoReader
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from ..compile_config import is_compile_enabled

from ..models import CLIPModel, WanT5EncoderModel
from ..models.vae_streaming import WanVAE
from ..models.editalive_transformer3d import EditaLiveTransformer3DModel
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas, retrieve_timesteps)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..utils.lora_utils import apply_lora_updates, plan_lora_updates
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Layers kept in bf16 under --fp8: embeddings, the output head, and the small
# face/motion encoders, where quantization costs accuracy and saves little.
FP8_IGNORE_KEYS = [
    'text_embedding', 'time_embedding', 'time_projection', 'head.head',
    'face_encoder', 'motion_encoder',
]


LOW_VRAM_MODES = ("off", "weights", "weights+kv")


@torch.no_grad()
def _place_transformer(transformer, device, weight_dtype, lora_updates, fp8, low_vram="off"):
    """Move the transformer onto ``device``, fuse its LoRAs, then quantize.
    """
    if low_vram != "off":
        from ..utils.block_swap import BlockStreamer, KVCacheStreamer
        from ..utils.fp8_linear import mark_fp8_ignored, replace_linear_with_scaled_fp8
        if fp8:
            mark_fp8_ignored(transformer, FP8_IGNORE_KEYS)
        with tqdm(total=len(transformer.blocks), desc="Preparing blocks", unit="block") as progress:
            for block in transformer.blocks:
                block.to(device, weight_dtype)
                apply_lora_updates(lora_updates, block)
                if fp8:
                    replace_linear_with_scaled_fp8(block)
                block.to("cpu")
                progress.update()
        transformer.block_streamer = BlockStreamer(transformer.blocks, device)
        if low_vram == "weights+kv":
            transformer.kv_streamer = KVCacheStreamer(transformer.blocks, device)

    transformer.to(device, weight_dtype)
    apply_lora_updates(lora_updates, transformer)

    if fp8:
        from ..utils.fp8_linear import replace_linear_with_scaled_fp8
        replace_linear_with_scaled_fp8(transformer, ignore_keys=FP8_IGNORE_KEYS)

    if lora_updates:
        raise RuntimeError(
            f"{len(lora_updates)} LoRA updates did not match any module that "
            "was placed; the checkpoint and the model may be out of sync.")

    # Fusing and quantizing churn through short-lived buffers (fp32 LoRA deltas,
    # the pre-quantization weights). Hand those segments back so the generation
    # loop starts on a clean pool.
    gc.collect()
    torch.cuda.empty_cache()

    target = torch.empty(0, device=device).device
    stray = sorted(
        name for name, tensor in
        list(transformer.named_parameters()) + list(transformer.named_buffers())
        if tensor.device.type != "meta" and tensor.device != target)
    if stray:
        raise RuntimeError(
            f"{len(stray)} tensors were left off {target}, e.g. {stray[:5]}")


def _packed_frames(video):
    """Pack a decoded chunk as uint8 HWC frames staged through pinned memory.

    The same frames as float32 in pageable memory are four times the bytes at a
    fraction of the bandwidth: about 24 ms per chunk at 384x672 against 0.3 ms.
    """
    pixels = video.permute(1, 2, 3, 0).float().add(1).mul(127.5).clamp(0, 255).byte()
    frames = torch.empty(pixels.shape, dtype=torch.uint8, pin_memory=True)
    frames.copy_(pixels)
    return frames


class EditaLiveStreamingPipeline:
    def __init__(
        self,
        config,
        tokenizer,
        text_encoder,
        vae: WanVAE,
        transformer: EditaLiveTransformer3DModel,
        clip_image_encoder,
        device=torch.device("cuda"),
        t5_cpu=False,
        vae_device=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.noise_model = transformer
        self.clip = clip_image_encoder
        self.device = torch.device(device)
        self.t5_cpu = t5_cpu
        self.vae_device = torch.device(vae_device) if vae_device is not None else self.device

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.text_len = transformer.text_len
        self.sample_prompt = config.prompt

        self.kv_cache = None
        self.crossattn_cache = None
        self._streaming_warmed_up = False
        self._forward_warmed_up = False
        self._warmup_key = None
        self._active_session = None

        self.frame_seq_length = 1560
        self.local_attn_size = config.get("local_attn_size", 6)
        self.chunk_size = config.get("chunk_size", 3)

    # ------------------------------------------------------------------ #
    #   Construction                                                     #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        config,
        device=torch.device("cuda"),
        weight_dtype=torch.bfloat16,
        t5_cpu=False,
        lora_paths=None,
        lora_strengths=None,
        fp8=False,
        low_vram="off",
        fast_decode=False,
        vsa_sparsity=0.0,
        vsa_backend="auto",
        vsa_tile_size=(1, 8, 8),
        vsa_sink_frames=1,
        fast_decoder_pth=None,
        vae_device=None,
    ):
        # torch._scaled_mm needs SM 8.9+; fail here rather than at the first
        # forward, after the 14B weights have loaded.
        if fp8 and torch.cuda.get_device_capability(device) < (8, 9):
            major, minor = torch.cuda.get_device_capability(device)
            raise RuntimeError(
                f"--fp8 needs a GPU with compute capability 8.9 or newer; "
                f"{torch.cuda.get_device_name(device)} is {major}.{minor}.")

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, config.t5_tokenizer))

        # T5 stays on the CPU here: generate() moves it to the GPU only while
        # encoding the prompt, so it never sits next to the full transformer
        # during loading (a ~10 GiB higher load peak otherwise).
        logger.info("Loading text encoder ...")
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_path, config.t5_checkpoint),
            additional_kwargs=dict(config.t5_kwargs),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        ).eval().requires_grad_(False)

        # Like T5, CLIP stays on the CPU: generate() needs it once per clip, for
        # the reference frame, and it would otherwise hold ~2 GiB of GPU memory
        # for the whole run.
        logger.info("Loading CLIP image encoder ...")
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(model_path, config.clip_checkpoint),
        ).eval().requires_grad_(False).to(torch.float16)

        logger.info("Loading streaming VAE ...")
        vae = WanVAE(
            vae_pth=os.path.join(model_path, config.vae_checkpoint),
            device=vae_device if vae_device is not None else device,
            fast_decode=fast_decode,
            fast_decoder_pth=fast_decoder_pth,
        )

        logger.info("Loading streaming transformer ...")
        transformer = EditaLiveTransformer3DModel.from_pretrained(
            os.path.join(model_path, config.transformer_subpath),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        transformer = transformer.eval().requires_grad_(False)

        if fp8 and not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("--fp8 needs a torch build with torch._scaled_mm.")
        if fp8:
            logger.info("Enabling FP8 quantization on the transformer")

        # The LoRA updates are resolved while the model is still on the CPU, so
        # _place_transformer can move it across one block at a time.
        lora_updates = plan_lora_updates(
            transformer, lora_paths, lora_strengths, verbose=False)
        _place_transformer(transformer, device, weight_dtype, lora_updates, fp8, low_vram)

        if vsa_sparsity > 0.0:
            transformer.set_vsa(
                sparsity=vsa_sparsity, backend=vsa_backend,
                tile_size=vsa_tile_size, sink_frames=vsa_sink_frames)
            logger.info(
                f"VSA enabled: sparsity={vsa_sparsity}, backend={vsa_backend}, "
                f"tile={tuple(vsa_tile_size)}, sink_frames={vsa_sink_frames}")

        return cls(
            config=config,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            clip_image_encoder=clip_image_encoder,
            device=device,
            t5_cpu=t5_cpu,
            vae_device=vae_device,
        )

    # ------------------------------------------------------------------ #
    #   Text conditioning                                                #
    # ------------------------------------------------------------------ #
    def encode_prompt(self, prompts, device=None):
        """Encode prompts to fixed-length embeddings for compile reuse.

        Padding positions are zeroed, matching the DiT's previous internal padding.
        """
        device = device or self.device
        if isinstance(prompts, str):
            prompts = [prompts]

        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.text_len,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        encoder_device = torch.device("cpu") if self.t5_cpu else device
        prompt_embeds = self.text_encoder(
            input_ids.to(encoder_device),
            attention_mask=attention_mask.to(encoder_device))[0]
        prompt_embeds = prompt_embeds.to(dtype=self.param_dtype, device=device)
        padding_mask = ~attention_mask.to(device=device, dtype=torch.bool)
        prompt_embeds = prompt_embeds.masked_fill(padding_mask.unsqueeze(-1), 0)
        return list(prompt_embeds)

    # ------------------------------------------------------------------ #
    #   Source preparation                                               #
    # ------------------------------------------------------------------ #
    def inputs_padding(self, array, target_len):
        """Extend a frame list to ``target_len`` by ping-ponging through it."""
        if len(array) == 0:
            raise ValueError("Cannot pad an empty frame sequence.")
        if len(array) == 1:
            return [deepcopy(array[0]) for _ in range(target_len)]
        idx = 0
        flip = False
        target_array = []
        while len(target_array) < target_len:
            target_array.append(deepcopy(array[idx]))
            if flip:
                idx -= 1
            else:
                idx += 1
            if idx == 0 or idx == len(array) - 1:
                flip = not flip
        return target_array[:target_len]

    @staticmethod
    def _floor_clip_len(n):
        """Largest valid 9+12*n clip length not exceeding n (minimum 9)."""
        n = int(n)
        if n < 9:
            return 9
        return 9 + 12 * ((n - 9) // 12)

    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        """Build the 4-channel temporal mask that rides alongside the latents."""
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk


    def resize_and_center_crop(self, img_array, target_height, target_width):
        """Scale to cover (target_height, target_width), then centre-crop -- no bars.

        Accepts a single ``(H, W, C)`` image or an ``(N, H, W, C)`` batch.
        """
        is_single = False
        if len(img_array.shape) == 3:
            img_array = img_array[np.newaxis, ...]
            is_single = True

        N, ori_h, ori_w, C = img_array.shape

        # Take the LARGER of the two scale factors so the short edge fills the
        # target and the long edge overflows (that overflow is what gets cropped).
        scale = max(target_height / ori_h, target_width / ori_w) + 0.01
        new_h, new_w = int(ori_h * scale), int(ori_w * scale)

        resized_frames = np.empty((N, new_h, new_w, C), dtype=img_array.dtype)
        for i in range(N):
            resized_frames[i] = cv2.resize(img_array[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        y_start = (new_h - target_height) // 2
        x_start = (new_w - target_width) // 2
        cropped = resized_frames[:, y_start:y_start + target_height, x_start:x_start + target_width, :]

        return cropped[0] if is_single else cropped

    def prepare_source(self, src_pose_path, src_face_path, src_ref_path,
                       long_edge=832, short_edge=480):
        """Read the preprocessed pose video, face video and reference image.

        The reference image's aspect ratio decides the orientation: landscape gets
        ``long_edge x short_edge``, portrait the transpose. Pose frames and the
        reference are resized+cropped to that; face crops keep their own size,
        since the motion encoder has its own input resolution.
        """
        validate_resolution(long_edge, short_edge)
        with Image.open(src_ref_path) as image:
            refer_image_pil = image.convert("RGB")
        orig_w, orig_h = refer_image_pil.size
        refer_images = np.array(refer_image_pil)

        if orig_w >= orig_h:
            target_width, target_height = long_edge, short_edge
        else:
            target_width, target_height = short_edge, long_edge

        pose_video_reader = VideoReader(src_pose_path)
        face_video_reader = VideoReader(src_face_path)
        frame_count = len(pose_video_reader)
        if frame_count == 0 or len(face_video_reader) == 0:
            raise ValueError("Source pose and face videos must contain frames.")
        if frame_count != len(face_video_reader):
            raise ValueError("Source pose and face videos must have the same frame count.")
        indices = list(range(frame_count))
        cond_images = pose_video_reader.get_batch(indices).asnumpy()
        face_images = face_video_reader.get_batch(indices).asnumpy()

        cond_images = self.resize_and_center_crop(
            cond_images, target_height=target_height, target_width=target_width)
        refer_images = self.resize_and_center_crop(
            refer_images, target_height=target_height, target_width=target_width)

        return cond_images, face_images, refer_images

    # ------------------------------------------------------------------ #
    #   Rolling caches                                                    #
    # ------------------------------------------------------------------ #
    def _initialize_kv_cache(self, dtype, device):
        """Allocate one empty rolling KV cache per transformer layer.

        The caches grow by concatenation as chunks are committed and are trimmed
        back to the attention window by :meth:`update_kv`, so they start at length
        zero rather than at a pre-allocated maximum.
        """
        streamer = getattr(self.noise_model, "kv_streamer", None)
        if streamer is not None:
            device = "cpu"
        self.kv_cache = [
            {
                "k": torch.zeros([1, 0, self.noise_model.num_heads, self.noise_model.d],
                                 dtype=dtype, device=device),
                "v": torch.zeros([1, 0, self.noise_model.num_heads, self.noise_model.d],
                                 dtype=dtype, device=device),
                "shift": 0,
            }
            for _ in range(self.noise_model.num_layers)
        ]
        if streamer is not None:
            streamer.bind(self.kv_cache, self.frame_seq_length, self.local_attn_size + 1)

    def _initialize_crossattn_cache(self):
        """Allocate the per-layer text/image cross-attention cache."""
        self.crossattn_cache = [
            {"k": None, "v": None, "k_img": None, "v_img": None, "is_init": False}
            for _ in range(self.noise_model.num_layers)
        ]

    def update_kv(self):
        """Trim the KV cache back to the attention window.

        Keeps the first 4 latent frames (the reference frame plus chunk 0, which
        act as an attention sink and anchor identity) and the most recent 3, so the
        cache stays at ``local_attn_size + 1`` frames no matter how long the clip is.
        """
        if getattr(self.noise_model, "kv_streamer", None) is not None:
            return  # the streamer trims on the GPU as it writes the cache back

        shift = self.kv_cache[0]["shift"]
        if shift > self.local_attn_size + 1:
            fsl = self.frame_seq_length
            for idx in range(self.noise_model.num_layers):
                k = self.kv_cache[idx]["k"]
                v = self.kv_cache[idx]["v"]
                self.kv_cache[idx]["k"] = torch.cat([k[:, :fsl * 4], k[:, -fsl * 3:]], dim=1)
                self.kv_cache[idx]["v"] = torch.cat([v[:, :fsl * 4], v[:, -fsl * 3:]], dim=1)
                self.kv_cache[idx]["shift"] = self.local_attn_size + 1

    def reset_cache(self):
        """Rewind to just the reference frame -- starts a new clip on the same identity."""
        fsl = self.frame_seq_length
        for idx in range(self.noise_model.num_layers):
            self.crossattn_cache[idx]["is_init"] = False
            self.kv_cache[idx]["k"] = self.kv_cache[idx]["k"][:, :fsl]
            self.kv_cache[idx]["v"] = self.kv_cache[idx]["v"][:, :fsl]
            self.kv_cache[idx]["shift"] = 1

    def reset_crossattn_cache(self):
        """Force the next forward to re-project the text context (e.g. a new prompt)."""
        for idx in range(self.noise_model.num_layers):
            self.crossattn_cache[idx]["is_init"] = False

    def clean_cache(self):
        """Drop both caches. Call between generations to release their memory."""
        self.kv_cache = None
        self.crossattn_cache = None
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    #   Warmup                                                            #
    # ------------------------------------------------------------------ #
    def _warmup_streaming(self, pixel_chunks, conditioning_pixel_values,
                          face_pixel_values, lat_h, lat_w, progress=None, progress_callback=None):
        """Pay the compile / autotune cost before the timed loop starts.

        Two separate costs: ``torch.compile`` guards inside the streaming VAE, and
        cuDNN's algorithm search for each new motion-encoder input shape (one
        search per distinct chunk length). Both would otherwise land as multi-second
        stalls in the middle of generation, which defeats the point of streaming.
        """
        if not is_compile_enabled():
            return

        if self._streaming_warmed_up or not hasattr(self.vae, "warmup_stream"):
            return

        if progress_callback is not None:
            progress_callback("Compiling & warming up VAE")
        if progress is not None:
            progress.set_postfix_str("VAE streaming", refresh=True)
        torch.backends.cudnn.benchmark = True
        pix_h, pix_w = conditioning_pixel_values.shape[-2:]
        self.vae.warmup_stream(
            pixel_chunks=sorted(set(pixel_chunks)),
            latent_chunk_size=self.chunk_size,
            pixel_hw=(pix_h, pix_w),
            latent_hw=(lat_h, lat_w),
            encode_dtype=conditioning_pixel_values.dtype,
        )
        if progress is not None:
            progress.update()

        if progress_callback is not None:
            progress_callback("Warming up motion encoder")
        if progress is not None:
            progress.set_postfix_str("motion encoder", refresh=True)
        face_h, face_w = face_pixel_values.shape[-2:]
        encode_bs = 12
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True), \
                torch.no_grad():
            for n in sorted(set(pixel_chunks)):
                dummy_face = torch.zeros(n, 3, face_h, face_w,
                                         device=self.device, dtype=face_pixel_values.dtype)
                motion_feats = []
                for i in range(math.ceil(dummy_face.shape[0] / encode_bs)):
                    motion_feats.append(self.noise_model.motion_encoder.get_motion(
                        dummy_face[i * encode_bs:(i + 1) * encode_bs]))
                motion_feat = torch.cat(motion_feats).to(face_pixel_values.dtype)
                motion_feat = rearrange(motion_feat, "(b t) c -> b t c", t=n)
                self.noise_model.face_encoder.forward_streaming(motion_feat)
        self.noise_model.face_encoder.reset_streaming_state()
        self._streaming_warmed_up = True
        if progress is not None:
            progress.update()

    def _warmup_forward(self, noise, context, clip_context, y, lat_h, lat_w,
                        max_seq_len, chunk_num, progress=None):
        """Compile every shape/branch variant of the DiT forward the loop will hit.

        Only meaningful with ``--enable_compile True``. The variants that matter are
        ``is_update`` True/False, ``context`` tensor/None (the crossattn cache flips
        ``is_init`` after the first call), and the two cache lengths the loop settles
        into (4 and 7 latent frames). Replaying on faithful copies of the real cache
        dicts matters: dynamo guards ``crossattn_cache['is_init']`` by the object id
        of the True/False singleton, so fresh dicts would compile a different guard
        sequence and force stray recompiles on the first real chunk.
        """
        if not is_compile_enabled():
            return

        if self._forward_warmed_up:
            return

        if progress is not None:
            progress.set_postfix_str("transformer forward", refresh=True)
        fsl = self.frame_seq_length
        warm_kv = [{"k": d["k"].clone(), "v": d["v"].clone(), "shift": d["shift"]}
                   for d in self.kv_cache]
        warm_cross = [dict(d) for d in self.crossattn_cache]
        for d in warm_cross:
            d["is_init"] = False
        warm_pose = torch.zeros(1, 16, self.chunk_size, lat_h, lat_w,
                                dtype=torch.float32, device=self.device)
        warm_motion = torch.zeros(1, self.chunk_size, 5, self.noise_model.dim,
                                  dtype=torch.bfloat16, device=self.device)
        warm_t = torch.zeros([1], dtype=torch.int64, device=self.device)
        warm_y = [y[:, 1:4].contiguous()]

        def _warm_forward(is_update):
            self.noise_model(
                [noise[0][:, 0:self.chunk_size].contiguous()],
                t=warm_t, is_update=is_update,
                context=context, seq_len=max_seq_len,
                clip_fea=clip_context, y=warm_y,
                pose_latents=warm_pose, motion_vec=warm_motion,
                kv_cache=warm_kv, crossattn_cache=warm_cross, is_causal=True)

        def _warm_update_kv():
            if warm_kv[0]["shift"] > self.local_attn_size + 1:
                for d in warm_kv:
                    d["k"] = torch.cat([d["k"][:, :fsl * 4], d["k"][:, -fsl * 3:]], dim=1)
                    d["v"] = torch.cat([d["v"][:, :fsl * 4], d["v"][:, -fsl * 3:]], dim=1)
                    d["shift"] = self.local_attn_size + 1

        for wc in range(min(3, chunk_num)):
            if wc == 0:
                # chunk 0 runs the sampler (is_update False; the 1st call compiles
                # the context=tensor path and flips is_init, the 2nd the
                # context=None path) then a trailing is_update=True call that
                # writes the clean latents into the cache.
                _warm_forward(False)
                _warm_forward(False)
                _warm_forward(True)
            else:
                # chunk >= 1: first step is_update False, last step is_update True.
                _warm_forward(False)
                _warm_forward(True)
            _warm_update_kv()


        self.noise_model.face_encoder.reset_streaming_state()
        del warm_kv, warm_cross, warm_pose, warm_motion, warm_t, warm_y
        torch.cuda.empty_cache()
        self._forward_warmed_up = True
        if progress is not None:
            progress.update()

    @torch.no_grad()
    def warmup(self, src_root_path, fast_decode=False,
               long_edge=832, short_edge=480):
        if not is_compile_enabled():
            return
        with Image.open(os.path.join(src_root_path, "src_ref.png")) as ref:
            width, height = ((long_edge, short_edge) if ref.width >= ref.height
                             else (short_edge, long_edge))
        reader = VideoReader(os.path.join(src_root_path, "src_face.mp4"))
        if not len(reader):
            raise ValueError("Source face video contains no frames.")
        face_hw = reader[0].shape[:2]
        self.warmup_shape(height, width, face_hw=face_hw, fast_decode=fast_decode)

    @torch.no_grad()
    def warmup_shape(self, height, width, face_hw=(512, 512), fast_decode=False, progress_callback=None):
        if self._active_session is not None:
            raise RuntimeError("Close the active session before warmup.")
        if min(height, width) <= 0 or height % 16 or width % 16:
            raise ValueError("Output dimensions must be positive multiples of 16.")
        if not is_compile_enabled():
            return
        face_h, face_w = face_hw
        key = (height, width, face_h, face_w, fast_decode)
        if key != self._warmup_key:
            self._streaming_warmed_up = False
            self._forward_warmed_up = False
        if self._streaming_warmed_up and self._forward_warmed_up:
            return

        if hasattr(self.vae, "set_fast_decode"):
            self.vae.set_fast_decode(fast_decode)

        lat_h, lat_w = height // 8, width // 8
        self.frame_seq_length = lat_h * lat_w // 4
        pixel_chunks = [9, 12]
        conditioning = torch.empty(
            1, 3, 1, height, width, dtype=torch.bfloat16)
        face = torch.empty(
            1, 3, 1, face_h, face_w, dtype=torch.bfloat16)

        pending = (0 if self._streaming_warmed_up else 2) + (
            0 if self._forward_warmed_up else 1)
        try:
            with tqdm(
                total=pending,
                desc="Warmup",
                unit="component",
            ) as progress:
                self._warmup_streaming(
                    pixel_chunks, conditioning, face, lat_h, lat_w,
                    progress=progress, progress_callback=progress_callback)

                if not self._forward_warmed_up:
                    if progress_callback is not None:
                        progress_callback("Compiling & warming up generator")
                    progress.set_postfix_str(
                        "transformer forward", refresh=True)
                    self._initialize_kv_cache(
                        dtype=torch.bfloat16, device=self.device)
                    self._initialize_crossattn_cache()

                    latent_channels = self.noise_model.out_dim
                    condition_channels = (
                        self.noise_model.in_dim - latent_channels)
                    clip_dim = self.noise_model.img_emb.proj[1].in_features
                    context = [torch.zeros(
                        self.text_len,
                        self.noise_model.text_dim,
                        dtype=self.param_dtype,
                        device=self.device,
                    )]
                    clip_context = torch.zeros(
                        1, 257, clip_dim,
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    ref_latent = torch.zeros(
                        latent_channels, 1, lat_h, lat_w,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    y_ref = torch.zeros(
                        condition_channels, 1, lat_h, lat_w,
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    y = torch.zeros(
                        condition_channels, 4, lat_h, lat_w,
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    noise = [torch.zeros(
                        latent_channels, self.chunk_size * 3, lat_h, lat_w,
                        dtype=torch.float32,
                        device=self.device,
                    )]
                    max_seq_len = self.chunk_size * self.frame_seq_length

                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=True,
                    ):
                        self.noise_model.forward_ref(
                            [ref_latent],
                            t=torch.zeros(
                                [1], dtype=torch.int64, device=self.device),
                            context=context,
                            seq_len=self.frame_seq_length,
                            clip_fea=clip_context,
                            y=[y_ref],
                            kv_cache=self.kv_cache,
                            crossattn_cache=self.crossattn_cache,
                            is_update=True,
                        )
                        self.reset_crossattn_cache()
                        self._warmup_forward(
                            noise, context, clip_context, y,
                            lat_h, lat_w, max_seq_len, chunk_num=3,
                            progress=progress)

                progress.set_postfix_str("done", refresh=True)
                self._warmup_key = key
        finally:
            self.vae.model.clear_cache()
            self.vae.model.first_encode = True
            self.vae.model.first_decode = True
            self.noise_model.face_encoder.reset_streaming_state()
            self.clean_cache()

    # ------------------------------------------------------------------ #
    #   Sampling                                                          #
    # ------------------------------------------------------------------ #
    def _build_scheduler(self, sample_solver, sampling_steps, custom_timesteps, shift):
        """Instantiate the flow-matching solver and its timestep schedule.

        ``custom_timesteps`` are given pre-shift (on the 0..num_train_timesteps
        scale) and are what the distilled few-step checkpoints expect; passing None
        falls back to a uniform ``sampling_steps`` schedule.
        """
        validate_sampling(sampling_steps, custom_timesteps, shift, self.num_train_timesteps)
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            if custom_timesteps is not None:
                # set_timesteps() applies the shift warp internally.
                pre_shift_sigmas = np.array(
                    [t / self.num_train_timesteps for t in custom_timesteps])
                sample_scheduler.set_timesteps(
                    device=self.device, sigmas=pre_shift_sigmas, shift=shift)
            else:
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            if custom_timesteps is not None:
                # dpm++ is constructed with shift=1, so warp outside (same formula
                # get_sampling_sigmas uses).
                pre_shift_sigmas = np.array(
                    [t / self.num_train_timesteps for t in custom_timesteps])
                sampling_sigmas = shift * pre_shift_sigmas / (
                    1 + (shift - 1) * pre_shift_sigmas)
            else:
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=self.device, sigmas=sampling_sigmas)
        else:
            raise NotImplementedError(f"Unsupported solver: {sample_solver}")

        return sample_scheduler, timesteps

    @torch.no_grad()
    def prepare_stream_model(self, height, width, fast_decode=False, progress_callback=None):
        if self._active_session is not None:
            raise RuntimeError("Close the active session before preparation.")
        self.warmup_shape(height, width, fast_decode=fast_decode, progress_callback=progress_callback)
        key = (height, width, fast_decode)
        cached = getattr(self, "_prepared_stream_model", None) or {}
        if cached.get("key") == key:
            return cached
        self._prepared_stream = None
        context_default = cached.get("context_default")
        if context_default is None:
            if progress_callback is not None:
                progress_callback("Preparing text encoder")
            try:
                if not self.t5_cpu:
                    self.text_encoder.to(self.device)
                context_default = self.encode_prompt(["Best quality."])
            finally:
                if not self.t5_cpu:
                    self.text_encoder.cpu()
                    torch.cuda.empty_cache()
        if progress_callback is not None:
            progress_callback("Warming up reference encoders")
        try:
            lat_h, lat_w = height // 8, width // 8
            y_reft = self.vae.encode([torch.zeros(3, 13, height, width, device=self.vae_device)])[0].to(self.device)
            mask = self.get_i2v_mask(4, lat_h, lat_w, 0, device=self.device)
            y = torch.concat([mask, y_reft]).to(dtype=torch.bfloat16, device=self.device)
            mask_clean = self.get_i2v_mask(3, lat_h, lat_w, 3, device=self.device)
            self.clip.to(self.device)
            self.clip([torch.zeros(3, 1, height, width, device=self.device, dtype=torch.bfloat16)])
        finally:
            self.clip.cpu()
            self._clear_stream_state()
        self._prepared_stream_model = dict(key=key, context_default=context_default, y=y, mask_clean=mask_clean)
        return self._prepared_stream_model


    @torch.no_grad()
    def prepare_stream(self, prompt, height, width, fast_decode=False):
        model = self.prepare_stream_model(height, width, fast_decode=fast_decode)
        prompt = prompt or self.sample_prompt
        cached = getattr(self, "_prepared_stream", None) or {}
        if cached.get("key") == model["key"] and cached.get("prompt") == prompt:
            return cached
        try:
            if not self.t5_cpu:
                self.text_encoder.to(self.device)
            context = self.encode_prompt([prompt])
        finally:
            if not self.t5_cpu:
                self.text_encoder.cpu()
                torch.cuda.empty_cache()
        self._prepared_stream = dict(model, prompt=prompt, context=context)
        return self._prepared_stream


    @torch.no_grad()
    def stream_start(self, ref_image, prompt="", seed=-1, generator=None,
                     fast_decode=False, offload_model=True, shift=5.0,
                     sample_solver="unipc", sampling_steps=2,
                     custom_timesteps=(1000, 250), prepared=None):
        if self._active_session is not None:
            raise RuntimeError("A streaming session is already active.")
        try:
            session = StreamingSession(
                self, ref_image, prompt or self.sample_prompt, seed, generator,
                fast_decode, offload_model, shift, sample_solver,
                sampling_steps, custom_timesteps, prepared=prepared)
        except Exception:
            self._clear_stream_state()
            raise
        self._active_session = session
        return session


    def _clear_stream_state(self):
        self.vae.model.clear_cache()
        self.vae.model.first_encode = True
        self.vae.model.first_decode = True
        self.noise_model.face_encoder.reset_streaming_state()
        self.clean_cache()


    @torch.no_grad()
    def generate(
        self,
        src_root_path,
        sink,
        input_prompt="",
        clip_len=-1,
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=2,
        custom_timesteps=(1000, 250),
        guide_scale=1.0,
        seed=-1,
        offload_model=True,
        fast_decode=False,
        long_edge=832,
        short_edge=480,
    ):
        r"""Generate an edited video, streaming it chunk by chunk.

        Args:
            src_root_path: preprocessed source directory containing
                ``src_pose.mp4``, ``src_face.mp4`` and ``src_ref.png``.
            sink: called once per chunk with its ``[T, H, W, C]`` uint8 frames.
                Chunks are handed over as they are decoded, so host memory stays
                flat no matter how long the clip is.
            input_prompt: the edit instruction.
            clip_len: frames generated by this run. Use ``-1`` for the full source
                length, rounded down to the nearest ``9 + 12*n``; explicit lengths
                must already satisfy that form.
            shift: flow-matching schedule shift.
            sample_solver: ``'unipc'`` or ``'dpm++'``.
            sampling_steps: number of denoising steps -- used only when
                ``custom_timesteps`` is None.
            custom_timesteps: explicit pre-shift timesteps. The distilled
                checkpoints are trained for ``(1000, 250)``; pass None to fall back
                to a uniform ``sampling_steps`` schedule.
            guide_scale: kept for API parity. This pipeline is CFG-free -- the
                distilled few-step model does not need it, and there is no
                negative-prompt branch -- so anything other than 1.0 is ignored
                with a warning.
            seed: RNG seed for the initial noise.
            offload_model: move the text encoder back to the CPU after encoding.
            fast_decode: decode with the lightweight Flash-VAED decoder (~4x
                faster per chunk) instead of the original Wan 2.1 decoder.
            long_edge / short_edge: target resolution; orientation follows the
                reference image.

        Returns:
            ``(frames, height, width)`` -- how many frames reached ``sink`` and
            the size they were written at.
        """
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed if seed >= 0 else torch.seed() % (2 ** 31))

        if guide_scale != 1.0:
            logger.warning(
                "guide_scale=%s ignored: the streaming pipeline runs CFG-free.",
                guide_scale)
        if input_prompt == "":
            input_prompt = self.sample_prompt

        # Direct pipeline users get the same one-time pre-loop warmup as the CLI.
        # The CLI calls this before entering its source/prompt loops, so this is a
        # no-op there.
        self.warmup(
            src_root_path,
            fast_decode=fast_decode,
            long_edge=long_edge,
            short_edge=short_edge,
        )

        src_pose_path = os.path.join(src_root_path, "src_pose.mp4")
        src_face_path = os.path.join(src_root_path, "src_face.mp4")
        src_ref_path = os.path.join(src_root_path, "src_ref.png")

        cond_images, face_images, refer_images = self.prepare_source(
            src_pose_path=src_pose_path,
            src_face_path=src_face_path,
            src_ref_path=src_ref_path,
            long_edge=long_edge,
            short_edge=short_edge)

        # ---- text conditioning (once per clip) ----
        if not self.t5_cpu:
            self.text_encoder.to(self.device)
        context = self.encode_prompt([input_prompt])
        context_default = self.encode_prompt(["Best quality."])
        if offload_model and not self.t5_cpu:
            self.text_encoder.cpu()
            torch.cuda.empty_cache()

        # ---- frame budget ----
        # The streaming loop generates one clip. In full-length mode, choose the
        # largest supported 9+12*n length that does not invent mirrored tail frames.
        source_frame_len = len(cond_images)
        if source_frame_len == 0:
            raise ValueError("Source pose video contains no frames.")
        if clip_len == -1:
            clip_len = self._floor_clip_len(source_frame_len)
        elif clip_len < 9 or (clip_len - 9) % 12 != 0:
            raise ValueError(
                f"clip_len must be -1 (full length) or 9+12*n; got {clip_len}.")
        self.clip_len = clip_len
        if source_frame_len > clip_len:
            cond_images = cond_images[:clip_len]
            face_images = face_images[:clip_len]
        elif source_frame_len < clip_len:
            cond_images = self.inputs_padding(cond_images, clip_len)
            face_images = self.inputs_padding(face_images, clip_len)
        logger.info('source frames: %d inference frames: %d', source_frame_len, clip_len)

        # Keep the selected pose/face frames in CPU memory and ship only the current
        # chunk to the GPU inside the loop. GPU residency is therefore tens of MiB
        # and independent of the requested clip length.
        conditioning_pixel_values = rearrange(
            torch.tensor(np.stack(cond_images) / 127.5 - 1),
            "t h w c -> 1 c t h w").to(dtype=torch.bfloat16).cpu()
        face_pixel_values = rearrange(
            torch.tensor(np.stack(face_images) / 127.5 - 1),
            "t h w c -> 1 c t h w").to(dtype=torch.bfloat16).cpu()
        del cond_images, face_images
        torch.cuda.empty_cache()

        refer_pixel_values = rearrange(
            torch.tensor(refer_images / 127.5 - 1), "h w c -> 1 c h w"
        ).to(self.device, dtype=torch.bfloat16)
        refer_pixel_values = rearrange(refer_pixel_values, "t c h w -> 1 c t h w")

        ref_latents = torch.stack(self.vae.encode(refer_pixel_values.to(torch.bfloat16)))

        # Pixel frames per streaming step. The VAE's temporal stride is 4 and the
        # first latent frame absorbs one extra pixel frame, so chunk 0 covers 9
        # pixel frames (-> 3 latent frames) and every later chunk covers 12.
        pixel_chunks = [9] + [12] * ((clip_len - 1) // 12)
        assert sum(pixel_chunks) == clip_len, (
            f"clip_len={clip_len} does not tile into 9 + 12*n pixel chunks")

        # Reset all streaming state so this clip starts clean.
        self.vae.model.first_encode = True
        self.vae.model.first_decode = True
        self.noise_model.face_encoder.reset_streaming_state()

        # Pick the decode path BEFORE warmup: the two decoders compile to different
        # guards, and warmup only primes the one that is active. The Flash-VAED
        # weights load lazily on first enable.
        if hasattr(self.vae, "set_fast_decode"):
            self.vae.set_fast_decode(fast_decode)
            if fast_decode:
                logger.info('decoding with the Flash-VAED lightweight decoder')

        B, C, F, H, W = refer_pixel_values.shape
        T = clip_len
        lat_h, lat_w = H // 8, W // 8

        lat_t = T // 4 + 1
        assert lat_t % self.chunk_size == 0, (
            "lat_t (clip_len // 4 + 1) must be divisible by chunk_size "
            f"({lat_t} % {self.chunk_size} != 0)")
        chunk_num = lat_t // self.chunk_size

        noise = [
            torch.randn(16, lat_t, lat_h, lat_w,
                        dtype=torch.float32, device=self.device, generator=seed_g)
        ]

        # Token count for one chunk after 2x2 spatial patching.
        max_seq_len = (self.chunk_size * lat_h * lat_w) // 4

        # Reference conditioning: an all-ones mask over the single reference latent.
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=self.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=self.device)

        # Per-chunk conditioning slots: 4 latent frames of zeros, masked off. Chunk 0
        # reads y[:, 0:3] (nothing carried in), later chunks y[:, 1:4].
        y_reft = self.vae.encode([torch.zeros(3, 13, H, W).to(self.device)])[0]
        msk_reft = self.get_i2v_mask(4, lat_h, lat_w, 0, device=self.device)
        msk_reft_one = self.get_i2v_mask(3, lat_h, lat_w, 3, device=self.device)
        y = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=self.device)

        # 2x2 spatial patching -> a quarter of the latent tokens per frame.
        self.frame_seq_length = lat_h * lat_w // 4

        if self.kv_cache is None:
            self._initialize_kv_cache(dtype=torch.bfloat16, device=self.device)
            self._initialize_crossattn_cache()

        img = refer_pixel_values[0, :, 0]
        self.clip.to(self.device)
        clip_context = self.clip([img[:, None, :, :]]).to(
            dtype=torch.bfloat16, device=self.device)
        if offload_model:
            self.clip.cpu()
            torch.cuda.empty_cache()

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True):
            # Prime the cache with the clean reference frame. It sits at the head of
            # the cache and survives every update_kv() trim, so identity stays
            # anchored no matter how far the clip runs.
            timestep = torch.zeros([1], device=self.device, dtype=torch.int64)
            self.noise_model.forward_ref(
                [ref_latents[0]],
                t=timestep,
                context=context_default,
                seq_len=self.frame_seq_length,
                clip_fea=clip_context,
                y=[y_ref],
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                is_update=True)

            # forward_ref cached "Best quality."; drop it so the real prompt is
            # projected on the first chunk.
            self.reset_crossattn_cache()

            frames_written = 0
            pixel_offset = 0
            with tqdm(total=chunk_num) as progress_bar:
                for chunk_idx in range(chunk_num):
                    l = chunk_idx * self.chunk_size
                    r = (chunk_idx + 1) * self.chunk_size

                    # --- stream-encode this chunk's pose frames ---
                    pixel_chunk_size = pixel_chunks[chunk_idx]
                    cond_chunk = conditioning_pixel_values[
                        :, :, pixel_offset:pixel_offset + pixel_chunk_size
                    ].contiguous().to(self.device)
                    pose_latents_chunk = self.vae.stream_encode([cond_chunk[0]])[0].unsqueeze(0)

                    # --- stream-encode this chunk's face frames ---
                    face_chunk = face_pixel_values[
                        :, :, pixel_offset:pixel_offset + pixel_chunk_size
                    ].contiguous().to(self.device)
                    face_chunk_flat = rearrange(face_chunk, "b c t h w -> (b t) c h w")
                    encode_bs = 12
                    motion_feats = []
                    for i in range(math.ceil(face_chunk_flat.shape[0] / encode_bs)):
                        motion_feats.append(self.noise_model.motion_encoder.get_motion(
                            face_chunk_flat[i * encode_bs:(i + 1) * encode_bs]))
                    motion_feat = torch.cat(motion_feats).to(face_pixel_values.dtype)
                    motion_feat = rearrange(motion_feat, "(b t) c -> b t c", t=pixel_chunk_size)
                    motion_vec_chunk = self.noise_model.face_encoder.forward_streaming(motion_feat)

                    pixel_offset += pixel_chunk_size
                    latents_chunk = [noise[0][:, l:r].contiguous()]
                    y_chunk = (
                        y[:, 0:3] if chunk_idx == 0 else y[:, 1:4]
                    ).contiguous()

                    sample_scheduler, timesteps = self._build_scheduler(
                        sample_solver, sampling_steps, custom_timesteps, shift)

                    arg_c = {
                        "context": context,
                        "seq_len": max_seq_len,
                        "clip_fea": clip_context,
                        "y": [y_chunk],
                        "pose_latents": pose_latents_chunk,
                        "motion_vec": motion_vec_chunk,
                        "kv_cache": self.kv_cache,
                        "crossattn_cache": self.crossattn_cache,
                        "is_causal": True,
                    }

                    for t in timesteps:
                        timestep = torch.stack([t])

                        # Commit to the cache on the LAST step only, and never for
                        # chunk 0 (it gets a dedicated clean-latent pass below) or
                        # the final chunk (nothing reads the cache after it).
                        is_update = bool(t == timesteps[-1]) and (0 < chunk_idx < chunk_num - 1)

                        noise_pred = self.noise_model(
                            latents_chunk, t=timestep, is_update=is_update, **arg_c)

                        temp_x0 = sample_scheduler.step(
                            noise_pred[0].unsqueeze(0),
                            t,
                            latents_chunk[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g,
                        )[0]
                        latents_chunk[0] = temp_x0.squeeze(0)

                    x0 = latents_chunk

                    if chunk_idx == 0:
                        # Chunk 0 seeds the whole clip, so re-run it at t=0 on its
                        # own denoised (clean) latents and commit those instead of
                        # the noisy last-step keys every other chunk contributes.
                        latent_model_input = latents_chunk[0]
                        y_chunk_clean = torch.concat(
                            [msk_reft_one, latent_model_input]
                        ).to(dtype=torch.bfloat16, device=self.device)
                        arg_c["y"] = [y_chunk_clean]
                        self.noise_model(
                            [latent_model_input],
                            t=torch.stack([torch.zeros_like(timesteps[-1])]),
                            is_update=True, **arg_c)

                    self.update_kv()

                    # --- stream-decode this chunk to pixels ---
                    # fast_decode is dispatched inside the VAE: stream_decode picks
                    # the original decoder or Flash-VAED from self.vae.model.fast_decode,
                    # and both share the same feat_cache protocol.
                    chunk_latent = x0[0].to(dtype=torch.float32)
                    chunk_video = self.vae.stream_decode([chunk_latent])[0]
                    sink(_packed_frames(chunk_video).numpy())
                    frames_written += chunk_video.shape[1]

                    progress_bar.update()


        return frames_written, H, W


def _pixel_values(frames, device=None):
    if isinstance(frames, torch.Tensor):
        return frames.to(dtype=torch.bfloat16)
    if device is not None:
        pixels = torch.from_numpy(np.ascontiguousarray(frames)).to(device).float()
        pixels.sub_(127.5).div_(127.5)
        return rearrange(pixels, "t h w c -> 1 c t h w").to(dtype=torch.bfloat16)
    return rearrange(torch.tensor(np.stack(frames) / 127.5 - 1),
                     "t h w c -> 1 c t h w").to(dtype=torch.bfloat16).cpu()


def _display_frames(video):
    """Return pinned uint8 frames and their asynchronous copy event."""
    with torch.cuda.device(video.device):
        pixels = video.permute(1, 2, 3, 0).float().add(1).mul(127.5).clamp(0, 255).byte()
        frames = torch.empty(pixels.shape, dtype=torch.uint8, pin_memory=True)
        frames.copy_(pixels, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record()
    return frames, ready


class StreamingSession:
    def __init__(self, pipeline, ref_image, prompt, seed, generator, fast_decode,
                 offload_model, shift, sample_solver, sampling_steps, custom_timesteps, prepared=None):
        self.pipeline = pipeline
        self.closed = False
        self.chunk_index = 0
        self.generator = generator or torch.Generator(device=pipeline.device)
        if generator is None:
            self.generator.manual_seed(seed if seed >= 0 else torch.seed() % (2 ** 31))
        self.scheduler_args = (sample_solver, sampling_steps, custom_timesteps, shift)
        p = pipeline
        ref_image = np.asarray(ref_image)
        if ref_image.ndim != 3 or ref_image.shape[-1] != 3:
            raise ValueError("Reference must be an RGB image.")
        self.height, self.width = ref_image.shape[:2]
        if min(self.height, self.width) <= 0 or self.height % 16 or self.width % 16:
            raise ValueError("Reference dimensions must be positive multiples of 16.")
        self.noise_shape = (16, p.chunk_size, self.height // 8, self.width // 8)
        if prepared is not None:
            if prepared["key"] != (self.height, self.width, fast_decode) or prepared["prompt"] != prompt:
                raise ValueError("Prepared inputs do not match this session.")
            self.context, context_default = prepared["context"], prepared["context_default"]
        else:
            if not p.t5_cpu:
                p.text_encoder.to(p.device)
            self.context = p.encode_prompt([prompt])
            context_default = p.encode_prompt(["Best quality."])
            if offload_model and not p.t5_cpu:
                p.text_encoder.cpu()
                torch.cuda.empty_cache()
        ref_pixels = rearrange(torch.tensor(ref_image / 127.5 - 1),
                               "h w c -> 1 c 1 h w").to(p.device, dtype=torch.bfloat16)
        ref_latents = torch.stack(p.vae.encode(ref_pixels.to(p.vae_device))).to(p.device)
        p.vae.model.first_encode = True
        p.vae.model.first_decode = True
        p.noise_model.face_encoder.reset_streaming_state()
        p.vae.set_fast_decode(fast_decode)
        lat_h, lat_w = self.height // 8, self.width // 8
        mask_ref = p.get_i2v_mask(1, lat_h, lat_w, 1, device=p.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=p.device)
        if prepared is not None:
            self.y, self.mask_clean = prepared["y"], prepared["mask_clean"]
        else:
            y_reft = p.vae.encode([torch.zeros(3, 13, self.height, self.width).to(p.vae_device)])[0].to(p.device)
            msk_reft = p.get_i2v_mask(4, lat_h, lat_w, 0, device=p.device)
            self.mask_clean = p.get_i2v_mask(3, lat_h, lat_w, 3, device=p.device)
            self.y = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=p.device)
        p.frame_seq_length = lat_h * lat_w // 4
        p._initialize_kv_cache(dtype=torch.bfloat16, device=p.device)
        p._initialize_crossattn_cache()
        p.clip.to(p.device)
        img = ref_pixels[0, :, 0]
        self.clip_context = p.clip([img[:, None, :, :]]).to(dtype=torch.bfloat16, device=p.device)
        if offload_model:
            p.clip.cpu()
            torch.cuda.empty_cache()
        with torch.autocast(device_type=p.device.type, dtype=torch.bfloat16):
            p.noise_model.forward_ref(
                [ref_latents[0]], t=torch.zeros([1], device=p.device, dtype=torch.int64),
                context=context_default, seq_len=p.frame_seq_length,
                clip_fea=self.clip_context, y=[y_ref], kv_cache=p.kv_cache,
                crossattn_cache=p.crossattn_cache, is_update=True)
        p.reset_crossattn_cache()

    @property
    def expected_frames(self):
        return 9 if self.chunk_index == 0 else 12

    @torch.no_grad()
    def encode_pose(self, frames, count):
        p = self.pipeline
        if self.closed:
            raise RuntimeError("Session is closed.")
        pose = _pixel_values(frames, p.vae_device).contiguous().to(p.vae_device)
        if tuple(pose.shape) != (1, 3, count, self.height, self.width):
            raise ValueError(f"Expected {count} pose frames at {self.width}x{self.height}.")
        with torch.autocast(device_type=p.device.type, dtype=torch.bfloat16):
            return p.vae.stream_encode([pose[0]])[0].unsqueeze(0)

    @torch.no_grad()
    def push(self, chunk, noise, display=False, pose_latents=None):
        if self.closed:
            raise RuntimeError("Session is closed.")
        p = self.pipeline
        count = self.expected_frames
        if pose_latents is None:
            pose, face = (_pixel_values(frames, p.device) for frames in chunk)
            if tuple(pose.shape) != (1, 3, count, self.height, self.width):
                raise ValueError(f"Expected {count} pose frames at {self.width}x{self.height}.")
        else:
            face = _pixel_values(chunk[1], p.device)
            if tuple(pose_latents.shape) != (1, *self.noise_shape):
                raise ValueError("Invalid encoded pose shape.")
        if face.ndim != 5 or tuple(face.shape[:3]) != (1, 3, count):
            raise ValueError(f"Expected {count} RGB face frames.")
        if tuple(noise.shape) != self.noise_shape:
            raise ValueError(f"Expected noise shape {self.noise_shape}.")
        with torch.autocast(device_type=p.device.type, dtype=torch.bfloat16):
            if pose_latents is None:
                cond_chunk = pose.contiguous().to(p.vae_device)
                pose_latents = p.vae.stream_encode([cond_chunk[0]])[0].unsqueeze(0).to(p.device)
            else:
                pose_latents = pose_latents.to(p.device)
            face_chunk = face.contiguous().to(p.device)
            flat_face = rearrange(face_chunk, "b c t h w -> (b t) c h w")
            motion_feats = [p.noise_model.motion_encoder.get_motion(flat_face[i:i + 12])
                            for i in range(0, len(flat_face), 12)]
            motion_feat = torch.cat(motion_feats).to(face.dtype)
            motion_feat = rearrange(motion_feat, "(b t) c -> b t c", t=count)
            motion = p.noise_model.face_encoder.forward_streaming(motion_feat)
            latents = [noise.contiguous().to(p.device, dtype=torch.float32)]
            y_chunk = (self.y[:, 0:3] if self.chunk_index == 0 else self.y[:, 1:4]).contiguous()
            scheduler, timesteps = p._build_scheduler(*self.scheduler_args)
            args = dict(context=self.context, seq_len=p.chunk_size * p.frame_seq_length,
                        clip_fea=self.clip_context, y=[y_chunk], pose_latents=pose_latents,
                        motion_vec=motion, kv_cache=p.kv_cache,
                        crossattn_cache=p.crossattn_cache, is_causal=True)
            for t in timesteps:
                is_update = bool(t == timesteps[-1]) and self.chunk_index > 0
                pred = p.noise_model(latents, t=torch.stack([t]), is_update=is_update, **args)
                latents[0] = scheduler.step(
                    pred[0].unsqueeze(0), t, latents[0].unsqueeze(0),
                    return_dict=False, generator=self.generator)[0].squeeze(0)
            if self.chunk_index == 0:
                args["y"] = [torch.concat([self.mask_clean, latents[0]]).to(dtype=torch.bfloat16, device=p.device)]
                p.noise_model([latents[0]], t=torch.stack([torch.zeros_like(timesteps[-1])]),
                              is_update=True, **args)
            p.update_kv()
            video = p.vae.stream_decode([latents[0].to(p.vae_device, dtype=torch.float32)])[0]
        self.chunk_index += 1
        return _display_frames(video) if display else video.cpu()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.pipeline._clear_stream_state()
        self.pipeline._active_session = None
        self.context = self.clip_context = self.y = self.mask_clean = None

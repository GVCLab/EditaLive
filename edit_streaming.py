import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from editalive.compile_config import configure_compile
from editalive.configs import editalive_14B
from editalive.utils.utils import merge_video_audio, str2bool, video_writer
from editalive.utils.validation import validate_resolution, validate_sampling


def _parse_args():
    parser = argparse.ArgumentParser(description="EditaLive streaming (causal) video editing inference")

    parser.add_argument("--ckpt_dir", type=str, default="./weights/Wan-Animate", help="Checkpoint directory.")
    parser.add_argument("--save_dir", type=str, default="./outputs_streaming", help="Output directory.")
    parser.add_argument("--fps", type=int, default=16, help="Output frame rate.")
    parser.add_argument("--keep_audio", type=str2bool, default=True, help="Copy the source video's audio onto the result (default: True).")

    # Inputs. Either a batch task file, or one sample straight from the command
    # line: --video (+ optional --ref_image), or --folder.
    parser.add_argument("--input_json", type=str, default=None, help="Batch task file; see demo/edit_causal.json.")
    parser.add_argument("--video", type=str, default=None, help="Source video for a single run.")
    parser.add_argument("--ref_image", type=str, default=None, help="Reference image; defaults to the video's first frame.")
    parser.add_argument("--folder", type=str, default=None, help="Preprocessed source directory for a single run.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt; also the fallback for task-file entries without one.")

    # Preprocessing
    parser.add_argument("--preprocess_ckpt_dir", type=str, default=None,
                        help="Preprocessing checkpoints; defaults to <ckpt_dir>/process_checkpoint.")
    parser.add_argument("--preprocess_output_dir", type=str, default=None,
                        help="Preprocessing cache; defaults to <save_dir>/preprocessed.")
    parser.add_argument("--preprocess_long_edge", type=int, default=832)
    parser.add_argument("--preprocess_short_edge", type=int, default=480)
    parser.add_argument("--preprocess_fps", type=int, default=16)
    parser.add_argument("--vitpose_batch_size", type=int, default=16)
    parser.add_argument("--smooth", type=str2bool, default=False, help="Temporally smooth pose and face crops.")
    parser.add_argument("--retarget", type=str2bool, default=False, help="Retarget the source-video pose onto the reference body proportions.")
    parser.add_argument("--force_preprocess", action="store_true", default=False, help="Ignore the preprocessing cache.")

    # Sampling
    parser.add_argument("--frame_num", type=int, default=-1, help="Frames per run; -1 uses the full source length.")
    parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'])
    parser.add_argument("--sample_steps", type=int, default=2, help="Denoising steps; used only when --custom_timesteps is empty.")
    parser.add_argument("--custom_timesteps", nargs='*', type=int, default=[1000, 250], help="Pre-shift timesteps for the distilled few-step model.")
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--base_seed", type=int, default=-1, help="Random seed.")
    parser.add_argument("--long_edge", type=int, default=832, help="Long edge of the output.")
    parser.add_argument("--short_edge", type=int, default=480, help="Short edge of the output.")

    # LoRA
    parser.add_argument("--lora_paths", nargs='+', default=None, help="LoRA safetensors to fuse, in order.")
    parser.add_argument("--lora_strengths", nargs='*', type=float, default=None,
                        help="Multiplier per LoRA: one shared value or one per path (default 1.0).")

    # Performance
    parser.add_argument("--enable_compile", type=str2bool, default=False,
                        help="Enable torch.compile and its warm-up (default: False).")
    parser.add_argument("--compile_cache_dir", type=str, default=None,
                        help="Keep the compilation cache here instead of the system temporary directory, "
                             "so later runs reuse it (~400 MB per configuration).")
    parser.add_argument("--fp8", action="store_true", default=False, help="Quantize the transformer to scaled FP8 (needs SM 8.9+).")
    parser.add_argument("--low_vram", type=str, default="off", choices=["off", "weights", "weights+kv"],
                        help="Trade speed for VRAM: 'weights' streams the transformer blocks from CPU memory, "
                             "'weights+kv' also keeps the rolling KV cache there.")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Keep the T5 text encoder on the CPU.")
    parser.add_argument("--offload_model", type=str2bool, default=True, help="Offload the text encoder to CPU after encoding.")
    parser.add_argument("--fast_decode", type=str2bool, default=False, help="Decode with the lightweight Flash-VAED decoder.")
    parser.add_argument("--fast_decoder_pth", type=str, default='./weights/Flash-VAED/Flash_VAED_Wan.pth', help="Flash-VAED decoder checkpoint; required with --fast_decode.")

    # Video Sparse Attention
    parser.add_argument("--vsa_sparsity", type=float, default=0.75, help="Fraction of KV blocks dropped in causal self-attention; 0 = dense.")
    parser.add_argument("--vsa_tile_size", nargs=3, type=int, default=(1, 8, 8), help="Spatio-temporal tile (t h w) for VSA.")
    parser.add_argument("--vsa_sink_frames", type=int, default=1, help="Leading cache frames always kept by VSA.")
    parser.add_argument("--vsa_backend", type=str, default="kernel", choices=["auto", "kernel", "sdpa"], help="Sparse attention backend.")

    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args):
    cfg = editalive_14B

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    if args.frame_num != -1 and (args.frame_num < 9 or (args.frame_num - 9) % 12 != 0):
        raise ValueError(f"--frame_num must be -1 (full length) or 9+12*n; got {args.frame_num}.")
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive; got {args.fps}.")
    if args.preprocess_fps != -1 and args.preprocess_fps <= 0:
        raise ValueError(f"--preprocess_fps must be -1 or positive; got {args.preprocess_fps}.")
    if args.vitpose_batch_size <= 0:
        raise ValueError("--vitpose_batch_size must be positive; " f"got {args.vitpose_batch_size}.")
    if args.preprocess_long_edge <= 0 or args.preprocess_short_edge <= 0:
        raise ValueError("--preprocess_long_edge and --preprocess_short_edge must be positive.")

    # An empty --custom_timesteps means "use a uniform --sample_steps schedule".
    if args.custom_timesteps is not None and len(args.custom_timesteps) == 0:
        args.custom_timesteps = None
    validate_resolution(args.long_edge, args.short_edge)
    validate_sampling(args.sample_steps, args.custom_timesteps, args.sample_shift,
                      cfg.num_train_timesteps)
    if not 0 <= args.vsa_sparsity <= 1:
        raise ValueError("--vsa_sparsity must be between 0 and 1.")
    if any(size <= 0 for size in args.vsa_tile_size):
        raise ValueError("--vsa_tile_size values must be positive.")
    if args.vsa_sink_frames < 0:
        raise ValueError("--vsa_sink_frames must be nonnegative.")

    # --lora_strengths: empty means 1.0 everywhere; otherwise one shared value or
    # one per --lora_paths entry.
    if args.lora_strengths is not None and len(args.lora_strengths) == 0:
        args.lora_strengths = None
    if args.lora_strengths is not None:
        n_lora = len(args.lora_paths or [])
        if not n_lora:
            raise ValueError("--lora_strengths requires --lora_paths.")
        if not all(math.isfinite(value) for value in args.lora_strengths):
            raise ValueError("--lora_strengths values must be finite.")
        if len(args.lora_strengths) not in (1, n_lora):
            raise ValueError(
                f"--lora_strengths takes 1 or {n_lora} values (one per "
                f"--lora_paths), got {len(args.lora_strengths)}.")

    if args.low_vram != "off" and args.enable_compile:
        raise ValueError("--low_vram cannot be combined with --enable_compile: streaming "
                         "the weights rebinds them every block, which invalidates the "
                         "compiled graphs.")

    # Fail before the 14B weights load rather than at the first decode.
    if args.fast_decode and not args.fast_decoder_pth:
        raise ValueError("--fast_decode requires --fast_decoder_pth "
                         "(path to the Flash-VAED checkpoint).")

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)

    # Preprocessing resamples the source video to --preprocess_fps while the
    # result is written at --fps, so the two must agree or the copied audio
    # drifts against a clip that no longer matches the source duration.
    if args.keep_audio and args.preprocess_fps != args.fps:
        logging.warning(
            "--keep_audio with --preprocess_fps %s and --fps %s: the result "
            "will not match the source video's duration, so the audio will "
            "drift. Set them equal to keep it in sync.",
            args.preprocess_fps, args.fps)

    modes = [bool(args.input_json), bool(args.video), bool(args.folder)]
    if sum(modes) != 1:
        raise ValueError(
            "Provide exactly one input: --input_json for a batch, or --video "
            "or --folder for a single run.")

    if args.ref_image and not args.video:
        raise ValueError("--ref_image is only valid with --video; use per-entry ref_image in JSON.")

    # Resolve every input up front: a typo should fail here, not after the 14B
    # weights have loaded. --prompt is the fallback in both modes.
    default_prompt = args.prompt or cfg.prompt
    if args.input_json:
        args.tasks = _load_tasks(args.input_json, default_prompt)
    else:
        args.tasks = [_make_task(
            {"video": args.video, "ref_image": args.ref_image, "folder": args.folder},
            "command line", default_prompt)]

    _check_one_orientation(args.tasks)


def _reference_size(task):
    """(width, height) of the reference a task will be generated against.

    The reference is the task's ``ref_image``, the preprocessed ``src_ref.png``,
    or -- when neither is given -- the source video's first frame, which is what
    preprocessing will copy in as the reference.
    """
    from PIL import Image

    from editalive.utils.preprocess import read_first_video_frame

    if task["ref_image"]:
        with Image.open(task["ref_image"]) as image:
            return image.size
    if task["folder"]:
        with Image.open(os.path.join(task["folder"], "src_ref.png")) as image:
            return image.size
    frame = read_first_video_frame(task["video"])
    return frame.shape[1], frame.shape[0]


def _check_one_orientation(tasks):
    """Reject a batch that mixes portrait and landscape references.

    One run produces one output size: the reference's orientation picks between
    ``long_edge x short_edge`` and its transpose. Mixing the two would change
    every tensor shape mid-run, which also means recompiling from scratch under
    --enable_compile. Run each orientation separately instead.
    """
    orientations = {}
    for task in tasks:
        width, height = _reference_size(task)
        orientations.setdefault(
            "landscape" if width >= height else "portrait", []).append(
                f'{task["id"]} ({width}x{height})')
    if len(orientations) > 1:
        raise ValueError(
            "All inputs in one run must share the same orientation; this run "
            "mixes " + "; ".join(f"{name}: {', '.join(ids)}"
                                 for name, ids in sorted(orientations.items()))
            + ". Split them into separate runs.")


def _init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])


SOURCE_FILES = ("src_pose.mp4", "src_face.mp4", "src_ref.png")


def _make_task(entry, where, default_prompt):
    """Validate one input description and normalise it into a generation job.

    An entry carries either a raw source video ("video", optionally with
    "ref_image") or an already preprocessed directory ("folder"). "id" names
    the output file and the preprocessing cache entry; "prompt" falls back to
    ``default_prompt``. The same shape backs a task-file entry and the
    single-run command line, so both report identical errors.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be an object.")

    prompt = entry.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError(f'{where}: "prompt" must be a string.')
    video = entry.get("video")
    folder = entry.get("folder")
    ref_image = entry.get("ref_image")
    if bool(video) == bool(folder):
        raise ValueError(f'{where} needs exactly one of "video" or "folder".')
    if ref_image and not video:
        raise ValueError(f'{where} has "ref_image" but no "video".')

    if video:
        if not os.path.isfile(video):
            raise FileNotFoundError(f"{where}: video does not exist: {video}")
        if ref_image and not os.path.isfile(ref_image):
            raise FileNotFoundError(f"{where}: ref_image does not exist: {ref_image}")
        fallback_id = os.path.splitext(os.path.basename(video))[0]
    else:
        missing = [name for name in SOURCE_FILES
                   if not os.path.isfile(os.path.join(folder, name))]
        if missing:
            raise FileNotFoundError(
                f"{where}: folder is missing {', '.join(missing)}: {folder}")
        fallback_id = os.path.basename(folder.rstrip('/'))

    # The id becomes a directory and a file name, so keep it path-safe.
    task_id = re.sub(r"[^A-Za-z0-9._-]+", "_",
                     str(entry.get("id") or fallback_id)).strip("._")

    return {
        "id": task_id or "task",
        "video": video,
        "ref_image": ref_image,
        "folder": folder,
        "prompt": entry.get("prompt") or default_prompt,
    }


def _add_source_audio(task, save_path):
    """Copy the source video's audio track onto a generated clip.

    ffmpeg takes the audio from the second input, so the source video is
    handed to ``merge_video_audio`` directly; it copies the video stream and
    trims to whichever input is shorter.
    """
    source = task["video"]
    if not source:
        logging.info(
            "%s: a preprocessed folder has no source audio; saved without "
            "audio.", task["id"])
        return

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", source],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if probe.returncode != 0:
        raise RuntimeError(f"Failed to inspect source audio: {probe.stderr}")
    if "audio" not in probe.stdout:
        logging.warning("%s: %s has no audio track. Saved without audio.",
                        task["id"], source)
        return

    merge_video_audio(save_path, source)


def _save_comparison(task, generated_path, comparison_path, fps, width, height):
    """Save the source and edited videos side by side at the output size."""
    source = task["video"]
    if not source:
        logging.info(
            "%s: no source comparison for a preprocessed folder.", task["id"])
        return

    # Match the exact frames used by inference: resample the raw source to the
    # output fps, fit both streams without cropping, and stop at the shorter one.
    fit = (
        f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    filter_complex = (
        f"[0:v:0]{fit}[source];"
        f"[1:v:0]{fit}[edited];"
        "[source][edited]hstack=inputs=2:shortest=1[comparison]")
    command = [
        "ffmpeg", "-y",
        "-i", source,
        "-i", generated_path,
        "-filter_complex", filter_complex,
        "-map", "[comparison]",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        comparison_path,
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        if os.path.exists(comparison_path):
            os.remove(comparison_path)
        raise RuntimeError(
            f"Failed to save comparison video: {result.stderr}")

    logging.info("Saved source comparison to %s", comparison_path)


def _load_tasks(input_json, default_prompt):
    """Read the JSON task file into a list of generation jobs, one per entry."""
    if not os.path.isfile(input_json):
        raise FileNotFoundError(f"--input_json does not exist: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{input_json} must hold a non-empty list of entries.")

    tasks = []
    seen_ids = set()
    for index, entry in enumerate(entries):
        where = f"{input_json} entry {index}"
        task = _make_task(entry, where, default_prompt)
        if task["id"] in seen_ids:
            raise ValueError(f'{where}: duplicate id "{task["id"]}".')
        seen_ids.add(task["id"])
        tasks.append(task)
    return tasks


def _check_runtime_files(args):
    cfg = editalive_14B
    required = [os.path.join(args.ckpt_dir, name) for name in (
        cfg.vae_checkpoint, cfg.clip_checkpoint, cfg.t5_checkpoint)]
    transformer_dir = os.path.join(args.ckpt_dir, cfg.transformer_subpath)
    required.append(os.path.join(transformer_dir, "config.json"))
    for filename in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"):
        checkpoint = os.path.join(transformer_dir, filename)
        index = checkpoint + ".index.json"
        if os.path.isfile(checkpoint):
            required.append(checkpoint)
            break
        if os.path.isfile(index):
            with open(index, encoding="utf-8") as file:
                shards = set(json.load(file)["weight_map"].values())
            if not shards:
                raise ValueError(f"Checkpoint index has no shards: {index}")
            required.extend(os.path.join(transformer_dir, shard) for shard in sorted(shards))
            break
    else:
        required.append(os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors"))
    required.extend(args.lora_paths or [])
    if args.fast_decode:
        required.append(args.fast_decoder_pth)
    missing = [path for path in required if not os.path.isfile(path)]
    tokenizer_dir = os.path.join(args.ckpt_dir, cfg.t5_tokenizer)
    if not os.path.isdir(tokenizer_dir):
        missing.append(tokenizer_dir)
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))
    if any(task["video"] for task in args.tasks):
        commands = ["ffmpeg"] + (["ffprobe"] if args.keep_audio else [])
        for command in commands:
            if shutil.which(command) is None:
                raise RuntimeError(f"{command} is required for raw-video output; install FFmpeg.")


def generate(args):
    _check_runtime_files(args)
    if args.compile_cache_dir:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.abspath(args.compile_cache_dir)

    configure_compile(args.enable_compile)
    from editalive.utils.preprocess import preprocess_source

    _init_logging()
    logging.info("Seed: %d", args.base_seed)
    if not args.lora_paths:
        logging.warning(
            "No --lora_paths given: the base Wan-Animate weights are not "
            "distilled for this few-step streaming schedule, so results will be "
            "poor. Pass the three EditaLive LoRAs (see README).")

    # Single-GPU: pick the GPU with CUDA_VISIBLE_DEVICES.
    device = torch.device("cuda:0")

    cfg = editalive_14B

    tasks = args.tasks

    # Preprocess every raw source video before loading the large inference
    # pipeline, so all preprocessing CUDA contexts are released first.
    for task in tasks:
        if task["video"]:
            task["folder"] = preprocess_source(
                driving_video=task["video"],
                reference_image=task["ref_image"],
                source_name=task["id"],
                inference_ckpt_dir=args.ckpt_dir,
                save_dir=args.save_dir,
                preprocess_ckpt_dir=args.preprocess_ckpt_dir,
                preprocess_output_dir=args.preprocess_output_dir,
                long_edge=args.preprocess_long_edge,
                short_edge=args.preprocess_short_edge,
                fps=args.preprocess_fps,
                vitpose_batch_size=args.vitpose_batch_size,
                smooth=args.smooth,
                retarget=args.retarget,
                force=args.force_preprocess,
            )
            print(f"Preprocessing finished for {task['id']}: {task['folder']}")

    # Delay CUDA context creation in this parent process until the preprocessing
    # subprocess has exited, so the preprocessing stage gets the full device.
    torch.cuda.set_device(device)
    from editalive.pipeline import EditaLiveStreamingPipeline
    logging.info("Creating the EditaLive streaming pipeline (weights load once) ...")
    pipeline = EditaLiveStreamingPipeline.from_pretrained(
        model_path=args.ckpt_dir,
        config=cfg,
        device=device,
        weight_dtype=cfg.param_dtype,
        t5_cpu=args.t5_cpu,
        lora_paths=args.lora_paths,
        lora_strengths=args.lora_strengths,
        fp8=args.fp8,
        low_vram=args.low_vram,
        fast_decode=args.fast_decode,
        vsa_sparsity=args.vsa_sparsity,
        vsa_backend=args.vsa_backend,
        vsa_tile_size=tuple(args.vsa_tile_size),
        vsa_sink_frames=args.vsa_sink_frames,
        fast_decoder_pth=args.fast_decoder_pth,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    pipeline.warmup(
        src_root_path=tasks[0]["folder"],
        fast_decode=args.fast_decode,
        long_edge=args.long_edge,
        short_edge=args.short_edge,
    )

    for index, task in enumerate(tasks):
        logging.info("[%d/%d] %s | %s", index + 1, len(tasks),
                     task["id"], task["prompt"])

        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(args.save_dir, f"{task['id']}_{formatted_time}.mp4")
        logging.info("Writing generated video to %s", save_path)
        with video_writer(save_path, fps=args.fps, crf=20) as sink:
            frames, height, width = pipeline.generate(
                src_root_path=task["folder"],
                sink=sink,
                input_prompt=task["prompt"],
                clip_len=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                custom_timesteps=args.custom_timesteps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
                fast_decode=args.fast_decode,
                long_edge=args.long_edge,
                short_edge=args.short_edge,
            )
        logging.info("Wrote %d frames at %dx%d", frames, width, height)

        pipeline.clean_cache()

        if args.keep_audio:
            _add_source_audio(task, save_path)

        _save_comparison(
            task=task,
            generated_path=save_path,
            comparison_path=os.path.join(
                args.save_dir, f"{task['id']}_{formatted_time}_comparison.mp4"),
            fps=args.fps,
            width=width,
            height=height)

        torch.cuda.empty_cache()

    logging.info("All tasks finished processing.")


if __name__ == "__main__":
    generate(_parse_args())

import json
import logging
import os
import re
import subprocess
import sys
import tempfile


def _file_stat(path):
    """Size and mtime, enough to notice that a file was replaced or edited."""
    stat = os.stat(path)
    return [stat.st_size, stat.st_mtime_ns]


def _read_settings(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_first_video_frame(video_path):
    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"Failed to read the first frame from: {video_path}")
    return frame


PREPROCESS_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "preprocess", "preprocess_data.py")


def preprocess_source(
    driving_video,
    inference_ckpt_dir,
    save_dir,
    reference_image=None,
    preprocess_ckpt_dir=None,
    preprocess_output_dir=None,
    source_name=None,
    long_edge=832,
    short_edge=480,
    fps=-1,
    vitpose_batch_size=16,
    smooth=False,
    retarget=False,
    force=False,
    python_executable=None,
):
    """Create ``src_pose``, ``src_face`` and ``src_ref`` from raw user input.

    The external Wan-Animate preprocessor is intentionally run in a subprocess.
    Its pose models and CUDA context are consequently released before the large
    EditaLive inference model is loaded.

    Returns:
        Path to a source directory accepted by ``EditaLiveStreamingPipeline``.
    """
    video_path = os.path.realpath(driving_video)
    reference_path = (os.path.realpath(reference_image)
                      if reference_image else None)
    preprocess_ckpt_dir = os.path.realpath(
        preprocess_ckpt_dir
        or os.path.join(inference_ckpt_dir, "process_checkpoint"))

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Source video does not exist: {video_path}")
    if reference_path and not os.path.isfile(reference_path):
        raise FileNotFoundError(f"Reference image does not exist: {reference_path}")
    if not os.path.isfile(PREPROCESS_SCRIPT):
        raise FileNotFoundError(
            "The bundled Wan-Animate preprocessing script is missing from this "
            f"checkout: {PREPROCESS_SCRIPT}")
    if not os.path.isdir(preprocess_ckpt_dir):
        raise FileNotFoundError(
            "Wan-Animate preprocessing checkpoint directory was not found. Pass "
            f"preprocess_ckpt_dir explicitly: {preprocess_ckpt_dir}")
    if fps != -1 and fps <= 0:
        raise ValueError(f"Preprocessing fps must be -1 or positive; got {fps}.")
    if (not isinstance(vitpose_batch_size, int)
            or isinstance(vitpose_batch_size, bool)
            or vitpose_batch_size <= 0):
        raise ValueError(
            "ViTPose batch size must be a positive integer; "
            f"got {vitpose_batch_size}.")
    if long_edge <= 0 or short_edge <= 0:
        raise ValueError(
            "Preprocessing long_edge and short_edge must be positive; "
            f"got {long_edge} and {short_edge}.")

    cache_root = os.path.realpath(
        preprocess_output_dir or os.path.join(save_dir, "preprocessed"))
    source_stem = source_name or os.path.splitext(os.path.basename(video_path))[0]
    source_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_stem).strip("._")
    source_dir = os.path.join(cache_root, source_stem or "driving_video")
    os.makedirs(source_dir, exist_ok=True)

    expected_outputs = [
        os.path.join(source_dir, "src_pose.mp4"),
        os.path.join(source_dir, "src_face.mp4"),
        os.path.join(source_dir, "src_ref.png"),
    ]

    # Everything that shapes the outputs. A cache entry is reused only if a
    # completed run with these exact inputs and settings wrote it, so changing
    # e.g. --ref_image or --retarget re-runs preprocessing instead of silently
    # returning the old files, and a run killed mid-write is redone.
    settings = {
        "video": video_path,
        "video_stat": _file_stat(video_path),
        "reference_image": reference_path,
        "reference_stat": _file_stat(reference_path) if reference_path else None,
        "long_edge": long_edge,
        "short_edge": short_edge,
        "fps": fps,
        "smooth": bool(smooth),
        "retarget": bool(retarget),
    }
    settings_path = os.path.join(source_dir, "preprocess_settings.json")

    if (not force and _read_settings(settings_path) == settings
            and all(os.path.isfile(path) and os.path.getsize(path) > 0
                    for path in expected_outputs)):
        logging.info("Reusing preprocessed source: %s", source_dir)
        return source_dir

    # Invalidate first, so an interrupted run leaves no reusable entry behind.
    if os.path.exists(settings_path):
        os.remove(settings_path)

    with tempfile.TemporaryDirectory() as scratch_dir:
        if reference_path is None:
            import cv2

            frame = read_first_video_frame(video_path)
            reference_path = os.path.join(scratch_dir, "driving_first_frame.png")
            if not cv2.imwrite(reference_path, frame):
                raise OSError(
                    f"Failed to save the source video's first frame: {reference_path}")
            logging.info("Using the source video's first frame as reference.")
        else:
            logging.info("Using reference image: %s", reference_path)

        command = [
            python_executable or sys.executable,
            PREPROCESS_SCRIPT,
            "--ckpt_path", preprocess_ckpt_dir,
            "--video_path", video_path,
            "--refer_path", reference_path,
            "--save_path", source_dir,
            "--resolution_area", str(long_edge), str(short_edge),
            "--fps", str(fps),
            "--vitpose_batch_size", str(vitpose_batch_size),
        ]
        if smooth:
            command.append("--smooth")
        if retarget:
            command.append("--retarget_flag")
        logging.info("Preprocessing source video before model loading: %s", video_path)
        subprocess.run(command, cwd=os.path.dirname(PREPROCESS_SCRIPT), check=True)

    missing = [path for path in expected_outputs
               if not os.path.isfile(path) or os.path.getsize(path) == 0]
    if missing:
        raise RuntimeError(
            "Preprocessing completed without all required outputs: " + ", ".join(missing))

    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    logging.info("Preprocessing completed: %s", source_dir)
    return source_dir

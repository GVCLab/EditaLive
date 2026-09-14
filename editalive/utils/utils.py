import argparse
import os
import subprocess
import tempfile
from contextlib import contextmanager

import imageio


@contextmanager
def video_writer(save_file, fps=30, crf=20):
    """Open an mp4 and stream chunks of uint8 HWC frames into it.

    Writing each chunk as it is decoded keeps host memory flat. Holding the whole
    clip instead costs four times the bytes as float32, and encoding it needs three
    more copies of that -- tens of GiB on a long source.
    """
    writer = imageio.get_writer(save_file, fps=fps, codec='libx264',
                               quality=None, output_params=['-crf', str(crf)])

    def write(frames):
        for frame in frames:
            writer.append_data(frame)

    try:
        yield write
    except BaseException:
        # A truncated mp4 sitting next to the finished ones would look like a result.
        writer.close()
        if os.path.exists(save_file):
            os.remove(save_file)
        raise
    writer.close()


def str2bool(v):
    """argparse helper: accept yes/true/t/y/1 and no/false/f/n/0 (case-insensitive)."""
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')


def merge_video_audio(video_path: str, audio_path: str):
    """Mux source audio into a video, preserving the original if FFmpeg fails."""
    for path in (video_path, audio_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    base, ext = os.path.splitext(os.path.abspath(video_path))
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(base)}.",
                                      suffix=ext, dir=os.path.dirname(base))
    os.close(fd)
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', video_path, '-i', audio_path,
             '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
             '-map', '0:v:0', '-map', '1:a:0', '-shortest', temporary],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to merge source audio: {result.stderr}")
        os.replace(temporary, video_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)

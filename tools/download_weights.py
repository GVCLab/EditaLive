#!/usr/bin/env python3
"""Download the checkpoints required by EditaLive."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights"

WAN_REPO = "Wan-AI/Wan2.2-Animate-14B"
EDITALIVE_REPO = "huaichang/EditaLive"
FLASH_VAED_REPO = "Aoko955/Flash-VAED"

EDITALIVE_FILES = (
    "editalive_edit.safetensors",
    "editalive_streaming.safetensors",
    "lightx2v.safetensors",
)


def _place_cached_file(cached_path: Path, destination: Path) -> None:
    """Hard-link a cached file when possible, otherwise copy it atomically."""
    cached_path = cached_path.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.incomplete")
    temporary_path.unlink(missing_ok=True)

    try:
        os.link(cached_path, temporary_path)
    except OSError:
        shutil.copy2(cached_path, temporary_path)

    temporary_path.replace(destination)


def _download_file(repo_id: str, filename: str, destination: Path) -> None:
    if destination.exists():
        if not destination.is_file():
            raise IsADirectoryError(f"Expected a file at {destination}")
        print(f"[skip] {destination}")
        return

    print(f"[download] {repo_id}/{filename}")
    cached_path = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    _place_cached_file(cached_path, destination)


# Parts of the Wan-Animate repo that EditaLive never reads (~18 GB): the
# character-replacement relighting LoRA and SAM2 segmenter, and the full
# xlm-roberta-large model (the CLIP encoder builds its own text tower).
WAN_IGNORE_PATTERNS = [
    "relighting_lora/*",
    "relighting_lora.ckpt",
    "process_checkpoint/sam2/*",
    "xlm-roberta-large/*",
]


def download_wan_animate(weights_dir: Path) -> None:
    destination = weights_dir / "Wan-Animate"
    print(f"[download] {WAN_REPO} -> {destination}")
    snapshot_download(repo_id=WAN_REPO, local_dir=str(destination),
                      ignore_patterns=WAN_IGNORE_PATTERNS)


def download_editalive_loras(weights_dir: Path) -> None:
    destination = weights_dir / "EditaLive"
    for filename in EDITALIVE_FILES:
        _download_file(
            repo_id=EDITALIVE_REPO,
            filename=f"weights/{filename}",
            destination=destination / filename,
        )


def download_flash_vaed(weights_dir: Path) -> None:
    _download_file(
        repo_id=FLASH_VAED_REPO,
        filename="models/wan/Flash_VAED_Wan.pth",
        destination=weights_dir / "Flash-VAED" / "Flash_VAED_Wan.pth",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Wan-Animate, EditaLive LoRAs, and Flash-VAED."
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help=f"Output directory (default: {DEFAULT_WEIGHTS_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_dir = args.weights_dir.expanduser().resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    download_wan_animate(weights_dir)
    download_editalive_loras(weights_dir)
    download_flash_vaed(weights_dir)

    print(f"All checkpoints are ready under {weights_dir}")


if __name__ == "__main__":
    main()

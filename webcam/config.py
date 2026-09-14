from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool

ROOT = Path(__file__).resolve().parents[1]


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resolution: Literal[384, 480] = 384
    orientation: Literal["landscape"] = "landscape"
    fp8: StrictBool = True
    decoder: Literal["flash", "original"] = "flash"
    dual_gpu: StrictBool = False

    @property
    def size(self):
        short, long = self.resolution, {384: 672, 480: 832}[self.resolution]
        return long, short


class ModelPaths(BaseModel):
    checkpoint: str = str(ROOT / "weights/Wan-Animate")
    loras: list[str] = [str(ROOT / "weights/EditaLive" / f) for f in (
        "editalive_edit.safetensors", "lightx2v.safetensors", "editalive_streaming.safetensors")]
    fast_decoder: str = str(ROOT / "weights/Flash-VAED/Flash_VAED_Wan.pth")
    compile_cache: str = str(ROOT / ".cache/compile/webcam")
    device: int = 0
    second_device: int | None = None
    compile: bool = True

    def resolve_devices(self, config):
        """(transformer card, auxiliary card) -- pose extraction and the VAE use the second."""
        if not config.dual_gpu:
            return self.device, self.device
        aux = self.device + 1 if self.second_device is None else self.second_device
        if aux == self.device:
            raise ValueError("Dual GPU mode needs a card other than --device.")
        return self.device, aux

    def validate_files(self, config):
        paths = [Path(self.checkpoint), *map(Path, self.loras),
                 Path(self.checkpoint) / "process_checkpoint/det/yolov10m.onnx",
                 Path(self.checkpoint) / "process_checkpoint/pose2d/vitpose_h_wholebody.onnx"]
        if config.decoder == "flash":
            paths.append(Path(self.fast_decoder))
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise ValueError("Missing weights: " + ", ".join(missing))

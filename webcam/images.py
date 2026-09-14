import io
import math

import cv2
import numpy as np
from PIL import Image, ImageOps


def decode_image(data):
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 32_000_000:
            raise ValueError("Image is too large.")
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def crop_reference(data, crop, size):
    image = decode_image(data)
    height, width = image.shape[:2]
    if not isinstance(crop, dict):
        raise ValueError("Reference crop is required.")
    values = [crop.get(k) for k in ("x", "y", "w", "h")]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Invalid crop coordinates.")
    x, y, w, h = values
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.00001 or y + h > 1.00001:
        raise ValueError("Crop is outside the image.")
    x0, y0 = int(x * width + 0.5), int(y * height + 0.5)
    cw, ch = int(w * width + 0.5), int(h * height + 0.5)
    cw, ch = min(cw, width - x0), min(ch, height - y0)
    if min(cw, ch) < 2 or abs(cw / ch - size[0] / size[1]) > 2 / ch:
        raise ValueError("Crop must match the output aspect ratio.")
    return cv2.resize(image[y0:y0 + ch, x0:x0 + cw], size, interpolation=cv2.INTER_LINEAR)


def encode_jpeg(image, quality=90):
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed.")
    return data.tobytes()

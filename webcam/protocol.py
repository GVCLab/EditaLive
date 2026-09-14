import json
import struct

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 8192


def pack(header, payload):
    data = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    return struct.pack("!I", len(data)) + data + payload


def unpack(data):
    if len(data) < 5 or len(data) > MAX_IMAGE_BYTES + MAX_HEADER_BYTES + 4:
        raise ValueError("Invalid image message size.")
    size = struct.unpack("!I", data[:4])[0]
    if not 0 < size <= MAX_HEADER_BYTES or 4 + size >= len(data):
        raise ValueError("Invalid image header.")
    header = json.loads(data[4:4 + size])
    if not isinstance(header, dict):
        raise ValueError("Image header must be an object.")
    return header, data[4 + size:]

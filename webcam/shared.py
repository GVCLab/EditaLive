import numpy as np

from .flow import CHUNK


class SharedFrames:
    """Two input slots shared by extraction and generation."""

    slots = 2

    def __init__(self, context, size):
        width, height = size
        self.pose_shape = (CHUNK, height, width, 3)
        self.face_shape = (CHUNK, 512, 512, 3)
        self.pose_bytes = int(np.prod(self.pose_shape))
        self.slot_bytes = self.pose_bytes + int(np.prod(self.face_shape))
        self.storage = context.RawArray("B", self.slots * self.slot_bytes)

    def read(self, slot, count):
        if not 0 <= slot < self.slots or count not in (9, CHUNK):
            raise ValueError("Invalid shared input slot or frame count.")
        offset = slot * self.slot_bytes
        pose = np.ndarray(self.pose_shape, np.uint8, self.storage, offset=offset)
        face = np.ndarray(self.face_shape, np.uint8, self.storage, offset=offset + self.pose_bytes)
        return pose[:count], face[:count]

    def write(self, slot, pose, face):
        targets = self.read(slot, len(pose))
        for target, value in zip(targets, (pose, face)):
            if value.shape != target.shape or value.dtype != np.uint8:
                raise ValueError("Invalid extracted frame shape or dtype.")
        for target, value in zip(targets, (pose, face)):
            np.copyto(target, value)

import sys
from pathlib import Path

import torch
import cv2
import numpy as np

from .flow import CHUNK
from .images import decode_image


class OnlineExtractor:
    def __init__(self, checkpoint, device, size):
        directory = Path(__file__).resolve().parents[1] / "editalive/utils/preprocess"
        sys.path.insert(0, str(directory))
        from pose2d import Pose2d
        from pose2d_utils import AAPoseMeta
        from human_visualization import draw_aapose_by_meta_new
        from utils import get_face_bboxes
        self.meta_class = AAPoseMeta
        self.draw = draw_aapose_by_meta_new
        self.face_box = get_face_bboxes
        self.size = tuple(size)
        base = Path(checkpoint) / "process_checkpoint"
        self.pose = Pose2d(str(base / "pose2d/vitpose_h_wholebody.onnx"),
                          detector_checkpoint=str(base / "det/yolov10m.onnx"),
                          device=device, vitpose_batch_size=CHUNK, smooth=False)

    def warmup(self, counts):
        """Run each batch shape once; ONNX Runtime plans a shape on its first use."""
        detector = self.pose.detector
        blank = np.zeros((1, 3, *detector.input_resolution), dtype=np.float32)
        detector.session.run([], {detector.input_name: blank})
        height, width = self.pose.model.input_resolution
        for count in counts:
            self.pose.model(np.zeros((count, 3, height, width), dtype=np.float32),
                            np.tile([width / 2, height / 2], (count, 1)),
                            np.ones((count, 2)))

    def reset(self):
        for name in ("last_raw_bboxes", "last_raw_keypoints", "last_smoothed_keypoints"):
            setattr(self.pose, name, None)

    @torch.no_grad()
    def __call__(self, frames):
        images = [decode_image(data) for _, _, data in frames]
        poses, faces = [], []
        for image, meta in zip(images, self.pose(images)):
            box = self.face_box(meta['keypoints_face'][:, :2], scale=1.3, image_shape=image.shape[:2])
            x0, x1, y0, y1 = box
            face = image[y0:y1, x0:x1]
            if face.size == 0:
                raise ValueError("Empty face crop.")
            faces.append(cv2.resize(face, (512, 512)))
            canvas = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
            pose_meta = self.meta_class.from_humanapi_meta(meta).resize(*self.size)
            poses.append(self.draw(canvas, pose_meta))
        return np.stack(poses), np.stack(faces)

# torch must be imported before decord: decord loads its own CUDA libraries, and
# once it has, torch.cuda's lazy init segfaults. pose2d imports torch too, but by
# then decord is already loaded, so the import has to stay here.
import torch  # noqa: F401  (import order matters, see above)

import os
import numpy as np
import shutil
import cv2
from loguru import logger
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy

from decord import VideoReader
from pose2d import Pose2d
from pose2d_utils import AAPoseMeta
from utils import resize_by_area, get_frame_indices, padding_resize, get_face_bboxes
from temporal_filter import smooth_face_bboxes
from human_visualization import draw_aapose_by_meta_new
from retarget_pose import get_retarget_pose


class ProcessPipeline():
    def __init__(self, det_checkpoint_path, pose2d_checkpoint_path,
                 vitpose_batch_size=16,
                 smooth=False):
        self.smooth = bool(smooth)
        self.pose2d = Pose2d(
            checkpoint=pose2d_checkpoint_path,
            detector_checkpoint=det_checkpoint_path,
            vitpose_batch_size=vitpose_batch_size,
            smooth=self.smooth)

    def _prepare_face_bboxes(self, pose_metas, image_shape, fps):
        raw = np.asarray([
            get_face_bboxes(meta['keypoints_face'][:, :2], scale=1.3,
                            image_shape=image_shape)
            for meta in pose_metas
        ], dtype=np.float64)
        if self.smooth and len(raw) > 1:
            smoothed = smooth_face_bboxes(raw, image_shape, freq=float(fps))
        else:
            smoothed = raw.astype(int)
        return smoothed


    def __call__(self, video_path, refer_image_path, output_path,
                 resolution_area=[1280, 720], fps=30, retarget_flag=False):
        logger.info(f"Processing reference image: {refer_image_path}")
        refer_img = cv2.imread(refer_image_path)
        src_ref_path = os.path.join(output_path, 'src_ref.png')
        shutil.copy(refer_image_path, src_ref_path)
        refer_img = refer_img[..., ::-1]
        refer_img = resize_by_area(refer_img, resolution_area[0] * resolution_area[1], divisor=16)

        logger.info(f"Processing template video: {video_path}")
        video_reader = VideoReader(video_path)
        frame_num = len(video_reader)
        print('frame_num: {}'.format(frame_num))

        video_fps = video_reader.get_avg_fps()
        print('video_fps: {}'.format(video_fps))
        print('fps: {}'.format(fps))

        # TODO: Maybe we can switch to PyAV later, which can get accurate frame num
        duration = video_reader.get_frame_timestamp(-1)[-1]
        expected_frame_num = int(duration * video_fps + 0.5)
        ratio = abs((frame_num - expected_frame_num)/frame_num)
        if ratio > 0.1:
            print("Warning: The difference between the actual number of frames and the expected number of frames is two large")
            frame_num = expected_frame_num

        if fps == -1:
            fps = video_fps

        target_num = int(frame_num / video_fps * fps)
        print('target_num: {}'.format(target_num))
        idxs = get_frame_indices(frame_num, video_fps, target_num, fps)
        frames = video_reader.get_batch(idxs).asnumpy()

        logger.info(f"Processing pose meta")

        # fps is resolved by here (-1 already replaced by video_fps); the
        # One-Euro filter needs the true frame rate as its sampling rate.
        self.pose2d.smooth_freq = float(fps)
        tpl_pose_metas = self.pose2d(frames)

        # Collect the crop boxes first, then smooth them before cropping.
        # get_face_bboxes reduces 68 landmarks to their min/max, so a single
        # noisy landmark shifts the whole box and the 512x512 face crop jumps.
        # Filtering the box is cheaper and better targeted than filtering the
        # landmarks, whose exact positions are not used anywhere else.
        img_shape = (frames[0].shape[0], frames[0].shape[1])
        face_bboxes = self._prepare_face_bboxes(tpl_pose_metas, img_shape, fps)

        face_images = []
        for idx, (x1, x2, y1, y2) in enumerate(face_bboxes):
            face_image = frames[idx][y1:y2, x1:x2]
            face_image = cv2.resize(face_image, (512, 512))
            face_images.append(face_image)

        if retarget_flag:
            # Only retargeting needs the reference pose and the template's first
            # frame, so both detections stay inside this branch.
            refer_pose_meta = self.pose2d([refer_img])[0]
            tpl_pose_meta0 = self.pose2d(frames[:1])[0]
            tpl_retarget_pose_metas = get_retarget_pose(
                tpl_pose_meta0, refer_pose_meta, tpl_pose_metas, None, None)
        else:
            tpl_retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in tpl_pose_metas]

        cond_images = []
        for meta in tpl_retarget_pose_metas:
            if retarget_flag:
                # Retargeted poses are already in the reference frame.
                canvas = np.zeros_like(refer_img)
                conditioning_image = draw_aapose_by_meta_new(canvas, meta)
            else:
                canvas = np.zeros_like(frames[0])
                conditioning_image = draw_aapose_by_meta_new(canvas, meta)
                conditioning_image = padding_resize(conditioning_image, refer_img.shape[0], refer_img.shape[1])
            cond_images.append(conditioning_image)

        src_face_path = os.path.join(output_path, 'src_face.mp4')
        mpy.ImageSequenceClip(face_images, fps=fps).write_videofile(src_face_path)

        src_pose_path = os.path.join(output_path, 'src_pose.mp4')
        mpy.ImageSequenceClip(cond_images, fps=fps).write_videofile(src_pose_path)
        return True

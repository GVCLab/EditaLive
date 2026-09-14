import os
import argparse
from process_pipepline import ProcessPipeline


def _parse_args():
    parser = argparse.ArgumentParser(
        description="The preprocessing pipeline."
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="The path to the preprocessing model's checkpoint directory. ")

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="The path to the source video.")
    parser.add_argument(
        "--refer_path",
        type=str,
        default=None,
        help="The path to the refererence image.")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="The path to save the processed results.")
    
    parser.add_argument(
        "--resolution_area",
        type=int,
        nargs=2,
        default=[1280, 720],
        help="The target resolution for processing, specified as [width, height]. To handle different aspect ratios, the video is resized to have a total area equivalent to width * height, while preserving the original aspect ratio."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=-1,
        help="The target FPS for processing the source video. Set to -1 to use the video's original FPS."
    )
    parser.add_argument(
        "--vitpose_batch_size",
        type=int,
        default=16,
        help="Number of frames per ViTPose inference batch. YOLO detection "
             "continues to run with batch size 1."
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        default=False,
        help="Apply One-Euro temporal smoothing to pose keypoints and face crops."
    )

    parser.add_argument(
        "--retarget_flag",
        action="store_true",
        default=False,
        help="Whether to use pose retargeting.")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = _parse_args()
    args_dict = vars(args)
    print(args_dict)

    assert len(args.resolution_area) == 2, "resolution_area should be a list of two integers [width, height]"
    assert args.vitpose_batch_size > 0, "vitpose_batch_size must be positive"

    pose2d_checkpoint_path = os.path.join(args.ckpt_path, 'pose2d/vitpose_h_wholebody.onnx')
    det_checkpoint_path = os.path.join(args.ckpt_path, 'det/yolov10m.onnx')

    process_pipeline = ProcessPipeline(
        det_checkpoint_path=det_checkpoint_path,
        pose2d_checkpoint_path=pose2d_checkpoint_path,
        vitpose_batch_size=args.vitpose_batch_size,
        smooth=args.smooth)
    os.makedirs(args.save_path, exist_ok=True)
    process_pipeline(video_path=args.video_path,
                     refer_image_path=args.refer_path,
                     output_path=args.save_path,
                     resolution_area=args.resolution_area,
                     fps=args.fps,
                     retarget_flag=args.retarget_flag)

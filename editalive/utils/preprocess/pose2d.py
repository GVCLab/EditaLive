import os
import cv2
from typing import Union, List

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime

from pose2d_utils import (
    read_img,
    box_convert_simple,
    bbox_from_detector,
    crop,
    keypoints_from_heatmaps,
    load_pose_metas_from_kp2ds_seq
)
from temporal_filter import smooth_kp2ds, KP_GROUP_PARAMS


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_UINT8_TO_FLOAT = np.float32(1.0 / 255.0)


class SimpleOnnxInference(object):
    def __init__(self, checkpoint, device='cuda', reverse_input=False, **kwargs):
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda':
            device_id = 0 if device.index is None else device.index
            device = '{}:{}'.format(device.type, device_id)
            providers = [("CUDAExecutionProvider", {"device_id": str(device_id)}), "CPUExecutionProvider"]
        else:
            device_id = None
            providers = ["CPUExecutionProvider"]
        self.device = device
        self.device_id = device_id
        if not os.path.exists(checkpoint):
            raise RuntimeError("{} is not existed!".format(checkpoint))
        
        if os.path.isdir(checkpoint):
            checkpoint = os.path.join(checkpoint, 'end2end.onnx')

        session_options = onnxruntime.SessionOptions()
        # ONNX Runtime otherwise creates one worker per host CPU and attempts to
        # pin all of them.  That is noisy and can heavily oversubscribe Slurm or
        # container jobs when CUDA execution is unavailable and ORT falls back
        # to CPU.  Respect an explicit override, then the Slurm CPU allocation.
        ort_threads = int(os.environ.get(
            'WAN_ORT_INTRA_THREADS', os.environ.get('SLURM_CPUS_PER_TASK', '0')))
        if ort_threads > 0:
            session_options.intra_op_num_threads = ort_threads
            session_options.inter_op_num_threads = 1
            session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        self.session = onnxruntime.InferenceSession(
            checkpoint, sess_options=session_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_resolution = self.session.get_inputs()[0].shape[2:] if not reverse_input else self.session.get_inputs()[0].shape[2:][::-1]
        self.input_resolution = np.array(self.input_resolution)
        

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

    def get_output_names(self):
        output_names = []
        for node in self.session.get_outputs():
            output_names.append(node.name)
        return output_names


    def set_device(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda':
            device_id = 0 if device.index is None else device.index
            device = '{}:{}'.format(device.type, device_id)
            providers = [("CUDAExecutionProvider", {"device_id": str(device_id)}), "CPUExecutionProvider"]
        else:
            device_id = None
            providers = ["CPUExecutionProvider"]
        self.session.set_providers(providers)
        self.device = device
        self.device_id = device_id


class Yolo(SimpleOnnxInference):
    def __init__(self, checkpoint, device='cuda', threshold_conf=0.05, threshold_multi_persons=0.1, input_resolution=(640, 640), threshold_iou=0.5, threshold_bbox_shape_ratio=0.4, cat_id=[1], select_type='max', strict=True, sorted_func=None, **kwargs):
        super(Yolo, self).__init__(checkpoint, device=device, **kwargs)
        
        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape

        self.input_width = 640
        self.input_height = 640
        
        self.threshold_multi_persons = threshold_multi_persons
        self.threshold_conf = threshold_conf
        self.threshold_iou = threshold_iou
        self.threshold_bbox_shape_ratio = threshold_bbox_shape_ratio
        self.input_resolution = input_resolution
        self.cat_id = cat_id
        self.select_type = select_type
        self.strict = strict
        self.sorted_func = sorted_func
        
        
    def preprocess(self, input_image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        img = read_img(input_image)
        # Get the height and width of the input image
        img_height, img_width = img.shape[:2]
        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_resolution[1], self.input_resolution[0]))
        # Convert directly to float32. ``img / 255.0`` would first allocate a
        # float64 image and then copy it again when casting to float32.
        image_data = np.transpose(img, (2, 0, 1)).astype(np.float32)
        image_data *= _UINT8_TO_FLOAT
        # Return the preprocessed image data
        return image_data, np.array([img_height, img_width])

    
    def postprocess(self, output, shape_raw, cat_id=[1]):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape

        outputs = np.squeeze(output)
        if len(outputs.shape) == 1:
            outputs = outputs[None]
        if output.shape[-1] != 6 and output.shape[1] == 84:
            outputs = np.transpose(outputs)
        
        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = shape_raw[1] / self.input_width
        y_factor = shape_raw[0] / self.input_height

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        if outputs.shape[-1] == 6:
            max_scores = outputs[:, 4]
            classid = outputs[:, -1]
            
            threshold_conf_masks = max_scores >= self.threshold_conf
            classid_masks = classid[threshold_conf_masks] != 3.14159

            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[threshold_conf_masks][classid_masks]

            boxes = outputs[:, :4][threshold_conf_masks][classid_masks]
            boxes[:, [0, 2]] *= x_factor
            boxes[:, [1, 3]] *= y_factor
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            boxes = boxes.astype(np.int32)

        else:
            classes_scores = outputs[:, 4:]
            max_scores = np.amax(classes_scores, -1)
            threshold_conf_masks = max_scores >= self.threshold_conf

            classid = np.argmax(classes_scores[threshold_conf_masks], -1)

            classid_masks = classid!=3.14159
            
            classes_scores = classes_scores[threshold_conf_masks][classid_masks]
            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[classid_masks]
    
            xywh = outputs[:, :4][threshold_conf_masks][classid_masks]

            x = xywh[:, 0:1]
            y = xywh[:, 1:2]
            w = xywh[:, 2:3]
            h = xywh[:, 3:4]
    
            left = ((x - w / 2) * x_factor)
            top = ((y - h / 2) * y_factor)
            width = (w * x_factor)
            height = (h * y_factor)
            boxes = np.concatenate([left, top, width, height], axis=-1).astype(np.int32)

        boxes = boxes.tolist()
        scores = max_scores.tolist()
        class_ids = classid.tolist()

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.threshold_conf, self.threshold_iou)
        # Iterate over the selected indices after non-maximum suppression
        
        results = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = box_convert_simple(boxes[i], 'xywh2xyxy')
            score = scores[i]
            class_id = class_ids[i]
            results.append(box + [score] + [class_id])
            # # Draw the detection on the input image

        # Return the modified input image
        return np.array(results)

    
    def process_results(self, results, shape_raw, cat_id=[1], single_person=True):
        if isinstance(results, tuple):
            det_results = results[0]
        else:
            det_results = results

        person_results = []
        person_count = 0
        if len(results):
            max_idx = -1
            max_bbox_size = shape_raw[0] * shape_raw[1] * -10
            max_bbox_shape = -1
            
            bboxes = []
            idx_list = []
            for i in range(results.shape[0]):
                bbox = results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    idx_list.append(i)
                    bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                    if bbox_shape > max_bbox_shape:
                        max_bbox_shape = bbox_shape
            
            results = results[idx_list]

            for i in range(results.shape[0]):
                bbox = results[i]
                bboxes.append(bbox)
                if self.select_type == 'max':
                    bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                if bbox_size > max_bbox_size:
                    if (self.strict or max_idx != -1) and bbox_shape < max_bbox_shape * self.threshold_bbox_shape_ratio:
                        continue
                    max_bbox_size = bbox_size
                    max_bbox_shape = bbox_shape
                    max_idx = i

            if self.sorted_func is not None and len(bboxes) > 0:
                max_idx = self.sorted_func(bboxes, shape_raw)
                bbox = bboxes[max_idx]
                if self.select_type == 'max':
                    max_bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    max_bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                
            if max_idx != -1:
                person_count = 1

            if max_idx != -1:
                person = {}
                person['bbox'] = results[max_idx, :5]
                person['track_id'] = int(0)
                person_results.append(person)

            for i in range(results.shape[0]):
                bbox = results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    if self.select_type == 'max':
                        bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                    elif self.select_type == 'center':
                        bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                    if i != max_idx and bbox_size > max_bbox_size * self.threshold_multi_persons and bbox_size < max_bbox_size:
                        person_count += 1
                        if not single_person:
                            person = {}
                            person['bbox'] = results[i, :5]
                            person['track_id'] = int(person_count - 1)
                            person_results.append(person)                   
            return person_results
        else:
            return None
        

    def postprocess_threading(self, outputs, shape_raw, person_results, i, single_person=True, **kwargs):
        result = self.postprocess(outputs[i], shape_raw[i], cat_id=self.cat_id)
        result = self.process_results(result, shape_raw[i], cat_id=self.cat_id, single_person=single_person)
        if result is not None and len(result) != 0:
            person_results[i] = result


    def forward(self, img, shape_raw, **kwargs):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            shape_raw = shape_raw.cpu().numpy()

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img})[0]
        person_results = [[{'bbox': np.array([0., 0., 1.*shape_raw[i][1], 1.*shape_raw[i][0], -1]), 'track_id': -1}] for i in range(len(outputs))]

        for i in range(len(outputs)):
            self.postprocess_threading(outputs, shape_raw, person_results, i, **kwargs)         
        return person_results


class ViTPose(SimpleOnnxInference):
    def __init__(self, checkpoint, device='cuda', **kwargs):
        super(ViTPose, self).__init__(checkpoint, device=device)
        requested_gpu_decode = os.environ.get(
            'WAN_POSE_GPU_DECODE', '1') not in ('0', 'false', 'False')
        self.gpu_decode = (
            requested_gpu_decode
            and self.device_id is not None
            and 'CUDAExecutionProvider' in self.session.get_providers())
        self._gpu_output_buffers = {}
        self._gaussian_weights = {}

        output_shape = self.session.get_outputs()[0].shape
        input_shape = self.session.get_inputs()[0].shape
        self.num_keypoints = (output_shape[1]
                              if isinstance(output_shape[1], int) else 133)
        # The exporter marks these dimensions as symbolic, but this ViTPose
        # model always downsamples its fixed 256x192 input by four.
        self.heatmap_height = (output_shape[2]
                               if isinstance(output_shape[2], int)
                               else input_shape[2] // 4)
        self.heatmap_width = (output_shape[3]
                              if isinstance(output_shape[3], int)
                              else input_shape[3] // 4)

    def _get_gaussian_weight(self, channels, kernel, dtype, device):
        key = (channels, kernel, dtype, device)
        weight = self._gaussian_weights.get(key)
        if weight is None:
            # Reuse OpenCV's sigma=0 rule so the CUDA kernel matches the legacy
            # CPU decoder's Gaussian weights.
            kernel_1d = cv2.getGaussianKernel(kernel, 0, cv2.CV_32F)
            kernel_2d = np.matmul(kernel_1d, kernel_1d.T)
            weight = torch.from_numpy(kernel_2d).to(
                device=device, dtype=dtype).view(1, 1, kernel, kernel)
            weight = weight.expand(channels, 1, kernel, kernel).contiguous()
            self._gaussian_weights[key] = weight
        return weight

    @torch.inference_mode()
    def _decode_heatmaps_gpu(self, heatmaps, center, scale, kernel=11):
        """Decode ViTPose heatmaps on CUDA and copy back only N x K x 3."""
        _, channels, height, width = heatmaps.shape
        flat = heatmaps.flatten(2)
        maxvals, indices = torch.max(flat, dim=2, keepdim=True)
        pred_x = torch.remainder(indices, width).to(torch.float32)
        pred_y = torch.div(indices, width, rounding_mode='floor').to(torch.float32)
        preds = torch.cat((pred_x, pred_y), dim=2)
        preds = torch.where(maxvals > 0, preds, -torch.ones_like(preds))

        weight = self._get_gaussian_weight(
            channels, kernel, heatmaps.dtype, heatmaps.device)
        origin_max = torch.amax(heatmaps, dim=(2, 3), keepdim=True)
        blurred = F.conv2d(
            heatmaps, weight, padding=kernel // 2, groups=channels)
        blurred_max = torch.amax(blurred, dim=(2, 3), keepdim=True)
        blurred = blurred * (origin_max / blurred_max)
        blurred = torch.log(torch.clamp_min(blurred, 1e-10))

        px = preds[..., 0].to(torch.long)
        py = preds[..., 1].to(torch.long)
        valid = ((px > 1) & (px < width - 2)
                 & (py > 1) & (py < height - 2))
        n, k = torch.where(valid)
        if n.numel():
            x = px[n, k]
            y = py[n, k]

            # NumPy promotes the legacy Taylor calculation to float64.  Match
            # that precision for stable offsets near singular Hessians.
            def value(yy, xx):
                return blurred[n, k, yy, xx].to(torch.float64)

            dx = 0.5 * (value(y, x + 1) - value(y, x - 1))
            dy = 0.5 * (value(y + 1, x) - value(y - 1, x))
            dxx = 0.25 * (
                value(y, x + 2) - 2 * value(y, x) + value(y, x - 2))
            dxy = 0.25 * (
                value(y + 1, x + 1) - value(y - 1, x + 1)
                - value(y + 1, x - 1) + value(y - 1, x - 1))
            dyy = 0.25 * (
                value(y + 2, x) - 2 * value(y, x) + value(y - 2, x))
            determinant = dxx * dyy - dxy.square()
            nonsingular = determinant != 0
            if torch.any(nonsingular):
                nn = n[nonsingular]
                kk = k[nonsingular]
                det = determinant[nonsingular]
                offset_x = -(
                    dyy[nonsingular] * dx[nonsingular]
                    - dxy[nonsingular] * dy[nonsingular]) / det
                offset_y = (
                    dxy[nonsingular] * dx[nonsingular]
                    - dxx[nonsingular] * dy[nonsingular]) / det
                preds[nn, kk, 0] += offset_x.to(preds.dtype)
                preds[nn, kk, 1] += offset_y.to(preds.dtype)

        center_gpu = torch.as_tensor(
            center, dtype=preds.dtype, device=preds.device)
        scale_gpu = torch.as_tensor(
            scale, dtype=preds.dtype, device=preds.device) * 200
        preds[..., 0] = (
            preds[..., 0] * (scale_gpu[:, None, 0] / width)
            + center_gpu[:, None, 0] - scale_gpu[:, None, 0] * 0.5)
        preds[..., 1] = (
            preds[..., 1] * (scale_gpu[:, None, 1] / height)
            + center_gpu[:, None, 1] - scale_gpu[:, None, 1] * 0.5)
        return torch.cat((preds, maxvals), dim=2).cpu().numpy()

    def _forward_gpu(self, img, center, scale):
        batch = img.shape[0]
        output_shape = (
            batch, self.num_keypoints, self.heatmap_height, self.heatmap_width)
        heatmaps = self._gpu_output_buffers.get(output_shape)
        if heatmaps is None:
            heatmaps = torch.empty(
                output_shape, dtype=torch.float32,
                device=torch.device('cuda', self.device_id))
            self._gpu_output_buffers[output_shape] = heatmaps

        io_binding = self.session.io_binding()
        io_binding.bind_cpu_input(self.input_name, img)
        io_binding.bind_output(
            self.output_name,
            device_type='cuda',
            device_id=self.device_id,
            element_type=np.float32,
            shape=output_shape,
            buffer_ptr=heatmaps.data_ptr())
        self.session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()
        return self._decode_heatmaps_gpu(heatmaps, center, scale)

    def forward(self, img, center, scale, **kwargs):
        if self.gpu_decode:
            try:
                return self._forward_gpu(img, center, scale)
            except Exception as error:
                # Preserve portability when an older ORT build advertises CUDA
                # but cannot bind its output to an external CUDA allocation.
                import warnings
                warnings.warn(
                    f'GPU pose heatmap decoding failed; falling back to CPU: {error}',
                    RuntimeWarning)
                self.gpu_decode = False
        heatmaps = self.session.run([], {self.session.get_inputs()[0].name: img})[0]
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                            center=center,
                                            scale=scale*200,
                                            unbiased=True, 
                                            use_udp=False)
        return np.concatenate([points, prob], axis=2)


    @staticmethod
    def preprocess(img, bbox=None, input_resolution=(256, 192), rescale=1.25, mask=None, **kwargs):
        if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
            bbox = np.array([0, 0, img.shape[1], img.shape[0]])
        
        bbox_xywh = bbox
        if mask is not None:
            img = np.where(mask>128, img, mask)

        if isinstance(input_resolution, int):
            center, scale = bbox_from_detector(bbox_xywh, (input_resolution, input_resolution), rescale=rescale)
            img, new_shape, old_xy, new_xy = crop(img, center, scale, (input_resolution, input_resolution))
        else:
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img, new_shape, old_xy, new_xy = crop(img, center, scale, (input_resolution[0], input_resolution[1]))

        # ``crop`` already returns a private float32 array. Normalize it in
        # place to avoid two float64 intermediates per frame, then make NCHW
        # contiguous for efficient stacking into ViTPose batches.
        img_norm = img
        img_norm *= _UINT8_TO_FLOAT
        img_norm -= _IMAGENET_MEAN
        img_norm /= _IMAGENET_STD
        img_norm = np.ascontiguousarray(img_norm.transpose(2, 0, 1))
        return img_norm, np.array(center), np.array(scale)


class Pose2d:
    def __init__(self, checkpoint, detector_checkpoint=None, device='cuda',
                 smooth=False, smooth_freq=16.0, smooth_min_cutoff=1.0,
                 smooth_beta=0.05, smooth_group_params=KP_GROUP_PARAMS,
                 vitpose_batch_size=16, **kwargs):
        """
        smooth: apply One-Euro temporal smoothing to the keypoint sequence.
            Detector + ViTPose run per-frame independently, so the raw skeleton
            jitters; see temporal_filter.py. No-op on single images.
        smooth_freq: video fps, used as the filter's sampling rate.
        smooth_min_cutoff / smooth_beta: fallback params for keypoint layouts
            other than wholebody-133, and for groups not covered by
            smooth_group_params.
        smooth_group_params: per-region (min_cutoff, beta); head/face get heavier
            smoothing than body/hands. Pass None to filter everything uniformly.
        """
        if (not isinstance(vitpose_batch_size, int)
                or isinstance(vitpose_batch_size, bool)
                or vitpose_batch_size <= 0):
            raise ValueError(
                "vitpose_batch_size must be a positive integer, "
                f"got {vitpose_batch_size}")
        self.vitpose_batch_size = vitpose_batch_size

        if detector_checkpoint is not None:
            self.detector = Yolo(detector_checkpoint, device)
        else:
            self.detector = None

        self.model = ViTPose(checkpoint, device)
        self.device = device

        self.smooth = bool(smooth)
        self.smooth_freq = smooth_freq
        self.smooth_min_cutoff = smooth_min_cutoff
        self.smooth_beta = smooth_beta
        self.smooth_group_params = smooth_group_params

    def load_images(self, inputs):
        """
        Load images from various input types.
        
        Args:
            inputs (Union[str, np.ndarray, List[np.ndarray]]): Input can be file path, 
                     single image array, or list of image arrays
            
        Returns:
            List[np.ndarray]: List of RGB image arrays
            
        Raises:
            ValueError: If file format is unsupported or image cannot be read
        """
        if isinstance(inputs, str):
            if inputs.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                cap = cv2.VideoCapture(inputs)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                images = frames
            elif inputs.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = cv2.cvtColor(cv2.imread(inputs), cv2.COLOR_BGR2RGB)
                if img is None:
                    raise ValueError(f"Cannot read image: {inputs}")
                images = [img]
            else:
                raise ValueError(f"Unsupported file format: {inputs}")
                
        elif isinstance(inputs, np.ndarray):
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in inputs]
        elif isinstance(inputs, list):
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in inputs]
        return images

    def __call__(
        self, 
        inputs: Union[str, np.ndarray, List[np.ndarray]],
        return_image: bool = False,
        **kwargs
    ):
        """
        Process input and estimate 2D keypoints.
        
        Args:
            inputs (Union[str, np.ndarray, List[np.ndarray]]): Input can be file path,
                     single image array, or list of image arrays
            **kwargs: Additional arguments for processing
            
        Returns:
            np.ndarray: Array of detected 2D keypoints for all input images
        """
        images = self.load_images(inputs)
        H, W = images[0].shape[:2]
        if self.detector is not None:
            bboxes = []
            for _image in images:
                img, shape = self.detector.preprocess(_image)
                bboxes.append(self.detector(img[None], shape[None])[0][0]["bbox"])
        else:
            bboxes = [None] * len(images)

        model_inputs = []
        centers = []
        scales = []
        for _image, _bbox in zip(images, bboxes):
            img, center, scale = self.model.preprocess(_image, _bbox)
            model_inputs.append(img)
            centers.append(center)
            scales.append(scale)

        # ViTPose was exported with a dynamic batch dimension. Group its fixed
        # 256x192 crops to reduce CUDA launch and synchronization overhead while
        # keeping YOLO's fixed-batch-1 model unchanged.
        kp2ds = []
        for start in range(0, len(model_inputs), self.vitpose_batch_size):
            stop = start + self.vitpose_batch_size
            kp2ds.append(self.model(
                np.stack(model_inputs[start:stop]),
                np.stack(centers[start:stop]),
                np.stack(scales[start:stop]),
            ))
        kp2ds = np.concatenate(kp2ds, 0)
        # Smooth here, while coordinates are still in pixels -- beta is tuned at
        # pixel scale, and load_pose_metas_from_kp2ds_seq normalises by W/H.
        if self.smooth and kp2ds.shape[0] > 1:
            kp2ds = smooth_kp2ds(kp2ds, freq=self.smooth_freq,
                                 min_cutoff=self.smooth_min_cutoff,
                                 beta=self.smooth_beta,
                                 group_params=self.smooth_group_params)
        metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)
        return metas

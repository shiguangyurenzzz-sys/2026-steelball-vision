"""Minimal YOLOv8 TopK detector used for K230 FPS A/B tests."""

import time

import nncase_runtime as nn
import ulab.numpy as np
from libs.AI2D import Ai2d
from libs.AIBase import AIBase


BOX_COLOR = (255, 0, 255, 0)
TEXT_COLOR = (255, 255, 255, 0)


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def letterbox_params(input_size, output_size):
    """Return resize scale and output-space padding used by AI2D."""
    scale = min(
        output_size[0] / input_size[0],
        output_size[1] / input_size[1],
    )
    resized_width = int(input_size[0] * scale)
    resized_height = int(input_size[1] * scale)
    horizontal_padding = (output_size[0] - resized_width) / 2
    vertical_padding = (output_size[1] - resized_height) / 2
    top = int(round(vertical_padding - 0.1))
    bottom = int(round(vertical_padding + 0.1))
    left = int(round(horizontal_padding - 0.1))
    right = int(round(horizontal_padding + 0.1))
    return (
        scale,
        horizontal_padding,
        vertical_padding,
        top,
        bottom,
        left,
        right,
    )


def elapsed_ms(start_us):
    return time.ticks_diff(time.ticks_us(), start_us) / 1000.0


class YoloTopKBenchmark(AIBase):
    """Run either the original full-frame model or the 320x32 ROI model."""

    def __init__(
        self,
        kmodel_path,
        model_input_size,
        rgb888p_size,
        topk_candidates,
        crop_region=None,
        confidence_threshold=0.20,
        nms_threshold=0.45,
        max_detections=1,
        debug_mode=0,
    ):
        aligned_rgb_size = [
            align_up(rgb888p_size[0], 16),
            rgb888p_size[1],
        ]
        super().__init__(
            kmodel_path,
            model_input_size,
            aligned_rgb_size,
            debug_mode,
        )
        self.model_input_size = model_input_size
        self.rgb888p_size = aligned_rgb_size
        self.topk_candidates = topk_candidates
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

        if crop_region is None:
            self.crop_region = [
                0,
                0,
                self.rgb888p_size[0],
                self.rgb888p_size[1],
            ]
            self.crop_enabled = False
        else:
            self.crop_region = crop_region
            self.crop_enabled = True

        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(
            nn.ai2d_format.NCHW_FMT,
            nn.ai2d_format.NCHW_FMT,
            np.uint8,
            np.uint8,
        )
        self.scale = 1.0
        self.pad_x = 0.0
        self.pad_y = 0.0

    def config_preprocess(self):
        region_x, region_y, region_width, region_height = self.crop_region
        if self.crop_enabled:
            # Crop is performed by AI2D; no Python ndarray slice or copy.
            self.ai2d.crop(
                region_x,
                region_y,
                region_width,
                region_height,
            )

        (
            self.scale,
            self.pad_x,
            self.pad_y,
            top,
            bottom,
            left,
            right,
        ) = letterbox_params(
            [region_width, region_height],
            self.model_input_size,
        )
        self.ai2d.pad(
            [0, 0, 0, 0, top, bottom, left, right],
            0,
            [114, 114, 114],
        )
        self.ai2d.resize(
            nn.interp_method.tf_bilinear,
            nn.interp_mode.half_pixel,
        )
        self.ai2d.build(
            [
                1,
                3,
                self.rgb888p_size[1],
                self.rgb888p_size[0],
            ],
            [
                1,
                3,
                self.model_input_size[1],
                self.model_input_size[0],
            ],
        )

    def postprocess(self, results):
        if len(results) != 1:
            raise RuntimeError("TopK YOLO model must have one output")

        output = results[0]
        topk = self.topk_candidates
        if output.shape == (1, 5, topk):
            output = output[0]
        elif output.shape == (1, topk, 5):
            output = output[0].transpose()
        elif output.shape == (topk, 5):
            output = output.transpose()
        elif output.shape != (5, topk):
            raise RuntimeError(
                "unexpected TopK output shape: %s" % str(output.shape)
            )

        region_x, region_y, region_width, region_height = self.crop_region
        min_x = float(region_x)
        min_y = float(region_y)
        max_x = float(region_x + region_width - 1)
        max_y = float(region_y + region_height - 1)

        candidates = []
        for index in range(topk):
            score = float(output[4, index])
            # The model-side TopK output is sorted by descending score.
            if score < self.confidence_threshold:
                break

            center_x = float(output[0, index])
            center_y = float(output[1, index])
            width = float(output[2, index])
            height = float(output[3, index])
            # Convert 320x32 letterbox coordinates to the full 640x480 AI frame.
            x1 = (
                (center_x - width / 2 - self.pad_x) / self.scale
                + region_x
            )
            y1 = (
                (center_y - height / 2 - self.pad_y) / self.scale
                + region_y
            )
            x2 = (
                (center_x + width / 2 - self.pad_x) / self.scale
                + region_x
            )
            y2 = (
                (center_y + height / 2 - self.pad_y) / self.scale
                + region_y
            )
            x1 = max(min_x, min(max_x, x1))
            y1 = max(min_y, min(max_y, y1))
            x2 = max(min_x, min(max_x, x2))
            y2 = max(min_y, min(max_y, y2))
            if x2 > x1 and y2 > y1:
                candidates.append([0, score, x1, y1, x2, y2])

        detections = []
        for candidate in candidates:
            keep = True
            for selected in detections:
                if self.box_iou(candidate, selected) > self.nms_threshold:
                    keep = False
                    break
            if keep:
                detections.append(candidate)
                if len(detections) >= self.max_detections:
                    break
        return detections

    @staticmethod
    def box_iou(first, second):
        left = max(first[2], second[2])
        top = max(first[3], second[3])
        right = min(first[4], second[4])
        bottom = min(first[5], second[5])
        intersection = max(0.0, right - left) * max(
            0.0,
            bottom - top,
        )
        first_area = max(0.0, first[4] - first[2]) * max(
            0.0,
            first[5] - first[3],
        )
        second_area = max(0.0, second[4] - second[2]) * max(
            0.0,
            second[5] - second[3],
        )
        union = first_area + second_area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    def draw_result(self, pipeline, detections, fps):
        """Clear the OSD and draw detections returned in full-frame coordinates."""
        pipeline.osd_img.clear()
        detection_count = 0 if not detections else len(detections)
        pipeline.osd_img.draw_string_advanced(
            8,
            8,
            24,
            "balls=%d  fps=%.1f" % (detection_count, fps),
            color=TEXT_COLOR,
        )
        if not detections:
            return

        display_size = pipeline.get_display_size()
        display_width = align_up(display_size[0], 16)
        display_height = display_size[1]
        for detection in detections:
            score = float(detection[1])
            x1 = float(detection[2])
            y1 = float(detection[3])
            x2 = float(detection[4])
            y2 = float(detection[5])

            # Detections are already mapped to the 640x480 AI frame.
            screen_x = int(x1 * display_width / self.rgb888p_size[0])
            screen_y = int(y1 * display_height / self.rgb888p_size[1])
            screen_width = max(
                2,
                int(
                    (x2 - x1)
                    * display_width
                    / self.rgb888p_size[0]
                ),
            )
            screen_height = max(
                2,
                int(
                    (y2 - y1)
                    * display_height
                    / self.rgb888p_size[1]
                ),
            )
            pipeline.osd_img.draw_rectangle(
                screen_x,
                screen_y,
                screen_width,
                screen_height,
                color=BOX_COLOR,
                thickness=4,
            )
            label_y = max(34, screen_y) - 34
            pipeline.osd_img.draw_string_advanced(
                screen_x,
                label_y,
                28,
                "steel_ball %.2f" % score,
                color=BOX_COLOR,
            )

    def run_profiled(self, input_np):
        """Return detections plus AI2D/KPU/postprocess stage times."""
        self.tensors.clear()
        start_us = time.ticks_us()
        self.tensors = self.preprocess(input_np)
        ai2d_ms = elapsed_ms(start_us)

        self.results.clear()
        start_us = time.ticks_us()
        for index in range(self.kpu.inputs_size()):
            self.kpu.set_input_tensor(index, self.tensors[index])
        set_input_ms = elapsed_ms(start_us)

        start_us = time.ticks_us()
        self.kpu.run()
        kpu_ms = elapsed_ms(start_us)

        start_us = time.ticks_us()
        for index in range(self.kpu.outputs_size()):
            output_tensor = self.kpu.get_output_tensor(index)
            self.results.append(output_tensor.to_numpy())
            del output_tensor
        get_output_ms = elapsed_ms(start_us)

        start_us = time.ticks_us()
        detections = self.postprocess(self.results)
        postprocess_ms = elapsed_ms(start_us)

        return detections, (
            ai2d_ms,
            set_input_ms,
            kpu_ms,
            get_output_ms,
            postprocess_ms,
        )

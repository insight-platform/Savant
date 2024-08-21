from typing import ContextManager, List, Tuple

import cupy as cp
import cv2
import tensorrt as trt
import torch
from torch2trt import TRTModule

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import alpha_comp, draw_rect, nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.parameter_storage import param_storage
from savant.utils.memory_repr_pytorch import pytorch_tensor_as_opencv_gpu_mat

IMAGE_WIDTH = param_storage()['frame']['width']
IMAGE_HEIGHT = param_storage()['frame']['height']
ELEMENT_NAME = param_storage()['element_name']
ATTR_NAME = param_storage()['attribute_name']


def load_mask_decoder_engine(path: str) -> TRTModule:
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    mask_decoder_trt = TRTModule(
        engine=engine,
        input_names=[
            'image_embeddings',
            'point_coords',
            'point_labels',
            'mask_input',
            'has_mask_input',
        ],
        output_names=['iou_predictions', 'low_res_masks'],
    )

    return mask_decoder_trt


class Predictor(NvDsPyFuncPlugin):

    def __init__(
        self,
        mask_decoder_engine_path: str,
        **kwargs,
    ):
        self.mask_decoder = load_mask_decoder_engine(mask_decoder_engine_path)
        self.image_encoder_size = 1024
        self.mask_input_size = 256
        self.point_color = (0, 0, 0, 255)
        self.device = 'cuda'
        self.mask_colors = [
            torch.tensor([0, 255, 0, 0], dtype=torch.uint8, device=self.device),
            torch.tensor([255, 0, 0, 0], dtype=torch.uint8, device=self.device),
            torch.tensor([0, 0, 255, 0], dtype=torch.uint8, device=self.device),
            torch.tensor([255, 255, 0, 0], dtype=torch.uint8, device=self.device),
        ]
        self.masks_transparency = [
            125,
            105,
            85,
            65,
        ]
        self.bg_color = torch.tensor(
            [0, 0, 0, 0], dtype=torch.uint8, device=self.device
        )
        # 1 - foreground point, 0 - background point (to exclude from segmentation)
        # number of labels should be equal to number of points passed to the model
        self.point_labels = torch.tensor([[1]], dtype=torch.float32, device=self.device)

        # do not use a masked input
        self.mask_input = torch.zeros(
            1,
            1,
            self.mask_input_size,
            self.mask_input_size,
            dtype=torch.float32,
            device=self.device,
        )
        self.has_mask_input = torch.tensor([0], dtype=torch.float32, device=self.device)

        super().__init__(**kwargs)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """

        # Get CUDA stream for asynchronous processing
        cuda_stream = self.get_cuda_stream(frame_meta)
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            image_embeddings = None
            for obj_meta in frame_meta.objects:
                if obj_meta.is_primary:
                    image_embeddings = obj_meta.get_attr_meta(ELEMENT_NAME, ATTR_NAME)
                    break

            if image_embeddings:
                # Segment by a point
                self.predict_and_draw(
                    [
                        [100, 100],
                        [IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2],
                        [IMAGE_WIDTH - 100, IMAGE_HEIGHT - 500],
                        [IMAGE_WIDTH - 300, IMAGE_HEIGHT - 100],
                    ],
                    self.point_labels,
                    image_embeddings.value,
                    frame_mat,
                    cuda_stream,
                )

            else:
                self.logger.warning(
                    'Model %s attribute %s not found.', ELEMENT_NAME, ATTR_NAME
                )

    def predict_and_draw(
        self,
        original_points: List[List[int]],
        point_labels: torch.Tensor,
        image_embeddings: cp.ndarray,
        frame_mat: ContextManager[cv2.cuda.GpuMat],
        cuda_stream: cv2.cuda.Stream,
    ) -> None:
        image_embeddings_tensor = torch.as_tensor(
            image_embeddings, device=self.device
        ).unsqueeze(0)

        for i, point in enumerate(original_points):
            points = self.scale_points(
                [point], (IMAGE_HEIGHT, IMAGE_WIDTH), self.image_encoder_size
            )

            mask, _, _ = self.predict(
                image_embeddings_tensor, points, point_labels
            )  # [1, 4, H, W]

            for mask_idx in range(4):
                bool_mask = (mask[0, mask_idx] > 0)[..., None]
                self.mask_colors[i][3] = self.masks_transparency[mask_idx]
                mask_overlay = torch.where(
                    bool_mask, self.mask_colors[i], self.bg_color
                )

                mask_overlay_gpu_mat = pytorch_tensor_as_opencv_gpu_mat(mask_overlay)

                alpha_comp(
                    frame_mat,
                    overlay=mask_overlay_gpu_mat,
                    start=(0, 0),
                    stream=cuda_stream,
                )

            draw_rect(
                frame_mat,
                rect=(point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5),
                color=self.point_color,
                thickness=5,
                stream=cuda_stream,
            )

    def scale_points(
        self,
        original_points: List[List[int]],
        image_shape: Tuple[int, int],
        size: int,
    ) -> torch.Tensor:
        points = torch.tensor(
            [original_points],
            dtype=torch.float32,
            device=self.device,
        )
        scale = size / max(*image_shape)
        points = points * scale

        return points

    def predict(
        self,
        image_embeddings: cp.ndarray,
        points: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_iou, low_res_mask = self.run_mask_decoder(
            self.mask_decoder, image_embeddings, points, point_labels
        )

        hi_res_mask = self.upscale_mask(low_res_mask, (IMAGE_HEIGHT, IMAGE_WIDTH))

        return hi_res_mask, mask_iou, low_res_mask

    def run_mask_decoder(
        self,
        mask_decoder_engine: TRTModule,
        image_embeddings: cp.ndarray,
        points: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        iou_predictions, low_res_masks = mask_decoder_engine(
            image_embeddings,
            points,
            point_labels,
            self.mask_input,
            self.has_mask_input,
        )

        return iou_predictions, low_res_masks

    def upscale_mask(
        self, mask: torch.Tensor, image_shape: Tuple[int, int]
    ) -> torch.Tensor:
        if image_shape[1] > image_shape[0]:
            lim_x = self.mask_input_size
            lim_y = int(self.mask_input_size * image_shape[0] / image_shape[1])
        else:
            lim_x = int(self.mask_input_size * image_shape[1] / image_shape[0])
            lim_y = self.mask_input_size

        mask = torch.nn.functional.interpolate(
            mask[:, :, :lim_y, :lim_x], size=image_shape, mode='bilinear'
        )

        return mask

"""Module custom model input preprocessing."""

from savant.base.input_preproc import BasePreprocessObjectImage
from savant.meta.object import ObjectMeta

import cv2

from savant.parameter_storage import param_storage
from savant.utils.image import GPUImage
import numpy as np

REFERENCE_FACIAL_POINTS = np.array(
    [
        [38.29459953, 51.69630051],
        [73.53179932, 51.50139999],
        [56.02519989, 71.73660278],
        [41.54930115, 92.3655014],
        [70.72990036, 92.20410156],
    ]
).astype(np.float32)

FACE_WIDTH = param_storage()['face_width']
FACE_HEIGHT = param_storage()['face_height']
MODEL_NAME = param_storage()['detection_model_name']


class FacePreprocessingObjectImageGPU(BasePreprocessObjectImage):
    """Object meta preprocessing interface."""

    def __call__(
        self,
        object_meta: ObjectMeta,
        frame_image: GPUImage,
        cuda_stream: cv2.cuda.Stream,
    ) -> GPUImage:
        """Aligned and crop face image from frame by applying affine transformation."""
        crop_size = (FACE_WIDTH, FACE_HEIGHT)

        landmarks = object_meta.get_attr_meta(
            element_name=MODEL_NAME, attr_name='landmarks'
        ).value
        face_landmarks = np.array(landmarks).reshape(-1, 5, 2)

        tfm, _ = cv2.estimateAffinePartial2D(face_landmarks, REFERENCE_FACIAL_POINTS)
        face_img = cv2.cuda.warpAffine(
            src=frame_image.gpu_mat, M=tfm, dsize=crop_size, stream=cuda_stream
        )
        return GPUImage(face_img, cuda_stream=cuda_stream)

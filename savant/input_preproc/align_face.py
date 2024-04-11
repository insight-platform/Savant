"""Face image preprocessor."""

import cv2
import numpy as np

from savant.base.input_preproc import BasePreprocessObjectImage
from savant.meta.object import ObjectMeta
from savant.utils.image import GPUImage

FACE_WIDTH = 112
FACE_HEIGHT = 112
REFERENCE_FACIAL_POINTS = np.array(
    [
        [38.29459953, 51.69630051],
        [73.53179932, 51.50139999],
        [56.02519989, 71.73660278],
        [41.54930115, 92.3655014],
        [70.72990036, 92.20410156],
    ]
).astype(np.float32)


class AlignFacePreprocessingObjectImageGPU(BasePreprocessObjectImage):
    """Aligns and crops face image using facial landmarks: left eye center,
    right eye center, nose tip, left mouth corner, and right mouth corner."""

    def __init__(
        self, attr_element_name: str = None, attr_name: str = 'landmarks', **kwargs
    ):
        self.attr_element_name = attr_element_name
        self.attr_name = attr_name
        super().__init__(**kwargs)

    def __call__(
        self,
        object_meta: ObjectMeta,
        frame_image: GPUImage,
        cuda_stream: cv2.cuda.Stream,
    ) -> GPUImage:
        """Aligns and crops face image from the frame."""
        landmarks = object_meta.get_attr_meta(
            element_name=self.attr_element_name or object_meta.element_name,
            attr_name=self.attr_name,
        )
        if landmarks:
            tfm, _ = cv2.estimateAffinePartial2D(
                np.array(landmarks.value).reshape(-1, 5, 2), REFERENCE_FACIAL_POINTS
            )
            face_img = cv2.cuda.warpAffine(
                src=frame_image.gpu_mat,
                M=tfm,
                dsize=(FACE_WIDTH, FACE_HEIGHT),
                stream=cuda_stream,
            )
            return GPUImage(face_img, cuda_stream=cuda_stream)

        # object was added by tracker and doesn't have landmarks
        face_img, _ = frame_image.cut(object_meta.bbox)
        return face_img.resize(resolution=(FACE_WIDTH, FACE_HEIGHT))

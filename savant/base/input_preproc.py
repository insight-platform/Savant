"""Base model input preprocessors."""
from abc import abstractmethod
from typing import Callable, Optional

import cv2
import pyds
from savant_rs.primitives.geometry import BBox

from savant.base.model import OutputImage
from savant.base.pyfunc import BasePyFuncCallableImpl, PyFuncNoopCallException
from savant.deepstream.meta.object import _NvDsObjectMetaImpl
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.deepstream.utils import nvds_frame_meta_iterator, nvds_obj_meta_iterator
from savant.gstreamer import Gst
from savant.meta.object import ObjectMeta
from savant.utils.image import GPUImage
from savant.utils.logging import get_logger


class BasePreprocessObjectMeta(BasePyFuncCallableImpl):
    """Object meta preprocessing interface."""

    @abstractmethod
    def __call__(
        self,
        object_meta: ObjectMeta,
    ) -> BBox:
        """Transforms object meta.

        :param object_meta: original object meta
        :return: changed bbox
        """


class BasePreprocessObjectImage(BasePyFuncCallableImpl):
    """Object image preprocessing interface."""

    @abstractmethod
    def __call__(
        self,
        object_meta: ObjectMeta,
        frame_image: GPUImage,
        cuda_stream: cv2.cuda.Stream,
    ) -> GPUImage:
        """Transforms object image.

        :param object_meta: object meta
        :param frame_image: original image
        :return: changed image
        """


class ObjectsPreprocessing:
    def __init__(self, batch_size: int):
        self._preprocessing_functions = {}
        self._frames_map = {}
        self._stream_pool = [cv2.cuda.Stream() for _ in range(batch_size)]
        self.logger = get_logger(__name__)

    def add_preprocessing_function(
        self,
        element_name: str,
        preprocessing_func: Callable[[ObjectMeta, GPUImage, cv2.cuda.Stream], GPUImage],
    ):
        """Add preprocessing function.

        :param element_name: element name
        :param preprocessing_func: preprocessing function
        """
        self._preprocessing_functions[element_name] = preprocessing_func

    def preprocessing(
        self,
        element_name: str,
        buffer: Gst.Buffer,
        model_uid: int,
        class_id: int,
        output_image: Optional[OutputImage] = None,
        dev_mode: bool = False,
    ):
        """Preprocesses objects by using user function.

        :param element_name: Element name for which preprocessing is done
        :param buffer: gst buffer
        :param model_uid: base on model uid is selected preprocessing object
        :param class_id: base on class id is selected preprocessing object
        :param output_image: max object image size for inference
        :param dev_mode: flag indicates how exceptions from user code are handled
        :return:
        """
        preprocessing_func = self._preprocessing_functions.get(element_name)
        if preprocessing_func is None:
            raise ValueError(f'Cannot find preprocessing function for {element_name}')

        self._frames_map[buffer] = {}

        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)

        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            left = 0
            top = 0
            row_height = 0
            cuda_stream = self._stream_pool[nvds_frame_meta.batch_id]
            with nvds_to_gpu_mat(buffer, nvds_frame_meta) as frame_mat:
                frame_image = GPUImage(image=frame_mat, cuda_stream=cuda_stream)
                copy_frame_image = GPUImage(
                    image=frame_mat.clone(), cuda_stream=cuda_stream
                )
                self._frames_map[buffer][nvds_frame_meta.batch_id] = copy_frame_image
                for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                    if nvds_obj_meta.class_id != class_id:
                        continue
                    if nvds_obj_meta.unique_component_id != model_uid:
                        continue
                    object_meta = _NvDsObjectMetaImpl.from_nv_ds_object_meta(
                        object_meta=nvds_obj_meta, frame_meta=nvds_frame_meta
                    )

                    try:
                        preprocess_image = preprocessing_func(
                            object_meta=object_meta,
                            frame_image=copy_frame_image,
                            cuda_stream=cuda_stream,
                        )
                    except Exception as exc:
                        if dev_mode:
                            if not isinstance(exc, PyFuncNoopCallException):
                                self.logger.exception(
                                    'Error in input image preprocessing.'
                                )
                            continue
                        raise exc

                    if not isinstance(preprocess_image, GPUImage):
                        raise ValueError(
                            'Preprocessing function must return Image object.'
                        )
                    if output_image is not None:
                        preprocess_image = preprocess_image.resize(
                            resolution=(output_image.width, output_image.height),
                            method=output_image.method,
                            interpolation=output_image.cv2_interpolation,
                        )
                    if left + preprocess_image.width > frame_image.width:
                        left = 0
                        if row_height == 0:
                            row_height = preprocess_image.height
                        top += row_height
                        row_height = 0
                    if top >= frame_image.height:
                        raise ValueError(
                            'There is no place on frame ' 'to put object image.'
                        )
                    if top + preprocess_image.height > frame_image.height:
                        raise ValueError(
                            'There is no place on frame ' 'to put object image.'
                        )
                    if preprocess_image.height > row_height:
                        row_height = preprocess_image.height

                    frame_image.paste(preprocess_image, (left, top))
                    nvds_obj_meta.rect_params.top = top
                    nvds_obj_meta.rect_params.left = left
                    nvds_obj_meta.rect_params.width = preprocess_image.width
                    nvds_obj_meta.rect_params.height = preprocess_image.height
                    left += preprocess_image.width

        for stream in self._stream_pool:
            stream.waitForCompletion()

    def restore_frame(self, buffer: Gst.Buffer):
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            restore_frame_image = self._frames_map[buffer].pop(nvds_frame_meta.batch_id)
            with nvds_to_gpu_mat(buffer, nvds_frame_meta) as frame_mat:
                restore_frame_image.gpu_mat.copyTo(frame_mat)

        self._frames_map.pop(buffer)

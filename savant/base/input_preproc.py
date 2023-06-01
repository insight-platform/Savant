"""Base model input preprocessors."""
import math
from abc import abstractmethod
from typing import Optional, Union, Callable, Tuple

import cv2
import numpy as np
import pyds
from savant.gstreamer import Gst
from savant.base.pyfunc import BasePyFuncCallableImpl
from savant.deepstream.meta.object import _NvDsObjectMetaImpl
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.deepstream.utils import nvds_frame_meta_iterator, nvds_obj_meta_iterator, \
    nvds_get_obj_bbox
from savant.meta.bbox import BBox, RBBox
from savant.meta.object import ObjectMeta
from savant.utils.artist.artist_gpumat import ArtistGPUMat


class BasePreprocessObjectMeta(BasePyFuncCallableImpl):
    """Object meta preprocessing interface."""

    @abstractmethod
    def __call__(
        self,
        bbox: pyds.NvBbox_Coords,
        *,
        parent_bbox: Optional[pyds.NvBbox_Coords] = None,
        **kwargs
    ) -> pyds.NvBbox_Coords:
        """Transforms object meta.

        :param bbox: original bbox
        :param parent_bbox: parent object bbox, eg frame
        :return: changed bbox
        """


class Image:
    @abstractmethod
    def cut(self, bbox: Union[BBox, RBBox]) -> 'Image':
        """Cuts image by bbox and padding.
        :param bbox: cutout bbox
        :return: cut image
        """
    @abstractmethod
    def paste(self, image: 'Image', point: Tuple[int, int]):
        """Pastes image to current image.
        :param image: image to paste
        :param point: point to paste
        """

    @property
    @abstractmethod
    def width(self) -> int:
        """Returns image width.
        :return: image width
        """

    @property
    @abstractmethod
    def height(self) -> int:
        """Returns image height.
        :return: image height
        """


class ImageCpu(Image):
    def __init__(self, image: np.ndarray):
        """Image data container.

        :param image: image data
        """
        self._image_np = image
        self._height, self._width, _ = image.shape

    @property
    def to_gpu(self) -> cv2.cuda.GpuMat:
        """Returns GPU image.

        :return: GPU image
        """
        image_gpu = cv2.cuda.GpuMat(self._image_np.shape)
        image_gpu.upload(self._image_np)
        return image_gpu

    def cut(self, bbox: Union[BBox, RBBox]) -> 'ImageCpu':
        """Cuts image by bbox and padding.
        :param bbox: cutout bbox
        :return:
        """
        if isinstance(bbox, BBox):
            cut_bbox = bbox
        elif isinstance(bbox, RBBox):
            cut_bbox = bbox.to_bbox()
        else:
            raise ValueError(f"Unknown bbox type {type(bbox)}")
        return ImageCpu(self._image_np[
            math.floor(cut_bbox.top):math.ceil(cut_bbox.bottom),
            math.floor(cut_bbox.left):math.ceil(cut_bbox.right)
        ])

    def paste(self, image: 'CpuImage', point: Tuple[int, int]):
        """Pastes image on current image.

        :param image: image to paste
        """
        self._image_np[point[1]:point[1]+image.height, point[0]:point[0]+image.width] \
            = image._image_np


class GpuImage(Image):

    def __init__(self, image: cv2.cuda.GpuMat):
        """Image data container.

        :param image: image data
        """
        self._gpu_image = image
        self._width, self._height = self._gpu_image.size()

    @property
    def gpu_mat(self):
        return self._gpu_image

    @property
    def width(self) -> int:
        """Returns image width.
        :return: image width
        """
        return self._gpu_image.size()[0]

    @property
    def height(self) -> int:
        """Returns image height.
        :return: image height
        """
        return self._gpu_image.size()[1]

    def to_cpu(self) -> ImageCpu:
        return self._gpu_image.download()

    def cut(self, bbox: Union[BBox, RBBox], ) -> Tuple['GpuImage', Union[BBox, RBBox]]:
        """Cuts image by bbox and padding.
        :param bbox: cutout bbox
        :return:
        """
        if isinstance(bbox, BBox):
            cut_left = int(bbox.left)
            cut_right = int(bbox.right)
            cut_top = int(bbox.top)
            cut_bottom = int(bbox.bottom)
            assert cut_left >= 0 and cut_right < self._width \
                and cut_top >= 0 and cut_bottom < self._height, \
                f"bbox (left={bbox.left}, right={bbox.right}, top={bbox.top}, " \
                f"bottom={bbox.bottom}) is out of " \
                f"image size {self._width}x{self._height}"

            cut_roi = self._gpu_image.colRange(cut_left, cut_right).\
                rowRange(cut_top, cut_bottom)
            res_image = cv2.cuda.GpuMat(size=cut_roi.size(), type=cut_roi.type())
            cut_roi.copyTo(res_image)
            return \
                GpuImage(res_image), \
                BBox(
                    x_center=cut_roi.size()[0] / 2,
                    y_center=cut_roi.size()[1] / 2,
                    width=cut_roi.size()[0],
                    height=cut_roi.size()[1],
                )

        elif isinstance(bbox, RBBox):
            aligned_bbox = bbox.to_bbox()
            start_row = int(aligned_bbox.top)
            end_row = int(aligned_bbox.bottom)
            start_col = int(aligned_bbox.left)
            end_col = int(aligned_bbox.right)
            if start_row < 0 or start_col < 0 or end_col >= self._width \
                    or end_row >= self._height:
                res_image = cv2.cuda.GpuMat(
                    rows=end_row - start_row + 1,
                    cols=end_col - start_col + 1,
                    type=self._gpu_image.type()
                )
                cut_roi = self._gpu_image \
                    .colRange(
                        max(start_col, 0),
                        min(end_col, self._width-1))\
                    .rowRange(
                        max(start_row, 0),
                        min(end_row, self._height-1))
                res_start_col = max(-start_col, 0)
                res_start_row = max(-start_row, 0)
                cut_roi.copyTo(
                    res_image\
                        .colRange(res_start_col, res_start_col+cut_roi.size()[0])\
                        .rowRange(res_start_row, res_start_row+cut_roi.size()[1])
                )
            else:
                cut_roi = self._gpu_image.colRange(start_col, end_col)\
                    .rowRange(start_row, end_row)
                res_image = cv2.cuda.GpuMat(size=cut_roi.size(), type=cut_roi.type())
                cut_roi.copyTo(res_image)

            res_bbox = RBBox(
                x_center=res_image.size()[0] / 2,
                y_center=res_image.size()[1] / 2,
                width=bbox.width,
                height=bbox.height,
                angle=bbox.angle,
            )

            return GpuImage(res_image), res_bbox
        raise ValueError(f"Unknown bbox type {type(bbox)}")

    def concat(self, image: 'GpuImage', axis: int = 0) -> 'GpuImage':
        """Concatenates images along axis.

        :param image: image to concatenate
        :param axis: axis to concatenate. 0 - is vertical, 1 - is horizontal
        :return: concatenated image
        """
        assert self.gpu_mat.type() == image.gpu_mat.type(), \
            f"Images have different types {self.gpu_mat.type()} != " \
            f"{image.gpu_mat.type()}"
        if axis == 0:
            assert self.width == image.width, \
                f"Images have different height {self.width} != {image.height}"

            res_rows = self.height + image.height
            res_image = cv2.cuda.GpuMat(
                size=(self.width, res_rows),
                type=self._gpu_image.type()
            )
            self.gpu_mat.copyTo(dst=res_image.rowRange(0, self.height))
            image.gpu_mat.copyTo(dst=res_image.rowRange(self.height, res_rows))
            return GpuImage(res_image)
        elif axis == 1:
            assert self.height == image.height, \
                f"Images have different width {self.height} != {image.height}"
            res_cols = self.width + image.width
            res_image = cv2.cuda.GpuMat(
                size=(res_cols, self.height),
                type=self._gpu_image.type()
            )
            self.gpu_mat.copyTo(dst=res_image.colRange(0, self.width))
            image.gpu_mat.copyTo(dst=res_image.colRange(self.width, res_cols))
            return GpuImage(res_image)
        else:
            raise ValueError(f"Unknown axis {axis}")

    def paste(self, image: 'GpuImage', point: Tuple[int, int]):
        """Pastes image on current image.

        :param image: image to paste
        :param point: point to paste (x, y)
        """

        frame_roi = self._gpu_image.colRange(point[0], point[0] + image.width).\
            rowRange(point[1], point[1]+image.height)
        image.gpu_mat.copyTo(frame_roi)

    def rotate(self, angle: float, bbox: Optional[RBBox] = None) -> Tuple['GpuImage', RBBox]:
        """Rotates image on angle. If.

        :param angle: angle to rotate in degrees
        :param bbox: image will be rotated on bbox.angle if bbox is not None
        :return: rotated image
        """

        bbox_image = RBBox(
            self.width // 2,
            self.height // 2,
            self.width,
            self.height,
            angle
        )

        polygon = bbox_image.polygons()
        width_new = int(math.ceil(max(polygon[:, 0] - min(polygon[:, 0]))))
        height_new = int(math.ceil(max(polygon[:, 1] - min(polygon[:, 1]))))
        resolution = (width_new, height_new)
        rotation_matrix = cv2.getRotationMatrix2D(
            (bbox.x_center, bbox.y_center),
            angle,
            1
        )
        rotation_matrix[0, 2] -= min(polygon[:, 0])
        rotation_matrix[1, 2] -= min(polygon[:, 1])
        res = cv2.cuda.warpAffine(
            self._gpu_image,
            rotation_matrix,
            resolution
        )

        if bbox is not None:
            return GpuImage(res), \
                RBBox(
                    x_center=width_new / 2,
                    y_center=height_new / 2,
                    width=bbox.width,
                    height=bbox.height,
                    angle=bbox.angle - angle
                )
        return GpuImage(res), \
            RBBox(
                x_center=width_new / 2,
                y_center=height_new / 2,
                width=bbox_image.width,
                height=bbox_image.height,
                angle=bbox_image.angle
            )


class BasePreprocessObjectImage(BasePyFuncCallableImpl):
    """Object image preprocessing interface."""
    @abstractmethod
    def __call__(
           self,
           object_meta: ObjectMeta,
           frame_image: Image,
    ) -> Image:
        """Transforms object image.

        :param object_meta: object meta
        :param frame_image: original image
        :return: changed image
        """


class ObjectsPreprocessing:

    def __init__(self):
        self._preprocessing_functions = {}
        self._frames_map = {}

    def add_preprocessing_function(
            self,
            element_name: str,
            preprocessing_func: Callable[[ObjectMeta, Image], Image]
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
            max_object_image_size: Optional[int] = None,
    ):
        """Preprocesses objects by using user function.

        :param element_name: Element name for which preprocessing is done
        :param buffer: gst buffer
        :param model_uid: base on model uid is selected preprocessing object
        :param class_id: base on class id is selected preprocessing object
        :param max_object_image_size:  max object image size for inference
        :return:
        """
        preprocessing_func = self._preprocessing_functions.get(element_name)
        if preprocessing_func is None:
            raise ValueError(f"Cannot find preprocessing function for {element_name}")

        self._frames_map[buffer] = {}

        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            left = 0
            top = 0
            row_height = 0
            with nvds_to_gpu_mat(buffer, nvds_frame_meta) as frame_mat:
                frame_image = GpuImage(frame_mat)
                copy_frame_image = GpuImage(frame_mat.clone())
                self._frames_map[buffer][nvds_frame_meta.batch_id] = copy_frame_image
                for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                    if nvds_obj_meta.class_id != class_id:
                        continue
                    if nvds_obj_meta.unique_component_id != model_uid:
                        continue
                    object_meta = _NvDsObjectMetaImpl.from_nv_ds_object_meta(
                        object_meta=nvds_obj_meta,
                        frame_meta=nvds_frame_meta
                    )

                    preprocess_image = preprocessing_func(object_meta, copy_frame_image)

                    if not isinstance(preprocess_image, Image):
                        raise ValueError("Preprocessing function must "
                                         "return Image object.")

                    if left + preprocess_image.width > frame_image.width:
                        left = 0
                        if row_height == 0:
                            row_height = preprocess_image.height
                        top += row_height
                        row_height = 0
                    if top >= frame_image.height:
                        raise ValueError("There is no place on frame "
                                         "to put object image.")
                    if top + preprocess_image.height > frame_image.height:
                        raise ValueError("There is no place on frame "
                                         "to put object image.")
                    if preprocess_image.height > row_height:
                        row_height = preprocess_image.height

                    frame_image.paste(preprocess_image, (left, top))
                    left += preprocess_image.width

    def restore_frame(self, buffer: Gst.Buffer):
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            restore_frame_image = self._frames_map[buffer].pop(nvds_frame_meta.batch_id)
            with nvds_to_gpu_mat(buffer, nvds_frame_meta) as frame_mat:
                restore_frame_image.gpu_mat.copyTo(frame_mat)

        self._frames_map.pop(buffer)



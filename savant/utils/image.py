import math
from typing import Union, Tuple, Optional

import cv2
import numpy as np

from savant.meta.bbox import BBox, RBBox


class CPUImage:
    def __init__(self, image: np.ndarray):
        """Image data container.

        :param image: image data
        """
        self._np_image = image

    @property
    def np_array(self) -> np.ndarray:
        """Returns numpy image.

        :return: numpy image
        """
        return self._np_image

    @property
    def to_gpu(self) -> cv2.cuda.GpuMat:
        """Returns GPU image.

        :return: GPU image
        """
        image_gpu = cv2.cuda.GpuMat(self._np_image.shape)
        image_gpu.upload(self._np_image)
        return image_gpu

    @property
    def height(self) -> int:
        """Returns image height.

        :return: image height
        """
        return self._np_image.shape[0]

    @property
    def width(self) -> int:
        """Returns image width.

        :return: image width
        """
        return self._np_image.shape[1]

    def cut(self, bbox: Union[BBox, RBBox]) -> Tuple['CPUImage', Union[BBox, RBBox]]:
        """Cuts image by bbox and padding.
        :param bbox: cutout bbox
        :return:
        """
        if isinstance(bbox, BBox):
            cutout_box = bbox
            res_bbox = BBox(
                x_center=0,
                y_center=0,
                width=bbox.width,
                height=bbox.height,
            )
        elif isinstance(bbox, RBBox):
            cutout_box = bbox.to_bbox()
            res_bbox = RBBox(
                x_center=0,
                y_center=0,
                width=bbox.width,
                height=bbox.height,
                angle=bbox.angle,
            )
        else:
            raise TypeError(f'Unknown type {type(bbox)}')

        # We need to ceil and floor values to avoid errors.
        # We will take rows and cols from start to end don't include end
        start_row = math.ceil(cutout_box.top)
        end_row = math.floor(cutout_box.bottom)
        start_col = math.ceil(cutout_box.left)
        end_col = math.floor(cutout_box.right)

        if (
                start_row < 0
                or start_col < 0
                or end_col > self.width
                or end_row > self.height
        ):
            res_image = np.zeros(
                shape=[
                    end_row - start_row + 1,
                    end_col - start_col + 1,
                    self._np_image.shape[2],
                ],
                dtype=np.uint8
            )
            intersection_start_row = max(0, start_row)
            intersection_end_row = min(end_row, self.height)
            intersection_start_col = max(0, start_col)
            intersection_end_col = min(end_col, self.width)
            # compute the area of intersection rectangle
            if max(0, intersection_end_row - intersection_start_row) * \
                    max(0, intersection_end_col - intersection_start_col) > 0:
                cut_roi = self._np_image[
                    max(start_row, 0): min(end_row, self.height - 1),
                    max(start_col, 0): min(end_col, self.width - 1),
                    :,
                ]
                res_start_col = max(-start_col, 0)
                res_start_row = max(-start_row, 0)
                res_image[
                    res_start_row: res_start_row + cut_roi.shape[0],
                    res_start_col: res_start_col + cut_roi.shape[1],
                    :,
                ] = cut_roi
        else:
            res_image = self._np_image[start_row:end_row, start_col:end_col, :]

        res_bbox.x_center = res_image.shape[1] / 2
        res_bbox.y_center = res_image.shape[0] / 2
        return CPUImage(res_image), res_bbox

    def paste(self, image: 'CPUImage', point: Tuple[int, int]):
        """Pastes image on current image.

        :param image: image to paste
        """
        self._np_image[
            point[1] : point[1] + image.height, point[0] : point[0] + image.width
        ] = image._np_image

    def concat(self, image: 'CPUImage', axis: int = 0) -> 'CPUImage':
        """Concatenates images along axis.

        :param image: image to concatenate
        :param axis: axis to concatenate. 0 - is vertical, 1 - is horizontal
        :return: concatenated image
        """
        if self._np_image.dtype != image.np_array.dtype:
            raise ValueError(
                f'Images have different types {self._np_image.dtype } != '
                f'{image.np_array.dtype}'
            )
        if axis == 0:
            if self.width != image.width:
                raise ValueError(f'Images have different '
                                 f'height {self.width} != {image.height}')

            return CPUImage(
                image=np.concatenate([self._np_image, image.np_array], axis=0)
            )
        elif axis == 1:
            if self.height != image.height:
                raise ValueError(f'Images have different '
                                 f'width {self.height} != {image.height}')
            return CPUImage(
                image=np.concatenate([self._np_image, image.np_array], axis=1)
            )
        else:
            raise ValueError(f'Unknown axis {axis}')

    def rotate(
        self, angle: float, bbox: Optional[RBBox] = None
    ) -> Tuple['CPUImage', RBBox]:
        """Rotates image on angle.

        :param angle: angle to rotate in degrees
        :param bbox: image will be rotated on bbox.angle if bbox is not None
        :return: rotated image
        """

        if bbox is not None:
            rotation_matrix, resolution = get_rotation_matrix(
                self, angle, (bbox.x_center, bbox.y_center)
            )
        else:
            rotation_matrix, resolution = get_rotation_matrix(
                self, angle, (self.width // 2, self.height // 2)
            )
        res = cv2.warpAffine(src=self._np_image, M=rotation_matrix, dsize=resolution)

        if bbox is not None:
            return CPUImage(res), RBBox(
                x_center=resolution[0] / 2,
                y_center=resolution[1] / 2,
                width=bbox.width,
                height=bbox.height,
                angle=bbox.angle - angle,
            )
        return CPUImage(image=res), RBBox(
            x_center=resolution[0] / 2,
            y_center=resolution[1] / 2,
            width=self.width,
            height=self.height,
            angle=angle,
        )

    def resize(
        self,
        resolution: Tuple[int, int],
        method: str = 'fit',
        interpolation: int = cv2.INTER_LINEAR,
    ) -> 'CPUImage':
        """Resizes image to resolution.

        :param resolution: resolution to resize [width, height]
        :param method: method to resize one of ['fit', 'scale']
        :param interpolation: interpolation method.
            Oone of [cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LINEAR,
            cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        :return: resized image
        """
        new_resolution = tuple(map(int, resolution))
        if method == 'scale':
            scale_factor = min(resolution[0] / self.width, resolution[1] / self.height)
            new_resolution = (
                int(self.width * scale_factor),
                int(self.height * scale_factor),
            )
        resized_image = cv2.resize(
            src=self._np_image, dsize=new_resolution, interpolation=interpolation
        )
        res = np.zeros(
            shape=(resolution[0], resolution[1], self._np_image.shape[2]),
            dtype=self._np_image.dtype,
        )
        start_col = (resolution[0] - new_resolution[0]) // 2
        start_row = (resolution[1] - new_resolution[1]) // 2
        res[
            start_row : start_row + new_resolution[1],
            start_col : start_col + new_resolution[0],
            :,
        ] = resized_image
        return CPUImage(image=res)


class GPUImage:
    """Image data container. GPU version."""

    def __init__(
        self,
        image: Union[cv2.cuda.GpuMat, np.ndarray, CPUImage],
        cuda_stream: Optional[cv2.cuda.Stream] = cv2.cuda.Stream_Null(),
    ):
        """Image data container.

        :param image: image data
        :param cuda_stream: cuda stream
        """
        if isinstance(image, np.ndarray):
            gpu_image = cv2.cuda_GpuMat(image)
            self._gpu_image = gpu_image
        elif isinstance(image, CPUImage):
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image.np_array)
            self._gpu_image = gpu_image
        elif isinstance(image, cv2.cuda_GpuMat):
            self._gpu_image = image
        else:
            raise TypeError(f'Unknown type {type(image)}')
        self._cuda_stream = cuda_stream

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

    def to_cpu(self) -> CPUImage:
        return CPUImage(self._gpu_image.download())

    def cut(self, bbox: Union[BBox, RBBox]) -> Tuple['GPUImage', Union[BBox, RBBox]]:
        """Cuts image by bbox
        :param bbox: cutout bbox
        :return:
        """
        if isinstance(bbox, BBox):
            cutout_box = bbox
            res_bbox = BBox(
                x_center=0,
                y_center=0,
                width=bbox.width,
                height=bbox.height,
            )
        elif isinstance(bbox, RBBox):
            cutout_box = bbox.to_bbox()
            res_bbox = RBBox(
                x_center=0,
                y_center=0,
                width=bbox.width,
                height=bbox.height,
                angle=bbox.angle,
            )
        else:
            raise TypeError(f'Unknown type {type(bbox)}')

        # We need to ceil and floor values to avoid errors.
        # We will take rows and cols from start to end don't include end
        start_row = math.ceil(cutout_box.top)
        end_row = math.floor(cutout_box.bottom)
        start_col = math.ceil(cutout_box.left)
        end_col = math.floor(cutout_box.right)

        if (
            start_row < 0
            or start_col < 0
            or end_col > self.width
            or end_row > self.height
        ):
            res_image = cv2.cuda.GpuMat(
                rows=end_row - start_row,
                cols=end_col - start_col,
                type=self._gpu_image.type(),
                s=0
            )
            intersection_start_row = max(0, start_row)
            intersection_end_row = min(end_row, self.height)
            intersection_start_col = max(0, start_col)
            intersection_end_col = min(end_col, self.width)
            # compute the area of intersection rectangle
            if max(0, intersection_end_row - intersection_start_row) * \
                    max(0, intersection_end_col - intersection_start_col) > 0:
                cut_roi = self._gpu_image\
                    .colRange(intersection_start_col, intersection_end_col)\
                    .rowRange(intersection_start_row, intersection_end_row)
                res_start_col = max(-start_col, 0)
                res_start_row = max(-start_row, 0)
                roi_res_image = res_image.colRange(
                    res_start_col, res_start_col + cut_roi.size()[0]
                ).rowRange(res_start_row, res_start_row + cut_roi.size()[1])
                cut_roi.copyTo(stream=self._cuda_stream, dst=roi_res_image)
        else:
            cut_roi = self._gpu_image.colRange(start_col, end_col).rowRange(
                start_row, end_row
            )
            res_image = cv2.cuda.GpuMat(size=cut_roi.size(), type=cut_roi.type())
            cut_roi.copyTo(stream=self._cuda_stream, dst=res_image)

        res_bbox.x_center = res_image.size()[0] / 2
        res_bbox.y_center = res_image.size()[1] / 2
        return GPUImage(res_image, cuda_stream=self._cuda_stream), res_bbox

    def concat(self, image: 'GPUImage', axis: int = 0) -> 'GPUImage':
        """Concatenates images along axis.

        :param image: image to concatenate
        :param axis: axis to concatenate. 0 - is vertical, 1 - is horizontal
        :return: concatenated image
        """
        if self.gpu_mat.type() != image.gpu_mat.type():
            raise ValueError(
                f'Images have different types {self.gpu_mat.type()} != '
                f'{image.gpu_mat.type()}'
            )
        if axis == 0:
            if self.width != image.width:
                raise ValueError(f'Images have different '
                                 f'height {self.width} != {image.height}')

            res_rows = self.height + image.height
            res_image = cv2.cuda.GpuMat(
                size=(self.width, res_rows), type=self._gpu_image.type()
            )
            self.gpu_mat.copyTo(
                stream=self._cuda_stream, dst=res_image.rowRange(0, self.height)
            )
            image.gpu_mat.copyTo(
                stream=self._cuda_stream, dst=res_image.rowRange(self.height, res_rows)
            )
            return GPUImage(image=res_image, cuda_stream=self._cuda_stream)
        elif axis == 1:
            if self.height != image.height:
                raise ValueError(f'Images have different '
                                 f'width {self.height} != {image.height}')
            res_cols = self.width + image.width
            res_image = cv2.cuda.GpuMat(
                size=(res_cols, self.height), type=self._gpu_image.type()
            )
            self.gpu_mat.copyTo(
                stream=self._cuda_stream, dst=res_image.colRange(0, self.width)
            )
            image.gpu_mat.copyTo(
                stream=self._cuda_stream, dst=res_image.colRange(self.width, res_cols)
            )
            return GPUImage(image=res_image, cuda_stream=self._cuda_stream)
        else:
            raise ValueError(f'Unknown axis {axis}')

    def paste(self, image: 'GPUImage', point: Tuple[int, int]):
        """Pastes image on current image.

        :param image: image to paste
        :param point: point to paste (x, y)
        """
        if (
            point[0] < 0
            or point[1] < 0
            or point[0] >= self.width
            or point[1] >= self.height
        ):
            raise ValueError(
                f'Point {point} is out of image {self.width}x{self.height}'
            )
        insert_width = min(image.width, self.width - point[0])
        insert_height = min(image.height, self.height - point[1])
        frame_roi = self._gpu_image.colRange(
            point[0], point[0] + insert_width
        ).rowRange(point[1], point[1] + insert_height)
        image.gpu_mat.rowRange(0, insert_height).colRange(0, insert_width).copyTo(
            stream=self._cuda_stream, dst=frame_roi
        )

    def rotate(
        self, angle: float, bbox: Optional[RBBox] = None
    ) -> Tuple['GPUImage', RBBox]:
        """Rotates image on angle. If.

        :param angle: angle to rotate in degrees
        :param bbox: bounding box on image
        :return: rotated image
        """
        rotation_matrix, resolution = get_rotation_matrix(
            self, angle, (int(bbox.x_center), int(bbox.y_center))
        )

        res = cv2.cuda.warpAffine(
            src=self._gpu_image,
            M=rotation_matrix,
            dsize=resolution,
            stream=self._cuda_stream,
        )

        if bbox is not None:
            return GPUImage(image=res, cuda_stream=self._cuda_stream), RBBox(
                x_center=resolution[0] / 2,
                y_center=resolution[1] / 2,
                width=bbox.width,
                height=bbox.height,
                angle=bbox.angle - angle,
            )
        return GPUImage(image=res, cuda_stream=self._cuda_stream), RBBox(
            x_center=resolution[0] / 2,
            y_center=resolution[1] / 2,
            width=self.width,
            height=self.height,
            angle=angle,
        )

    def resize(
        self,
        resolution: Tuple[int, int],
        method: str = 'fit',
        interpolation: int = cv2.INTER_LINEAR,
    ) -> 'GPUImage':
        """Resizes image to resolution.

        :param resolution: resolution to resize [width, height]
        :param method: method to resize one of ['fit', 'scale']
        :param interpolation: interpolation method.
            Oone of [cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LINEAR,
            cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        :return: resized image
        """
        new_resolution = resolution
        if method == 'scale':
            scale_factor = min(resolution[0] / self.width, resolution[1] / self.height)
            new_resolution = (
                int(self.width * scale_factor),
                int(self.height * scale_factor),
            )
        res = cv2.cuda.GpuMat(size=resolution, type=self._gpu_image.type(), s=0)
        start_col = (resolution[0] - new_resolution[0]) // 2
        start_row = (resolution[1] - new_resolution[1]) // 2
        res_roi = res.colRange(start_col, start_col + new_resolution[0]).rowRange(
            start_row, start_row + new_resolution[1]
        )
        resized_image = cv2.cuda.resize(
            src=self._gpu_image,
            dst=res_roi,
            dsize=new_resolution,
            interpolation=interpolation,
            stream=self._cuda_stream,
        )
        resized_image.copyTo(stream=self._cuda_stream, dst=res_roi)
        return GPUImage(image=res, cuda_stream=self._cuda_stream)


def get_rotation_matrix(
    image: Union[CPUImage, GPUImage], angle: float, rotation_point: Tuple[float, float]
):
    """Returns rotation matrix and new resolution of image.

    :param image: image to rotate
    :param angle: angle to rotate in degrees
    :param rotation_point: point to rotate (x, y)
    """

    bbox_image = RBBox(
        image.width // 2, image.height // 2, image.width, image.height, angle
    )
    polygon = bbox_image.polygon()
    width_new = int(math.ceil(max(polygon[:, 0] - min(polygon[:, 0]))))
    height_new = int(math.ceil(max(polygon[:, 1] - min(polygon[:, 1]))))
    resolution = (width_new, height_new)
    rotation_matrix = cv2.getRotationMatrix2D(
        (rotation_point[0], rotation_point[1]), angle, 1
    )
    rotation_matrix[0, 2] -= min(polygon[:, 0])
    rotation_matrix[1, 2] -= min(polygon[:, 1])
    return rotation_matrix, resolution

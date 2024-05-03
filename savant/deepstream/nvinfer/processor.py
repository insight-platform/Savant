import logging
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pyds
from pygstsavantframemeta import (
    gst_buffer_get_savant_batch_meta,
    nvds_frame_meta_get_nvds_savant_frame_meta,
)
from savant_rs.pipeline2 import VideoPipeline
from savant_rs.primitives.geometry import BBox
from savant_rs.utils.symbol_mapper import (
    build_model_object_key,
    get_model_id,
    get_object_id,
    parse_compound_key,
)

from savant.base.converter import TensorFormat
from savant.base.input_preproc import ObjectsPreprocessing
from savant.base.pyfunc import PyFuncNoopCallException
from savant.config.schema import FrameParameters, ModelElement
from savant.deepstream.meta.object import _NvDsObjectMetaImpl
from savant.deepstream.nvinfer.element_config import MERGED_CLASSES
from savant.deepstream.nvinfer.model import (
    NvInferAttributeModel,
    NvInferComplexModel,
    NvInferDetector,
)
from savant.deepstream.utils.attribute import (
    nvds_add_attr_meta_to_obj,
    nvds_attr_meta_iterator,
)
from savant.deepstream.utils.iterator import (
    nvds_clf_meta_iterator,
    nvds_frame_meta_iterator,
    nvds_label_info_iterator,
    nvds_obj_meta_iterator,
    nvds_tensor_output_iterator,
)
from savant.deepstream.utils.object import (
    nvds_add_obj_meta_to_frame,
    nvds_set_obj_selection_type,
    nvds_set_obj_uid,
)
from savant.deepstream.utils.tensor import (
    nvds_infer_tensor_meta_to_outputs,
    nvds_infer_tensor_meta_to_outputs_cupy,
)
from savant.gstreamer import Gst  # noqa:F401
from savant.meta.errors import UIDError
from savant.meta.object import ObjectMeta
from savant.meta.type import ObjectSelectionType
from savant.utils.logging import get_logger


class NvInferProcessor:
    """NvInfer element processor.
    Performs nvinfer model pre- and post-processing.
    """

    def __init__(
        self,
        element: ModelElement,
        objects_preprocessing: ObjectsPreprocessing,
        frame_params: FrameParameters,
        video_pipeline: VideoPipeline,
    ):
        self._logger = get_logger(
            f'{self.__class__.__module__}.{self.__class__.__name__}'
        )

        # c++ object preprocessing
        self._objects_preprocessing = objects_preprocessing

        # frame rect to clip objects
        self._frame_rect = (  # format: left-top-right-bottom
            0.0,
            0.0,
            frame_params.width - 1.0,
            frame_params.height - 1.0,
        )
        if frame_params.padding:
            self._frame_rect = (
                self._frame_rect[0] + frame_params.padding.left,
                self._frame_rect[1] + frame_params.padding.top,
                self._frame_rect[2] + frame_params.padding.left,
                self._frame_rect[3] + frame_params.padding.top,
            )

        # video pipeline (frame/source info, telemetry span etc.)
        self._video_pipeline = video_pipeline

        self._element_name = element.name

        self._model: Union[
            NvInferAttributeModel, NvInferComplexModel, NvInferDetector
        ] = element.model
        self._is_attribute_model = isinstance(self._model, NvInferAttributeModel)
        self._is_complex_model = isinstance(self._model, NvInferComplexModel)
        self._is_object_model = isinstance(self._model, NvInferDetector)

        self._model_uid = get_model_id(self._element_name)

        input_object_model_name, input_object_label = parse_compound_key(
            self._model.input.object
        )
        self._input_object_model_uid, self._input_object_class_id = get_object_id(
            input_object_model_name, input_object_label
        )

        # preprocessor
        self.preproc: Optional[Callable] = None

        # postprocessor
        self.postproc: Optional[Callable] = None

        # no operation
        def no_op(*args):
            pass

        # to finalize preproc in postproc
        self._restore_object_meta = no_op
        self._restore_frame = no_op

        # setup pre- and post-processor
        if self._model.input.preprocess_object_meta:
            self.preproc = self._preprocess_object_meta
            self._restore_object_meta = self._restore_object_meta_

        elif self._model.input.preprocess_object_image:
            self._objects_preprocessing.add_preprocessing_function(
                element_name=self._element_name,
                preprocessing_func=self._model.input.preprocess_object_image,
            )
            self.preproc = self._preprocess_object_image
            self._restore_object_meta = self._restore_object_meta_
            self._restore_frame = self._restore_frame_

        if self._model.output.converter:
            self.postproc = self._process_custom_model_output
            self._tensor_meta_to_outputs = nvds_infer_tensor_meta_to_outputs
            if self._model.output.converter.instance.tensor_format == TensorFormat.CuPy:
                self._tensor_meta_to_outputs = nvds_infer_tensor_meta_to_outputs_cupy

        elif self._is_object_model:
            self.postproc = self._process_regular_detector_output

        elif self._is_attribute_model:
            self.postproc = self._process_regular_classifier_output

    def _preprocess_object_meta(self, buffer: Gst.Buffer):
        """Preprocesses input object metadata."""
        self._logger.debug(
            'Preprocessing "%s" element input object meta, buffer PTS %s.',
            self._element_name,
            buffer.pts,
        )
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                if not self._is_model_input_object(nvds_obj_meta):
                    continue
                # TODO: Unify and also switch to the box representation system
                #  through the center point during meta preprocessing.
                object_meta = _NvDsObjectMetaImpl.from_nv_ds_object_meta(
                    nvds_obj_meta, nvds_frame_meta
                )
                if not isinstance(object_meta.bbox, BBox):
                    raise NotImplementedError(
                        'Preprocessing only supports objects represented '
                        'by regular (axis-aligned) boxes.'
                    )

                user_parent_object_meta = None
                if object_meta.parent:
                    parent_object_meta = object_meta.parent

                    if not isinstance(parent_object_meta.bbox, BBox):
                        raise NotImplementedError(
                            'Preprocessing only supports objects whose parents are '
                            'represented by regular (axis-aligned) boxes.'
                        )

                    if parent_object_meta.parent:
                        self._logger.warning(
                            'Preprocessing only supports 1 level of hierarchy.'
                        )

                    user_parent_object_meta = ObjectMeta(
                        parent_object_meta.element_name,
                        parent_object_meta.label,
                        parent_object_meta.bbox.copy(),
                        parent_object_meta.confidence,
                        parent_object_meta.track_id,
                        attributes=nvds_attr_meta_iterator(
                            nvds_frame_meta,
                            parent_object_meta.ds_object_meta,
                        ),
                    )

                else:
                    source_id, frame_idx = self._get_frame_source_id_and_idx(
                        buffer,
                        nvds_frame_meta,
                    )
                    self._logger.warning(
                        'The object (%s.%s, bbox %s) of a frame %s/%s with IDX %s '
                        'is an orphan (no parent object is assigned). '
                        'It is a non-typical case: the object should either '
                        'have the frame or an ROI object as a parent.',
                        object_meta.element_name,
                        object_meta.label,
                        object_meta.bbox.as_ltrb_int(),
                        source_id,
                        nvds_frame_meta.buf_pts,
                        frame_idx,
                    )

                user_object_meta = ObjectMeta(
                    object_meta.element_name,
                    object_meta.label,
                    object_meta.bbox.copy(),
                    object_meta.confidence,
                    object_meta.track_id,
                    user_parent_object_meta,
                    attributes=nvds_attr_meta_iterator(
                        frame_meta=nvds_frame_meta,
                        obj_meta=object_meta.ds_object_meta,
                    ),
                )

                try:
                    res_bbox = self._model.input.preprocess_object_meta(
                        object_meta=user_object_meta
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    if self._model.input.preprocess_object_meta.dev_mode:
                        if not isinstance(exc, PyFuncNoopCallException):
                            self._logger.exception(
                                'Error calling preprocess input object meta.'
                            )
                        res_bbox = user_object_meta.bbox
                    else:
                        raise exc

                if self._logger.isEnabledFor(logging.TRACE):
                    self._logger.trace(
                        'Preprocessing "%s" object bbox %s -> %s',
                        user_object_meta.label,
                        user_object_meta.bbox,
                        res_bbox,
                    )

                rect_params = nvds_obj_meta.rect_params
                rect_params.left = res_bbox.left
                rect_params.top = res_bbox.top
                rect_params.width = res_bbox.width
                rect_params.height = res_bbox.height

    def _restore_object_meta_(self, nvds_obj_meta: pyds.NvDsObjectMeta):
        if self._is_model_input_object(nvds_obj_meta):
            bbox_coords = nvds_obj_meta.detector_bbox_info.org_bbox_coords
            if nvds_obj_meta.tracker_bbox_info.org_bbox_coords.width > 0:
                bbox_coords = nvds_obj_meta.tracker_bbox_info.org_bbox_coords
            rect_params = nvds_obj_meta.rect_params
            rect_params.left = bbox_coords.left
            rect_params.top = bbox_coords.top
            rect_params.width = bbox_coords.width
            rect_params.height = bbox_coords.height

    def _preprocess_object_image(self, buffer: Gst.Buffer):
        """Preprocesses input object image."""
        self._logger.debug(
            'Preprocessing "%s" element input object image, buffer PTS %s.',
            self._element_name,
            buffer.pts,
        )
        self._objects_preprocessing.preprocessing(
            self._element_name,
            hash(buffer),
            self._input_object_model_uid,
            self._input_object_class_id,
            self._model.input.preprocess_object_image.output_image,
            self._model.input.preprocess_object_image.dev_mode,
        )

    def _restore_frame_(self, buffer: Gst.Buffer):
        self._objects_preprocessing.restore_frame(hash(buffer))

    def _process_custom_model_output(self, buffer: Gst.Buffer):
        """Processes custom model output (converter wrapper)."""
        self._logger.debug(
            'Preparing "%s" element custom model output, buffer PTS %s.',
            self._element_name,
            buffer.pts,
        )
        self._model: Union[NvInferAttributeModel, NvInferComplexModel]
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                self._restore_object_meta(nvds_obj_meta)
                if not self._is_model_input_object(nvds_obj_meta):
                    continue

                parent_nvds_obj_meta = nvds_obj_meta
                for tensor_meta in nvds_tensor_output_iterator(
                    parent_nvds_obj_meta, gie_uid=self._model_uid
                ):
                    if self._logger.isEnabledFor(logging.TRACE):
                        self._logger.trace(
                            'Converting "%s" element tensor output for frame '
                            'with PTS %s.',
                            self._element_name,
                            nvds_frame_meta.buf_pts,
                        )
                    # parse and post-process model output
                    output_layers = self._tensor_meta_to_outputs(
                        tensor_meta=tensor_meta,
                        layer_names=self._model.output.layer_names,
                    )
                    try:
                        outputs = self._model.output.converter(
                            *output_layers,
                            model=self._model,
                            roi=(
                                parent_nvds_obj_meta.rect_params.left,
                                parent_nvds_obj_meta.rect_params.top,
                                parent_nvds_obj_meta.rect_params.width,
                                parent_nvds_obj_meta.rect_params.height,
                            ),
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        if self._model.output.converter.dev_mode:
                            if not isinstance(exc, PyFuncNoopCallException):
                                self._logger.exception('Error calling converter')
                            # provide some placeholders
                            # so that the pipeline processing can continue
                            if self._is_complex_model:
                                outputs = np.zeros((0, 6)), np.zeros((0, 1))
                            elif self._is_object_model:
                                outputs = np.zeros((0, 6))
                            else:
                                outputs = np.zeros((0, 1))
                        else:
                            raise exc
                    # for object/complex models output - `bbox_tensor` and
                    # `selected_bboxes` - indices of selected bboxes and meta
                    # for attribute/complex models output - `values`
                    bbox_tensor: Optional[np.ndarray] = None
                    selected_bboxes: Optional[List] = None
                    values: Optional[List] = None
                    # complex model
                    if self._is_complex_model:
                        # output converter returns tensor and attribute values
                        bbox_tensor, values = outputs
                        assert bbox_tensor.shape[0] == len(
                            values
                        ), 'Number of detected boxes and attributes do not match.'

                    # object model
                    elif self._is_object_model:
                        # output converter returns tensor with
                        # (class_id, confidence, xc, yc, width, height, [angle]),
                        # coordinates in roi cloud (parent object cloud)
                        bbox_tensor = outputs

                    # attribute model
                    else:
                        # output converter returns attribute values
                        values = outputs

                    if bbox_tensor is not None and bbox_tensor.shape[0] > 0:
                        # object or complex model with non-empty output
                        if bbox_tensor.shape[1] == 6:  # no angle
                            selection_type = ObjectSelectionType.REGULAR_BBOX

                            # xc -> left, yc -> top
                            bbox_tensor[:, 2] -= bbox_tensor[:, 4] / 2
                            bbox_tensor[:, 3] -= bbox_tensor[:, 5] / 2

                            # width to right, height to bottom
                            bbox_tensor[:, 4] += bbox_tensor[:, 2]
                            bbox_tensor[:, 5] += bbox_tensor[:, 3]

                            # clip
                            bbox_tensor[:, 2][
                                bbox_tensor[:, 2] < self._frame_rect[0]
                            ] = self._frame_rect[0]
                            bbox_tensor[:, 3][
                                bbox_tensor[:, 3] < self._frame_rect[1]
                            ] = self._frame_rect[1]
                            bbox_tensor[:, 4][
                                bbox_tensor[:, 4] > self._frame_rect[2]
                            ] = self._frame_rect[2]
                            bbox_tensor[:, 5][
                                bbox_tensor[:, 5] > self._frame_rect[3]
                            ] = self._frame_rect[3]

                            # right to width, bottom to height
                            bbox_tensor[:, 4] -= bbox_tensor[:, 2]
                            bbox_tensor[:, 5] -= bbox_tensor[:, 3]

                            # left -> xc , top-> yc
                            bbox_tensor[:, 2] += bbox_tensor[:, 4] / 2
                            bbox_tensor[:, 3] += bbox_tensor[:, 5] / 2

                            # add 0 angle
                            bbox_tensor = np.concatenate(
                                [
                                    bbox_tensor,
                                    np.zeros(
                                        (bbox_tensor.shape[0], 1), dtype=np.float32
                                    ),
                                ],
                                axis=1,
                            )
                        else:
                            selection_type = ObjectSelectionType.ROTATED_BBOX

                        # add index column to further filter attribute values
                        bbox_tensor = np.concatenate(
                            [
                                bbox_tensor,
                                np.arange(
                                    bbox_tensor.shape[0], dtype=np.float32
                                ).reshape(-1, 1),
                            ],
                            axis=1,
                        )

                        selected_bboxes = []
                        for obj in self._model.output.objects:
                            cls_bbox_tensor = bbox_tensor[
                                bbox_tensor[:, 0] == obj.class_id
                            ]
                            if cls_bbox_tensor.shape[0] == 0:
                                continue
                            if obj.selector:
                                try:
                                    cls_bbox_tensor = obj.selector(cls_bbox_tensor)
                                except Exception as exc:  # pylint: disable=broad-except
                                    if obj.selector.dev_mode:
                                        if not isinstance(exc, PyFuncNoopCallException):
                                            self._logger.exception(
                                                'Error calling selector.'
                                            )
                                        cls_bbox_tensor = np.zeros((0, 8))
                                    else:
                                        raise exc

                            obj_label = build_model_object_key(
                                self._element_name, obj.label
                            )
                            obj_cls_id = MERGED_CLASSES[self._element_name].get(
                                obj.class_id
                            )
                            if obj_cls_id is None:
                                obj_cls_id = obj.class_id
                            else:
                                if self._logger.isEnabledFor(logging.TRACE):
                                    self._logger.trace(
                                        'Updating %s custom objs id %s -> %s, '
                                        'label "%s".',
                                        len(cls_bbox_tensor),
                                        obj.class_id,
                                        obj_cls_id,
                                        obj_label,
                                    )
                            for bbox in cls_bbox_tensor:
                                if self._logger.isEnabledFor(logging.TRACE):
                                    self._logger.trace(
                                        'Adding obj %s into pyds meta for frame '
                                        'with PTS %s.',
                                        bbox[2:7],
                                        nvds_frame_meta.buf_pts,
                                    )
                                _nvds_obj_meta = nvds_add_obj_meta_to_frame(
                                    nvds_batch_meta,
                                    nvds_frame_meta,
                                    selection_type,
                                    obj_cls_id,
                                    self._model_uid,
                                    bbox[2:7],
                                    bbox[1],
                                    obj_label,
                                    parent=parent_nvds_obj_meta,
                                )
                                selected_bboxes.append((int(bbox[7]), _nvds_obj_meta))

                    # attribute or complex model
                    if values:
                        if self._is_complex_model:
                            values = [values[i] for i, _ in selected_bboxes]
                        else:
                            selected_bboxes = [(0, nvds_obj_meta)]
                            values = [values]
                        for (_, _nvds_obj_meta), _values in zip(
                            selected_bboxes, values
                        ):
                            for attr_name, value, confidence in _values:
                                nvds_add_attr_meta_to_obj(
                                    frame_meta=nvds_frame_meta,
                                    obj_meta=_nvds_obj_meta,
                                    element_name=self._element_name,
                                    name=attr_name,
                                    value=value,
                                    confidence=confidence,
                                )
        self._restore_frame(buffer)

    def _process_regular_detector_output(self, buffer: Gst.Buffer):
        """Processes output of nvinfer detector.
        Corrects nvds_obj_meta.obj_label, sets object uid and selection type.
        """
        self._logger.debug(
            'Preparing "%s" element regular object model output, buffer PTS %s.',
            self._element_name,
            buffer.pts,
        )
        self._model: NvInferDetector
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                self._restore_object_meta(nvds_obj_meta)
                if nvds_obj_meta.unique_component_id != self._model_uid:
                    continue

                for obj in self._model.output.objects:
                    if nvds_obj_meta.class_id == obj.class_id:
                        obj_cls_id = MERGED_CLASSES[self._element_name].get(
                            obj.class_id
                        )
                        if obj_cls_id is not None:
                            if self._logger.isEnabledFor(logging.TRACE):
                                self._logger.trace(
                                    'Updating regular obj id %s -> %s, label "%s".',
                                    obj.class_id,
                                    obj_cls_id,
                                    obj.label,
                                )
                            nvds_obj_meta.class_id = obj_cls_id
                        nvds_set_obj_selection_type(
                            obj_meta=nvds_obj_meta,
                            selection_type=ObjectSelectionType.REGULAR_BBOX,
                        )
                        try:
                            nvds_set_obj_uid(
                                frame_meta=nvds_frame_meta,
                                obj_meta=nvds_obj_meta,
                            )
                        except UIDError:
                            pass
                        nvds_obj_meta.obj_label = build_model_object_key(
                            self._element_name, obj.label
                        )
        self._restore_frame(buffer)

    def _process_regular_classifier_output(self, buffer: Gst.Buffer):
        """Processes output of nvinfer classifier."""
        self._logger.debug(
            'Preparing "%s" element regular classifier output, buffer PTS %s.',
            self._element_name,
            buffer.pts,
        )
        self._model: NvInferAttributeModel
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                self._restore_object_meta(nvds_obj_meta)
                for nvds_clf_meta in nvds_clf_meta_iterator(nvds_obj_meta):
                    if nvds_clf_meta.unique_component_id != self._model_uid:
                        continue

                    for attr, label_info in zip(
                        self._model.output.attributes,
                        nvds_label_info_iterator(nvds_clf_meta),
                    ):
                        nvds_add_attr_meta_to_obj(
                            frame_meta=nvds_frame_meta,
                            obj_meta=nvds_obj_meta,
                            element_name=self._element_name,
                            name=attr.name,
                            value=label_info.result_label,
                            confidence=label_info.result_prob,
                        )
        self._restore_frame(buffer)

    def _is_model_input_object(self, nvds_obj_meta: pyds.NvDsObjectMeta):
        return (
            nvds_obj_meta.unique_component_id == self._input_object_model_uid
            and nvds_obj_meta.class_id == self._input_object_class_id
        )

    def _get_frame_source_id_and_idx(
        self,
        buffer: Gst.Buffer,
        nvds_frame_meta: pyds.NvDsFrameMeta,
    ) -> Tuple[Optional[str], Optional[int]]:
        savant_batch_meta = gst_buffer_get_savant_batch_meta(buffer)
        if savant_batch_meta is None:
            return None, None

        savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(nvds_frame_meta)
        if savant_frame_meta is None:
            return None, None

        frame_idx = savant_frame_meta.idx
        video_frame, _ = self._video_pipeline.get_batched_frame(
            savant_batch_meta.idx,
            frame_idx,
        )
        source_id = video_frame.source_id

        return source_id, frame_idx

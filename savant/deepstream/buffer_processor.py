"""Buffer processor for DeepStream pipeline."""
import logging
from collections import deque
from heapq import heappop, heappush
from queue import Queue
from typing import Deque, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pyds
from pygstsavantframemeta import (
    gst_buffer_get_savant_batch_meta,
    gst_buffer_get_savant_frame_meta,
    nvds_frame_meta_get_nvds_savant_frame_meta,
)
from savant_rs.pipeline2 import VideoPipeline
from savant_rs.primitives import VideoFrame, VideoFrameTransformation
from savant_rs.primitives.geometry import BBox, RBBox
from savant_rs.utils.symbol_mapper import (
    build_model_object_key,
    get_model_id,
    get_object_id,
    parse_compound_key,
)
from savant_rs.video_object_query import MatchQuery

from savant.api.parser import parse_attribute_value
from savant.base.input_preproc import ObjectsPreprocessing
from savant.base.model import ComplexModel, ObjectModel
from savant.base.pyfunc import PyFuncNoopCallException
from savant.config.schema import FrameParameters, ModelElement, PipelineElement
from savant.deepstream.meta.object import _NvDsObjectMetaImpl
from savant.deepstream.nvinfer.model import NvInferAttributeModel, NvInferDetector
from savant.deepstream.source_output import (
    SourceOutput,
    SourceOutputEncoded,
    SourceOutputH26X,
    SourceOutputWithFrame,
)
from savant.deepstream.utils import (
    get_nvds_buf_surface,
    nvds_add_attr_meta_to_obj,
    nvds_add_obj_meta_to_frame,
    nvds_clf_meta_iterator,
    nvds_frame_meta_iterator,
    nvds_infer_tensor_meta_to_outputs,
    nvds_label_info_iterator,
    nvds_obj_meta_iterator,
    nvds_set_obj_selection_type,
    nvds_set_obj_uid,
    nvds_tensor_output_iterator,
)
from savant.deepstream.utils.attribute import nvds_get_all_obj_attrs
from savant.gstreamer import Gst  # noqa:F401
from savant.gstreamer.buffer_processor import GstBufferProcessor
from savant.gstreamer.codecs import Codec, CodecInfo
from savant.meta.constants import PRIMARY_OBJECT_KEY, UNTRACKED_OBJECT_ID
from savant.meta.errors import UIDError
from savant.meta.object import ObjectMeta
from savant.meta.type import ObjectSelectionType
from savant.utils.fps_meter import FPSMeter
from savant.utils.logging import LoggerMixin
from savant.utils.platform import is_aarch64
from savant.utils.sink_factories import SinkVideoFrame
from savant.utils.source_info import SourceInfo, SourceInfoRegistry


class _OutputFrame(NamedTuple):
    """Output frame with its metadata."""

    idx: int
    pts: int
    dts: Optional[int]
    frame: Optional[bytes]
    codec: Optional[CodecInfo]
    keyframe: bool


class NvDsBufferProcessor(GstBufferProcessor, LoggerMixin):
    def __init__(
        self,
        queue: Queue,
        fps_meter: FPSMeter,
        sources: SourceInfoRegistry,
        objects_preprocessing: ObjectsPreprocessing,
        frame_params: FrameParameters,
        video_pipeline: VideoPipeline,
    ):
        """Buffer processor for DeepStream pipeline.

        :param queue: Queue for output data.
        :param fps_meter: FPS meter.
        :param sources: Source info registry.
        :param objects_preprocessing: Objects processing registry.
        :param frame_params: Processing frame parameters (after nvstreammux).
        """

        super().__init__(queue, fps_meter)
        self._sources = sources
        self._objects_preprocessing = objects_preprocessing
        self._frame_params = frame_params
        self._queue = queue
        self._video_pipeline = video_pipeline

        self._scale_transformation = VideoFrameTransformation.scale(
            frame_params.width,
            frame_params.height,
        )
        if frame_params.padding and frame_params.padding.keep:
            self._padding_transformation = VideoFrameTransformation.padding(
                frame_params.padding.left,
                frame_params.padding.top,
                frame_params.padding.right,
                frame_params.padding.bottom,
            )
        else:
            self._padding_transformation = None

    def prepare_input(self, buffer: Gst.Buffer):
        """Input meta processor.

        :param buffer: gstreamer buffer that is being processed.
        """

        self.logger.debug('Preparing input for buffer with PTS %s.', buffer.pts)
        savant_batch_meta = gst_buffer_get_savant_batch_meta(buffer)
        if savant_batch_meta is None:
            # TODO: add VideoFrame to VideoPipeline?
            self.logger.warning(
                'Failed to prepare input for batch at buffer %s. '
                'Batch has no Savant Frame Meta.',
                buffer.pts,
            )
            return

        batch_id = savant_batch_meta.idx
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(
                nvds_frame_meta
            )
            if savant_frame_meta is None:
                # TODO: add VideoFrame to VideoPipeline?
                self.logger.warning(
                    'Failed to prepare input for frame %s at buffer %s. '
                    'Frame has no Savant Frame Meta.',
                    nvds_frame_meta.buf_pts,
                    buffer.pts,
                )
                continue

            frame_idx = savant_frame_meta.idx
            video_frame, video_frame_span = self._video_pipeline.get_batched_frame(
                batch_id,
                frame_idx,
            )
            with video_frame_span.nested_span('prepare-input'):
                self._prepare_input_frame(
                    frame_idx=frame_idx,
                    nvds_batch_meta=nvds_batch_meta,
                    nvds_frame_meta=nvds_frame_meta,
                    video_frame=video_frame,
                )

    def _prepare_input_frame(
        self,
        frame_idx: int,
        nvds_batch_meta: pyds.NvDsBatchMeta,
        nvds_frame_meta: pyds.NvDsFrameMeta,
        video_frame: VideoFrame,
    ):
        frame_pts = nvds_frame_meta.buf_pts
        self.logger.debug(
            'Preparing input for frame of source %s with IDX %s and PTS %s.',
            video_frame.source_id,
            frame_idx,
            frame_pts,
        )

        # full frame primary object by default
        source_info = self._sources.get_source(video_frame.source_id)
        self._add_transformations(
            frame_idx=frame_idx,
            source_info=source_info,
            video_frame=video_frame,
        )
        scale_factor_x = self._frame_params.width / source_info.src_resolution.width
        scale_factor_y = self._frame_params.height / source_info.src_resolution.height
        primary_bbox = BBox(
            self._frame_params.width / 2,
            self._frame_params.height / 2,
            self._frame_params.width,
            self._frame_params.height,
        )
        self.logger.debug(
            'Init primary bbox for frame with PTS %s: %s', frame_pts, primary_bbox
        )

        all_nvds_obj_metas = {}
        # add external objects to nvds meta
        for obj_meta in video_frame.access_objects(MatchQuery.idle()):
            obj_key = build_model_object_key(obj_meta.namespace, obj_meta.label)
            bbox = obj_meta.detection_box
            if isinstance(bbox, RBBox) and not bbox.angle:
                bbox = BBox(*bbox.as_xcycwh())
            # skip primary object for now, will be added later
            if obj_key == PRIMARY_OBJECT_KEY:
                # if not a full frame then correct primary object
                if not bbox.almost_eq(primary_bbox, 1e-6):
                    primary_bbox = bbox
                    primary_bbox.scale(scale_factor_x, scale_factor_y)
                    self.logger.debug(
                        'Corrected primary bbox for frame with PTS %s: %s',
                        frame_pts,
                        primary_bbox,
                    )
                continue
            # obj_key was only registered if
            # it was required by the pipeline model elements (this case)
            # or equaled the output object of one of the pipeline model elements
            model_name, label = parse_compound_key(obj_key)
            model_uid, class_id = get_object_id(model_name, label)
            if isinstance(bbox, RBBox):
                selection_type = ObjectSelectionType.ROTATED_BBOX
            else:
                selection_type = ObjectSelectionType.REGULAR_BBOX

            bbox.scale(scale_factor_x, scale_factor_y)
            if self._frame_params.padding:
                bbox.left += self._frame_params.padding.left
                bbox.top += self._frame_params.padding.top

            track_id = obj_meta.get_track_id()
            if track_id is None:
                track_id = UNTRACKED_OBJECT_ID
            # create nvds obj meta
            nvds_obj_meta = nvds_add_obj_meta_to_frame(
                nvds_batch_meta,
                nvds_frame_meta,
                selection_type,
                class_id,
                model_uid,
                bbox,
                obj_meta.confidence,
                obj_key,
                track_id,
            )

            # save nvds obj meta ref in case it is some other obj's parent
            # and save nvds obj meta ref in case it has a parent
            # this is done to avoid one more full iteration of frame's objects
            # because the parent meta may be created after the child
            all_nvds_obj_metas[obj_meta.id] = nvds_obj_meta

            for namespace, name in obj_meta.attributes:
                attr = obj_meta.get_attribute(namespace, name)
                value = attr.values[0]
                nvds_add_attr_meta_to_obj(
                    frame_meta=nvds_frame_meta,
                    obj_meta=nvds_obj_meta,
                    element_name=namespace,
                    name=name,
                    value=parse_attribute_value(value),
                    confidence=value.confidence,
                )

        # finish configuring obj metas by assigning the parents
        # TODO: fix query to iterate only objects with children
        for parent in video_frame.access_objects(MatchQuery.idle()):
            for child in video_frame.get_children(parent.id):
                all_nvds_obj_metas[child.id].parent = all_nvds_obj_metas[parent.id]

        video_frame.clear_objects()
        # add primary frame object
        model_name, label = parse_compound_key(PRIMARY_OBJECT_KEY)
        model_uid, class_id = get_object_id(model_name, label)
        if self._frame_params.padding:
            primary_bbox.xc += self._frame_params.padding.left
            primary_bbox.yc += self._frame_params.padding.top
        self.logger.debug(
            'Add primary object to frame meta with PTS %s, bbox: %s',
            frame_pts,
            primary_bbox,
        )
        nvds_add_obj_meta_to_frame(
            nvds_batch_meta,
            nvds_frame_meta,
            ObjectSelectionType.REGULAR_BBOX,
            class_id,
            model_uid,
            primary_bbox,
            # confidence should be bigger than tracker minDetectorConfidence
            # to prevent the tracker from deleting the object
            # use tracker display-tracking-id=0 to avoid labelling
            0.999,
            PRIMARY_OBJECT_KEY,
        )

        nvds_frame_meta.bInferDone = True  # required for tracker (DS 6.0)

    def _add_transformations(
        self,
        frame_idx: Optional[int],
        source_info: SourceInfo,
        video_frame: VideoFrame,
    ):
        self.logger.debug(
            'Adding transformations for frame of source %s with IDX %s and PTS %s.',
            video_frame.source_id,
            frame_idx,
            video_frame.pts,
        )

        if source_info.add_scale_transformation:
            self.logger.debug(
                'Adding scale transformation for frame of source %s with IDX %s and PTS %s.',
                video_frame.source_id,
                frame_idx,
                video_frame.pts,
            )
            video_frame.add_transformation(self._scale_transformation)

        if self._padding_transformation is not None:
            self.logger.debug(
                'Adding padding transformation for frame of source %s with IDX %s and PTS %s.',
                video_frame.source_id,
                frame_idx,
                video_frame.pts,
            )
            video_frame.add_transformation(self._padding_transformation)

    def prepare_output(
        self,
        buffer: Gst.Buffer,
        source_info: SourceInfo,
    ) -> SinkVideoFrame:
        """Enqueue output messages based on frame meta.

        :param buffer: gstreamer buffer that is being processed.
        :param source_info: output source info
        """

        self.logger.debug(
            'Preparing output for buffer with PTS %s and DTS %s for source %s.',
            buffer.pts,
            buffer.dts,
            source_info.source_id,
        )
        for output_frame in self._iterate_output_frames(buffer, source_info):
            yield self._build_sink_video_frame(output_frame, source_info)

    def _build_sink_video_frame(
        self,
        output_frame: _OutputFrame,
        source_info: SourceInfo,
    ) -> SinkVideoFrame:
        self.logger.debug(
            'Preparing output for frame of source %s with IDX %s, PTS %s and DTS %s.',
            source_info.source_id,
            output_frame.idx,
            output_frame.pts,
            output_frame.dts,
        )

        video_frame, video_frame_span = self._video_pipeline.get_independent_frame(
            output_frame.idx,
        )
        with video_frame_span.nested_span('prepare_output'):
            video_frame.dts = output_frame.dts
            video_frame.width = self._frame_params.output_width
            video_frame.height = self._frame_params.output_height
            if output_frame.codec is not None:
                video_frame.codec = output_frame.codec.name
            video_frame.keyframe = output_frame.keyframe
            # TODO: Do we need to delete frame after it was sent to sink?
            spans = self._video_pipeline.delete(output_frame.idx)
            span_context = spans[output_frame.idx].propagate()

        return SinkVideoFrame(
            video_frame=video_frame,
            frame=output_frame.frame,
            span_context=span_context,
        )

    def on_eos(self, source_info: SourceInfo) -> SinkVideoFrame:
        """Pipeline EOS handler."""

    def prepare_element_input(self, element: PipelineElement, buffer: Gst.Buffer):
        """Model input preprocessing.

        :param element: element that this probe was added to.
        :param buffer: gstreamer buffer that is being processed.
        """
        if not isinstance(element, ModelElement):
            return

        model = element.model
        if (
            not model.input.preprocess_object_meta
            and not model.input.preprocess_object_image
        ):
            return
        self.logger.debug(
            'Preparing "%s" element input for buffer with PTS %s.',
            element.name,
            buffer.pts,
        )
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if model.input.preprocess_object_meta:
            for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
                self.logger.debug(
                    'Preprocessing "%s" element object meta for frame with PTS %s.',
                    element.name,
                    nvds_frame_meta.buf_pts,
                )
                for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                    if not self._is_model_input_object(element, nvds_obj_meta):
                        continue
                    # TODO: Unify and also switch to the box representation system
                    #  through the center point during meta preprocessing.

                    object_meta = _NvDsObjectMetaImpl.from_nv_ds_object_meta(
                        nvds_obj_meta, nvds_frame_meta
                    )
                    parent_object_meta = object_meta.parent

                    if isinstance(object_meta.bbox, BBox):
                        bbox = object_meta.bbox.copy()
                        if isinstance(parent_object_meta.bbox, BBox):
                            parent_bbox = parent_object_meta.bbox.copy()
                        else:
                            raise NotImplementedError(
                                'You try apply preprocessing to object that have '
                                'rotated bbox in parent object. '
                                'Only BBox is supported now'
                            )
                    else:
                        raise NotImplementedError(
                            'You try apply preprocessing to rotated bbox. '
                            'Only BBox is supported now'
                        )
                    if parent_object_meta.parent is not None:
                        self.logger.warning(
                            'Preprocessing is supported only 1 level of hierarchy.'
                        )

                    user_parent_object = None
                    if object_meta.parent is not None:
                        user_parent_object = ObjectMeta(
                            parent_object_meta.element_name,
                            parent_object_meta.label,
                            parent_bbox,
                            parent_object_meta.confidence,
                            parent_object_meta.track_id,
                            attributes=nvds_get_all_obj_attrs(
                                nvds_frame_meta,
                                parent_object_meta.ds_object_meta,
                            ),
                        )

                    user_object_meta = ObjectMeta(
                        object_meta.element_name,
                        object_meta.label,
                        bbox,
                        object_meta.confidence,
                        object_meta.track_id,
                        user_parent_object,
                        attributes=nvds_get_all_obj_attrs(
                            frame_meta=nvds_frame_meta,
                            obj_meta=parent_object_meta.ds_object_meta,
                        ),
                    )

                    try:
                        res_bbox = model.input.preprocess_object_meta(
                            object_meta=user_object_meta
                        )
                    except Exception as exc:
                        if model.input.preprocess_object_meta.dev_mode:
                            if not isinstance(exc, PyFuncNoopCallException):
                                self.logger.exception(
                                    'Error calling preprocess input object meta.'
                                )
                            res_bbox = user_object_meta.bbox
                        else:
                            raise exc

                    if self.logger.isEnabledFor(logging.TRACE):
                        self.logger.trace(
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

        elif model.input.preprocess_object_image:
            self.logger.debug('Preprocessing "%s" element object image.', element.name)
            model_name, label = parse_compound_key(model.input.object)
            model_uid, class_id = get_object_id(model_name, label)
            self._objects_preprocessing.preprocessing(
                element.name,
                hash(buffer),
                model_uid,
                class_id,
                model.input.preprocess_object_image.output_image,
                model.input.preprocess_object_image.dev_mode,
            )

    def prepare_element_output(self, element: PipelineElement, buffer: Gst.Buffer):
        """Model output postprocessing.

        :param element: element that this probe was added to.
        :param buffer: gstreamer buffer that is being processed.
        """
        if not isinstance(element, ModelElement):
            return
        self.logger.debug(
            'Preparing "%s" element output for buffer with PTS %s.',
            element.name,
            buffer.pts,
        )
        frame_left = 0.0
        frame_top = 0.0
        frame_right = self._frame_params.width - 1.0
        frame_bottom = self._frame_params.height - 1.0
        if self._frame_params.padding:
            frame_left += self._frame_params.padding.left
            frame_top += self._frame_params.padding.top
            frame_right += self._frame_params.padding.left
            frame_bottom += self._frame_params.padding.top

        model_uid = get_model_id(element.name)
        model: Union[
            NvInferDetector,
            NvInferAttributeModel,
        ] = element.model
        is_complex_model = isinstance(model, ComplexModel)
        is_object_model = isinstance(model, ObjectModel)

        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                # convert custom model output and save meta
                if model.output.converter:
                    if not self._is_model_input_object(element, nvds_obj_meta):
                        continue
                    parent_nvds_obj_meta = nvds_obj_meta
                    for tensor_meta in nvds_tensor_output_iterator(
                        parent_nvds_obj_meta, gie_uid=model_uid
                    ):
                        if self.logger.isEnabledFor(logging.TRACE):
                            self.logger.trace(
                                'Converting "%s" element tensor output for frame with PTS %s.',
                                element.name,
                                nvds_frame_meta.buf_pts,
                            )
                        # parse and post-process model output
                        output_layers = nvds_infer_tensor_meta_to_outputs(
                            tensor_meta=tensor_meta,
                            layer_names=model.output.layer_names,
                        )
                        try:
                            outputs = model.output.converter(
                                *output_layers,
                                model=model,
                                roi=(
                                    parent_nvds_obj_meta.rect_params.left,
                                    parent_nvds_obj_meta.rect_params.top,
                                    parent_nvds_obj_meta.rect_params.width,
                                    parent_nvds_obj_meta.rect_params.height,
                                ),
                            )
                        except Exception as exc:
                            if model.output.converter.dev_mode:
                                if not isinstance(exc, PyFuncNoopCallException):
                                    self.logger.exception('Error calling converter')
                                # provide some placeholders so that the pipeline processing can continue
                                if is_complex_model:
                                    outputs = np.zeros((0, 6)), np.zeros((0, 1))
                                elif is_object_model:
                                    outputs = np.zeros((0, 6))
                                else:
                                    outputs = np.zeros((0, 1))
                            else:
                                raise exc
                        # for object/complex models output - `bbox_tensor` and
                        # `selected_bboxes` - indices of selected bboxes and meta
                        # for attribute/complex models output - `values`
                        bbox_tensor, selected_bboxes, values = None, None, None
                        # complex model
                        if is_complex_model:
                            # output converter returns tensor and attribute values
                            bbox_tensor, values = outputs
                            assert bbox_tensor.shape[0] == len(
                                values
                            ), 'Number of detected boxes and attributes do not match.'

                        # object model
                        elif is_object_model:
                            # output converter returns tensor with
                            # (class_id, confidence, xc, yc, width, height, [angle]),
                            # coordinates in roi scale (parent object scale)
                            bbox_tensor = outputs

                        # attribute model
                        else:
                            # output converter returns attribute values
                            values = outputs

                        if bbox_tensor is not None and bbox_tensor.shape[0] > 0:
                            # object or complex model with non empty output
                            if bbox_tensor.shape[1] == 6:  # no angle
                                selection_type = ObjectSelectionType.REGULAR_BBOX
                                # xc -> left, yc -> top
                                bbox_tensor[:, 2] -= bbox_tensor[:, 4] / 2
                                bbox_tensor[:, 3] -= bbox_tensor[:, 5] / 2

                                # clip
                                # width to right, height to bottom
                                bbox_tensor[:, 4] += bbox_tensor[:, 2]
                                bbox_tensor[:, 5] += bbox_tensor[:, 3]
                                # clip
                                bbox_tensor[:, 2][
                                    bbox_tensor[:, 2] < frame_left
                                ] = frame_left
                                bbox_tensor[:, 3][
                                    bbox_tensor[:, 3] < frame_top
                                ] = frame_top
                                bbox_tensor[:, 4][
                                    bbox_tensor[:, 4] > frame_right
                                ] = frame_right
                                bbox_tensor[:, 5][
                                    bbox_tensor[:, 5] > frame_bottom
                                ] = frame_bottom

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
                            for obj in model.output.objects:
                                cls_bbox_tensor = bbox_tensor[
                                    bbox_tensor[:, 0] == obj.class_id
                                ]
                                if cls_bbox_tensor.shape[0] == 0:
                                    continue
                                if obj.selector:
                                    try:
                                        cls_bbox_tensor = obj.selector(cls_bbox_tensor)
                                    except Exception as exc:
                                        if obj.selector.dev_mode:
                                            if not isinstance(
                                                exc, PyFuncNoopCallException
                                            ):
                                                self.logger.exception(
                                                    'Error calling selector.'
                                                )
                                            cls_bbox_tensor = np.zeros((0, 8))
                                        else:
                                            raise exc

                                obj_label = build_model_object_key(
                                    element.name, obj.label
                                )
                                for bbox in cls_bbox_tensor:
                                    # add NvDsObjectMeta
                                    if self.logger.isEnabledFor(logging.TRACE):
                                        self.logger.trace(
                                            'Adding obj %s into pyds meta for frame with PTS %s.',
                                            bbox[2:7],
                                            nvds_frame_meta.buf_pts,
                                        )
                                    _nvds_obj_meta = nvds_add_obj_meta_to_frame(
                                        nvds_batch_meta,
                                        nvds_frame_meta,
                                        selection_type,
                                        obj.class_id,
                                        model_uid,
                                        bbox[2:7],
                                        bbox[1],
                                        obj_label,
                                        parent=parent_nvds_obj_meta,
                                    )
                                    selected_bboxes.append(
                                        (int(bbox[7]), _nvds_obj_meta)
                                    )

                        if values:
                            # attribute or complex model
                            if is_complex_model:
                                values = [
                                    v
                                    for i, v in enumerate(values)
                                    if i in {i for i, o in selected_bboxes}
                                ]
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
                                        element_name=element.name,
                                        name=attr_name,
                                        value=value,
                                        confidence=confidence,
                                    )

                # regular object model (detector)
                # correct nvds_obj_meta.obj_label
                elif is_object_model:
                    if nvds_obj_meta.unique_component_id == model_uid:
                        for obj in model.output.objects:
                            if nvds_obj_meta.class_id == obj.class_id:
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
                                    element.name, obj.label
                                )
                                break

                # regular attribute model (classifier)
                # convert nvds_clf_meta to attr_meta
                else:
                    for nvds_clf_meta in nvds_clf_meta_iterator(nvds_obj_meta):
                        if nvds_clf_meta.unique_component_id != model_uid:
                            continue
                        for attr, label_info in zip(
                            model.output.attributes,
                            nvds_label_info_iterator(nvds_clf_meta),
                        ):
                            nvds_add_attr_meta_to_obj(
                                frame_meta=nvds_frame_meta,
                                obj_meta=nvds_obj_meta,
                                element_name=element.name,
                                name=attr.name,
                                value=label_info.result_label,
                                confidence=label_info.result_prob,
                            )

                # restore nvds_obj_meta.rect_params if there was preprocessing
                if (
                    model.input.preprocess_object_meta
                    or model.input.preprocess_object_image
                ) and self._is_model_input_object(element, nvds_obj_meta):
                    bbox_coords = nvds_obj_meta.detector_bbox_info.org_bbox_coords
                    if nvds_obj_meta.tracker_bbox_info.org_bbox_coords.width > 0:
                        bbox_coords = nvds_obj_meta.tracker_bbox_info.org_bbox_coords
                    rect_params = nvds_obj_meta.rect_params
                    rect_params.left = bbox_coords.left
                    rect_params.top = bbox_coords.top
                    rect_params.width = bbox_coords.width
                    rect_params.height = bbox_coords.height

        # restore frame
        if model.input.preprocess_object_image:
            self._objects_preprocessing.restore_frame(hash(buffer))

    def _is_model_input_object(
        self, element: ModelElement, nvds_obj_meta: pyds.NvDsObjectMeta
    ):
        model_name, label = parse_compound_key(element.model.input.object)
        model_uid, class_id = get_object_id(model_name, label)
        return (
            nvds_obj_meta.unique_component_id == model_uid
            and nvds_obj_meta.class_id == class_id
        )

    def _iterate_output_frames(
        self,
        buffer: Gst.Buffer,
        source_info: SourceInfo,
    ) -> Iterator[_OutputFrame]:
        """Iterate output frames."""


class NvDsEncodedBufferProcessor(NvDsBufferProcessor):
    def __init__(
        self,
        queue: Queue,
        fps_meter: FPSMeter,
        sources: SourceInfoRegistry,
        objects_preprocessing: ObjectsPreprocessing,
        frame_params: FrameParameters,
        codec: CodecInfo,
        video_pipeline: VideoPipeline,
    ):
        """Buffer processor for DeepStream pipeline.

        :param queue: Queue for output data.
        :param fps_meter: FPS meter.
        :param sources: Source info registry.
        :param objects_preprocessing: Objects processing registry.
        :param frame_params: Processing frame parameters (after nvstreammux).
        :param codec: Codec of the output frames.
        """

        self._codec = codec
        super().__init__(
            queue=queue,
            fps_meter=fps_meter,
            sources=sources,
            objects_preprocessing=objects_preprocessing,
            frame_params=frame_params,
            video_pipeline=video_pipeline,
        )

    def _iterate_output_frames(
        self,
        buffer: Gst.Buffer,
        source_info: SourceInfo,
    ) -> Iterator[_OutputFrame]:
        """Iterate output frames from Gst.Buffer."""

        yield self._build_output_frame(
            idx=extract_frame_idx(buffer),
            pts=buffer.pts,
            dts=buffer.dts if buffer.dts != Gst.CLOCK_TIME_NONE else None,
            buffer=buffer,
        )

    def _build_output_frame(
        self,
        idx: Optional[int],
        pts: int,
        dts: Optional[int],
        buffer: Gst.Buffer,
    ) -> _OutputFrame:
        if buffer.get_size() > 0:
            # get encoded frame for output
            frame = buffer.extract_dup(0, buffer.get_size())
        else:
            frame = None
            dts = None
        is_keyframe = not buffer.has_flags(Gst.BufferFlags.DELTA_UNIT)
        return _OutputFrame(
            idx=idx,
            pts=pts,
            dts=dts,
            frame=frame,
            codec=self._codec,
            keyframe=is_keyframe,
        )


class NvDsJetsonH26XBufferProcessor(NvDsEncodedBufferProcessor):
    def __init__(self, *args, **kwargs):
        """Buffer processor for DeepStream pipeline.

        Workaround for a bug in h264x encoders on Jetson devices.
        https://forums.developer.nvidia.com/t/nvv4l2h264enc-returns-frames-in-wrong-order-when-pts-doesnt-align-with-framerate/257363

        Encoder "nvv4l2h26xenc" on Jetson devices produces frames with correct
        DTS but with PTS and metadata from different frames. We store buffers
        in a queue and wait for a buffer with correct PTS.
        """

        # source_id -> (DTS, buffer)
        self._dts_queues: Dict[str, Deque[Tuple[int, Gst.Buffer]]] = {}
        # source_id -> (PTS, IDX)
        self._pts_queues: Dict[str, List[Tuple[int, Optional[int]]]] = {}
        super().__init__(*args, **kwargs)

    def _iterate_output_frames(
        self,
        buffer: Gst.Buffer,
        source_info: SourceInfo,
    ) -> Iterator[_OutputFrame]:
        """Iterate output frames from Gst.Buffer."""

        frame_idx = extract_frame_idx(buffer)
        dts_queue = self._dts_queues.setdefault(source_info.source_id, deque())
        pts_queue = self._pts_queues.setdefault(source_info.source_id, [])
        # When the frame is empty assign DTS=PTS
        dts = buffer.dts if buffer.get_size() > 0 else buffer.pts
        if dts_queue or buffer.pts != dts:
            self.logger.debug(
                'Storing frame with PTS %s, DTS %s and IDX %s to queue.',
                buffer.pts,
                dts,
                frame_idx,
            )
            dts_queue.append((dts, buffer))
            heappush(pts_queue, (buffer.pts, frame_idx))
            while dts_queue and dts_queue[0][0] == pts_queue[0][0]:
                next_dts, next_buf = dts_queue.popleft()
                next_pts, next_idx = heappop(pts_queue)
                self.logger.debug(
                    'Pushing output frame frame with PTS %s, DTS %s and IDX %s.',
                    next_pts,
                    next_dts,
                    next_idx,
                )
                yield self._build_output_frame(
                    idx=next_idx,
                    pts=next_pts,
                    dts=next_dts,
                    buffer=next_buf,
                )
        else:
            self.logger.debug(
                'PTS and DTS of the frame are the same: %s. Pushing output frame.', dts
            )
            yield self._build_output_frame(
                idx=frame_idx,
                pts=buffer.pts,
                dts=dts,
                buffer=buffer,
            )

    def on_eos(self, source_info: SourceInfo):
        """Pipeline EOS handler."""
        dts_queue = self._dts_queues.pop(source_info.source_id, None)
        pts_queue = self._pts_queues.pop(source_info.source_id, None)
        if dts_queue is None:
            return
        while dts_queue:
            dts, buffer = dts_queue.popleft()
            pts, idx = dts, None

            if pts_queue:
                while pts_queue and pts_queue[0][0] <= dts:
                    pts, idx = heappop(pts_queue)
            if pts != dts:
                pts = dts
                idx = None

            output_frame = self._build_output_frame(
                idx=idx,
                pts=pts,
                dts=dts,
                buffer=buffer,
            )
            sink_message = self._build_sink_video_frame(output_frame, source_info)
            self._queue.put(sink_message)
            # measure and logging FPS
            if self._fps_meter():
                self.logger.info(self._fps_meter.message)


class NvDsRawBufferProcessor(NvDsBufferProcessor):
    def __init__(
        self,
        queue: Queue,
        fps_meter: FPSMeter,
        sources: SourceInfoRegistry,
        objects_preprocessing: ObjectsPreprocessing,
        frame_params: FrameParameters,
        output_frame: bool,
        video_pipeline: VideoPipeline,
    ):
        """Buffer processor for DeepStream pipeline.

        :param queue: Queue for output data.
        :param fps_meter: FPS meter.
        :param sources: Source info registry.
        :param objects_preprocessing: Objects processing registry.
        :param frame_params: Processing frame parameters (after nvstreammux).
        :param output_frame: Whether to output frame or not.
        """

        self._output_frame = output_frame
        self._codec = Codec.RAW_RGBA.value if output_frame else None
        super().__init__(
            queue=queue,
            fps_meter=fps_meter,
            sources=sources,
            objects_preprocessing=objects_preprocessing,
            frame_params=frame_params,
            video_pipeline=video_pipeline,
        )

    def _iterate_output_frames(
        self,
        buffer: Gst.Buffer,
        source_info: SourceInfo,
    ) -> Iterator[_OutputFrame]:
        """Iterate output frames from NvDs batch.

        NvDs batch contains raw RGBA frames. They are all keyframes.
        """

        if buffer.get_size() == 0:
            frame_idx = extract_frame_idx(buffer)
            yield _OutputFrame(
                idx=frame_idx,
                pts=buffer.pts,
                dts=None,  # Raw frames do not have dts
                frame=None,
                codec=self._codec,
                # Any frame is keyframe since it was not encoded
                keyframe=True,
            )
            return

        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            # get frame if required for output
            if self._output_frame:
                with get_nvds_buf_surface(buffer, nvds_frame_meta) as np_frame:
                    frame = np_frame.tobytes()
            else:
                frame = None
            savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(
                nvds_frame_meta
            )
            frame_idx = savant_frame_meta.idx if savant_frame_meta else None
            frame_pts = nvds_frame_meta.buf_pts
            yield _OutputFrame(
                idx=frame_idx,
                pts=frame_pts,
                dts=None,  # Raw frames do not have dts
                frame=frame,
                codec=self._codec,
                # Any frame is keyframe since it was not encoded
                keyframe=True,
            )


def extract_frame_idx(buffer: Gst.Buffer) -> Optional[int]:
    """Extract frame index from the buffer."""

    savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
    return savant_frame_meta.idx if savant_frame_meta else None


def create_buffer_processor(
    queue: Queue,
    fps_meter: FPSMeter,
    sources: SourceInfoRegistry,
    objects_preprocessing: ObjectsPreprocessing,
    frame_params: FrameParameters,
    source_output: SourceOutput,
    video_pipeline: VideoPipeline,
):
    """Create buffer processor."""

    if isinstance(source_output, SourceOutputEncoded):
        if isinstance(source_output, SourceOutputH26X) and is_aarch64():
            buffer_processor_class = NvDsJetsonH26XBufferProcessor
        else:
            buffer_processor_class = NvDsEncodedBufferProcessor
        return buffer_processor_class(
            queue=queue,
            fps_meter=fps_meter,
            sources=sources,
            objects_preprocessing=objects_preprocessing,
            frame_params=frame_params,
            codec=source_output.codec,
            video_pipeline=video_pipeline,
        )

    return NvDsRawBufferProcessor(
        queue=queue,
        fps_meter=fps_meter,
        sources=sources,
        objects_preprocessing=objects_preprocessing,
        frame_params=frame_params,
        output_frame=isinstance(source_output, SourceOutputWithFrame),
        video_pipeline=video_pipeline,
    )

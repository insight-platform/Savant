"""Buffer processor for DeepStream pipeline."""
from queue import Queue
from typing import Optional, Union, NamedTuple, Iterator

import numpy as np
import pyds
from pygstsavantframemeta import (
    gst_buffer_get_savant_frame_meta,
    nvds_frame_meta_get_nvds_savant_frame_meta,
)
from pysavantboost import ObjectsPreprocessing

from savant.base.model import ObjectModel, ComplexModel
from savant.config.schema import PipelineElement, ModelElement
from savant.converter.scale import scale_rbbox
from savant.deepstream.nvinfer.model import (
    NvInferRotatedObjectDetector,
    NvInferDetector,
    NvInferAttributeModel,
)
from savant.deepstream.utils import (
    nvds_frame_meta_iterator,
    nvds_obj_meta_iterator,
    nvds_clf_meta_iterator,
    nvds_label_info_iterator,
    nvds_tensor_output_iterator,
    nvds_infer_tensor_meta_to_outputs,
    nvds_add_obj_meta_to_frame,
    nvds_add_attr_meta_to_obj,
    nvds_set_selection_type,
    nvds_set_obj_uid,
)
from savant.gstreamer import Gst  # noqa:F401
from savant.gstreamer.buffer_processor import GstBufferProcessor
from savant.gstreamer.codecs import CodecInfo, Codec
from savant.gstreamer.metadata import metadata_get_frame_meta, metadata_pop_frame_meta
from savant.meta.type import ObjectSelectionType
from savant.utils.fps_meter import FPSMeter
from savant.utils.model_registry import ModelObjectRegistry
from savant.utils.sink_factories import SinkVideoFrame
from savant.utils.source_info import SourceInfoRegistry, SourceInfo


class _OutputFrame(NamedTuple):
    """Output frame with its metadata."""

    idx: int
    pts: int
    frame: Optional[bytes]
    codec: Optional[CodecInfo]
    keyframe: bool


class NvDsBufferProcessor(GstBufferProcessor):
    def __init__(
        self,
        queue: Queue,
        fps_meter: FPSMeter,
        sources: SourceInfoRegistry,
        model_object_registry: ModelObjectRegistry,
        objects_preprocessing: ObjectsPreprocessing,
        frame_width: int,
        frame_height: int,
    ):
        """Buffer processor for DeepStream pipeline.

        :param queue: Queue for output data.
        :param fps_meter: FPS meter.
        :param sources: Source info registry.
        :param model_object_registry: Model.Object registry.
        :param objects_preprocessing: Objects processing registry.
        :param frame_width: Processing frame width (after nvstreammux).
        :param frame_height: Processing frame height (after nvstreammux).
        """

        super().__init__(queue, fps_meter)
        self._sources = sources
        self._model_object_registry = model_object_registry
        self._objects_preprocessing = objects_preprocessing
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._queue = queue

    def prepare_input(self, buffer: Gst.Buffer):
        """Input meta processor.

        :param buffer: gstreamer buffer that is being processed.
        """

        self._logger.debug('Preparing input for buffer with PTS %s.', buffer.pts)
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            # TODO: add source_id to SavantFrameMeta and always attach SavantFrameMeta to the buffers
            source_id = self._sources.get_id_by_pad_index(nvds_frame_meta.pad_index)
            savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(
                nvds_frame_meta
            )
            frame_idx = savant_frame_meta.idx if savant_frame_meta else None
            frame_pts = nvds_frame_meta.buf_pts

            self._logger.debug(
                'Preparing input for frame %s of source %s.',
                nvds_frame_meta.buf_pts,
                source_id,
            )
            frame_meta = metadata_get_frame_meta(source_id, frame_idx, frame_pts)

            # add external objects to nvds meta
            for obj_meta in frame_meta.metadata['objects']:
                obj_key = self._model_object_registry.model_object_key(
                    obj_meta['model_name'], obj_meta['label']
                )
                # skip frame, will be added with proper width/height
                if obj_key == 'frame':
                    continue
                # obj_key was only registered if
                # it was required by the pipeline model elements (this case)
                # or equaled the output object of one of the pipeline model elements
                if self._model_object_registry.is_model_object_key_registered(obj_key):
                    (
                        model_uid,
                        class_id,
                    ) = self._model_object_registry.get_model_object_ids(obj_key)
                    if obj_meta['bbox']['angle']:
                        scaled_bbox = scale_rbbox(
                            bboxes=np.array(
                                [
                                    [
                                        obj_meta['bbox']['xc'],
                                        obj_meta['bbox']['yc'],
                                        obj_meta['bbox']['width'],
                                        obj_meta['bbox']['height'],
                                        obj_meta['bbox']['angle'],
                                    ]
                                ]
                            ),
                            scale_factor_x=self._frame_width,
                            scale_factor_y=self._frame_height,
                        )[0]
                        selection_type = ObjectSelectionType.ROTATED_BBOX
                    else:
                        scaled_bbox = (
                            obj_meta['bbox']['xc'] * self._frame_width,
                            obj_meta['bbox']['yc'] * self._frame_height,
                            obj_meta['bbox']['width'] * self._frame_width,
                            obj_meta['bbox']['height'] * self._frame_height,
                            obj_meta['bbox']['angle'],
                        )
                        selection_type = ObjectSelectionType.REGULAR_BBOX

                    nvds_add_obj_meta_to_frame(
                        batch_meta=nvds_batch_meta,
                        frame_meta=nvds_frame_meta,
                        selection_type=selection_type,
                        class_id=class_id,
                        gie_uid=model_uid,
                        bbox=scaled_bbox,
                        object_id=obj_meta['object_id'],
                        obj_label=obj_key,
                        confidence=obj_meta['confidence'],
                    )

            # add primary frame object
            obj_label = 'frame'
            model_uid, class_id = self._model_object_registry.get_model_object_ids(
                obj_label
            )
            nvds_add_obj_meta_to_frame(
                batch_meta=nvds_batch_meta,
                frame_meta=nvds_frame_meta,
                selection_type=ObjectSelectionType.REGULAR_BBOX,
                class_id=class_id,
                gie_uid=model_uid,
                # tuple(xc, yc, width, height, angle)
                bbox=(
                    self._frame_width / 2,
                    self._frame_height / 2,
                    self._frame_width,
                    self._frame_height,
                    0,
                ),
                obj_label=obj_label,
                # confidence should be bigger than tracker minDetectorConfidence
                # to prevent the tracker from deleting the object
                # use tracker display-tracking-id=0 to avoid labelling
                confidence=0.999,
            )

            nvds_frame_meta.bInferDone = True  # required for tracker (DS 6.0)

    def prepare_output(
        self,
        buffer: Gst.Buffer,
        source_info: SourceInfo,
    ) -> SinkVideoFrame:
        """Enqueue output messages based on frame meta.

        :param buffer: gstreamer buffer that is being processed.
        :param source_info: output source info
        """

        self._logger.debug(
            'Preparing output for buffer with PTS %s for source %s.',
            buffer.pts,
            source_info.source_id,
        )
        for output_frame in self._iterate_output_frames(buffer):
            self._logger.debug(
                'Preparing output for frame %s with PTS %s of source %s.',
                output_frame.idx,
                output_frame.pts,
                source_info.source_id,
            )
            frame_meta = metadata_pop_frame_meta(
                source_info.source_id,
                output_frame.idx,
                output_frame.pts,
            )
            yield SinkVideoFrame(
                source_id=source_info.source_id,
                frame_meta=frame_meta,
                frame_width=source_info.dest_resolution.width,
                frame_height=source_info.dest_resolution.height,
                frame=output_frame.frame,
                frame_codec=output_frame.codec,
                keyframe=output_frame.keyframe,
            )

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
            and not model.input.preprocess_object_tensor
        ):
            self._logger.debug(
                'No need to preprocess input for element %s.', element.full_name
            )
            return

        self._logger.debug('Preprocessing input for element %s.', element.full_name)

        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if model.input.preprocess_object_meta:
            for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
                for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                    if not self._is_model_input_object(element, nvds_obj_meta):
                        continue
                    # TODO: Unify and also switch to the box representation system
                    #  through the center point during meta preprocessing.
                    bbox = pyds.NvBbox_Coords()  # why not?
                    bbox.left = nvds_obj_meta.rect_params.left
                    bbox.top = nvds_obj_meta.rect_params.top
                    bbox.width = nvds_obj_meta.rect_params.width
                    bbox.height = nvds_obj_meta.rect_params.height

                    parent_bbox = pyds.NvBbox_Coords()
                    if nvds_obj_meta.parent:
                        parent_bbox.left = nvds_obj_meta.parent.rect_params.left
                        parent_bbox.top = nvds_obj_meta.parent.rect_params.top
                        parent_bbox.width = nvds_obj_meta.parent.rect_params.width
                        parent_bbox.height = nvds_obj_meta.parent.rect_params.height
                    else:
                        parent_bbox.left = 0
                        parent_bbox.top = 0
                        parent_bbox.width = self._frame_width
                        parent_bbox.height = self._frame_height

                    bbox = model.input.preprocess_object_meta(
                        bbox, parent_bbox=parent_bbox
                    )

                    rect_params = nvds_obj_meta.rect_params
                    rect_params.left = bbox.left
                    rect_params.top = bbox.top
                    rect_params.width = bbox.width
                    rect_params.height = bbox.height

        elif model.input.preprocess_object_tensor:
            model_uid, class_id = self._model_object_registry.get_model_object_ids(
                model.input.object
            )
            self._objects_preprocessing.preprocessing(
                element.name,
                hash(buffer),
                model_uid,
                class_id,
                model.input.preprocess_object_tensor.padding[0],
                model.input.preprocess_object_tensor.padding[1],
            )

        self._logger.debug('Preprocessed input for element %s.', element.full_name)

    def prepare_element_output(self, element: PipelineElement, buffer: Gst.Buffer):
        """Model output postprocessing.

        :param element: element that this probe was added to.
        :param buffer: gstreamer buffer that is being processed.
        """
        if not isinstance(element, ModelElement):
            return

        self._logger.debug('Postprocessing output for element %s.', element.full_name)

        model_uid = self._model_object_registry.get_model_uid(element.name)
        model: Union[
            NvInferRotatedObjectDetector,
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
                        # parse and post-process model output
                        output_layers = nvds_infer_tensor_meta_to_outputs(
                            tensor_meta=tensor_meta,
                            layer_names=model.output.layer_names,
                        )
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

                            if bbox_tensor.shape[1] == 6:  # no angle
                                # xc -> left, yc -> top
                                bbox_tensor[:, 2] -= bbox_tensor[:, 4] / 2
                                bbox_tensor[:, 3] -= bbox_tensor[:, 5] / 2

                                # clip
                                # width to right, height to bottom
                                bbox_tensor[:, 4] += bbox_tensor[:, 2]
                                bbox_tensor[:, 5] += bbox_tensor[:, 3]
                                # clip
                                bbox_tensor[:, 2][bbox_tensor[:, 2] < 0.0] = 0.0
                                bbox_tensor[:, 3][bbox_tensor[:, 3] < 0.0] = 0.0
                                bbox_tensor[:, 4][
                                    bbox_tensor[:, 4] > self._frame_width - 1.0
                                ] = (self._frame_width - 1.0)
                                bbox_tensor[:, 5][
                                    bbox_tensor[:, 5] > self._frame_height - 1.0
                                ] = (self._frame_height - 1.0)

                                # right to width, bottom to height
                                bbox_tensor[:, 4] -= bbox_tensor[:, 2]
                                bbox_tensor[:, 5] -= bbox_tensor[:, 3]

                                # left -> xc , top-> yc
                                bbox_tensor[:, 2] += bbox_tensor[:, 4] / 2
                                bbox_tensor[:, 3] += bbox_tensor[:, 5] / 2

                                # add 0 angle
                                bbox_tensor = np.concatenate(
                                    [bbox_tensor, np.zeros((bbox_tensor.shape[0], 1))],
                                    axis=1,
                                )

                            # add index column to further filter attribute values
                            bbox_tensor = np.concatenate(
                                [
                                    bbox_tensor,
                                    np.arange(bbox_tensor.shape[0]).reshape(-1, 1),
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
                                    cls_bbox_tensor = obj.selector(cls_bbox_tensor)

                                obj_label = (
                                    self._model_object_registry.model_object_key(
                                        element.name, obj.label
                                    )
                                )
                                for bbox in cls_bbox_tensor:
                                    # add NvDsObjectMeta
                                    _nvds_obj_meta = nvds_add_obj_meta_to_frame(
                                        batch_meta=nvds_batch_meta,
                                        frame_meta=nvds_frame_meta,
                                        selection_type=model.output.selection_type,
                                        class_id=obj.class_id,
                                        gie_uid=model_uid,
                                        bbox=bbox[2:7],
                                        parent=parent_nvds_obj_meta,
                                        obj_label=obj_label,
                                        confidence=bbox[1],
                                    )
                                    selected_bboxes.append(
                                        (int(bbox[7]), _nvds_obj_meta)
                                    )

                        if values:
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
                                nvds_set_selection_type(
                                    obj_meta=nvds_obj_meta,
                                    selection_type=ObjectSelectionType.REGULAR_BBOX,
                                )
                                nvds_set_obj_uid(
                                    frame_meta=nvds_frame_meta, obj_meta=nvds_obj_meta
                                )
                                nvds_obj_meta.obj_label = (
                                    self._model_object_registry.model_object_key(
                                        element.name, obj.label
                                    )
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
                    or model.input.preprocess_object_tensor
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
        if model.input.preprocess_object_tensor:
            self._objects_preprocessing.restore_frame(hash(buffer))

        self._logger.debug('Prepared output for element %s.', element.full_name)

    def _is_model_input_object(
        self, element: ModelElement, nvds_obj_meta: pyds.NvDsObjectMeta
    ):
        model_uid, class_id = self._model_object_registry.get_model_object_ids(
            element.model.input.object
        )
        return (
            nvds_obj_meta.unique_component_id == model_uid
            and nvds_obj_meta.class_id == class_id
        )

    def _iterate_output_frames(self, buffer: Gst.Buffer) -> Iterator[_OutputFrame]:
        """Iterate output frames."""


class NvDsEncodedBufferProcessor(NvDsBufferProcessor):
    def __init__(
        self,
        queue: Queue,
        fps_meter: FPSMeter,
        sources: SourceInfoRegistry,
        model_object_registry: ModelObjectRegistry,
        objects_preprocessing: ObjectsPreprocessing,
        frame_width: int,
        frame_height: int,
        codec: CodecInfo,
    ):
        """Buffer processor for DeepStream pipeline.

        :param queue: Queue for output data.
        :param fps_meter: FPS meter.
        :param sources: Source info registry.
        :param model_object_registry: Model.Object registry.
        :param objects_preprocessing: Objects processing registry.
        :param frame_width: Processing frame width (after nvstreammux).
        :param frame_height: Processing frame height (after nvstreammux).
        :param codec: Codec of the output frames.
        """

        self._codec = codec
        super().__init__(
            queue=queue,
            fps_meter=fps_meter,
            sources=sources,
            model_object_registry=model_object_registry,
            objects_preprocessing=objects_preprocessing,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def _iterate_output_frames(self, buffer: Gst.Buffer) -> Iterator[_OutputFrame]:
        """Iterate output frames from Gst.Buffer."""

        # get encoded frame for output
        frame = buffer.extract_dup(0, buffer.get_size())
        savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
        frame_idx = savant_frame_meta.idx if savant_frame_meta else None
        frame_pts = buffer.pts
        is_keyframe = not buffer.has_flags(Gst.BufferFlags.DELTA_UNIT)
        yield _OutputFrame(
            idx=frame_idx,
            pts=frame_pts,
            frame=frame,
            codec=self._codec,
            keyframe=is_keyframe,
        )


class NvDsRawBufferProcessor(NvDsBufferProcessor):
    def __init__(
        self,
        queue: Queue,
        fps_meter: FPSMeter,
        sources: SourceInfoRegistry,
        model_object_registry: ModelObjectRegistry,
        objects_preprocessing: ObjectsPreprocessing,
        frame_width: int,
        frame_height: int,
        output_frame: bool,
    ):
        """Buffer processor for DeepStream pipeline.

        :param queue: Queue for output data.
        :param fps_meter: FPS meter.
        :param sources: Source info registry.
        :param model_object_registry: Model.Object registry.
        :param objects_preprocessing: Objects processing registry.
        :param frame_width: Processing frame width (after nvstreammux).
        :param frame_height: Processing frame height (after nvstreammux).
        :param output_frame: Whether to output frame or not.
        """

        self._output_frame = output_frame
        self._codec = Codec.RAW_RGBA.value if output_frame else None
        super().__init__(
            queue=queue,
            fps_meter=fps_meter,
            sources=sources,
            model_object_registry=model_object_registry,
            objects_preprocessing=objects_preprocessing,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def _iterate_output_frames(self, buffer: Gst.Buffer) -> Iterator[_OutputFrame]:
        """Iterate output frames from NvDs batch.

        NvDs batch contains raw RGBA frames. They are all keyframes.
        """

        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            # get frame if required for output
            frame = (
                pyds.get_nvds_buf_surface(
                    hash(buffer), nvds_frame_meta.batch_id
                ).tobytes()
                if self._output_frame
                else None
            )
            savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(
                nvds_frame_meta
            )
            frame_idx = savant_frame_meta.idx if savant_frame_meta else None
            frame_pts = nvds_frame_meta.buf_pts
            yield _OutputFrame(
                idx=frame_idx,
                pts=frame_pts,
                frame=frame,
                codec=self._codec,
                # Any frame is keyframe since it was not encoded
                keyframe=True,
            )

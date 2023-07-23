"""NvInfer PyFunc Postprocessor."""
import numpy as np
import pyds

from savant_rs.utils.symbol_mapper import (
    parse_compound_key,
    get_model_id,
    get_object_id,
    build_model_object_key,
)
from savant.config.schema import ModelElement
from savant.meta.type import ObjectSelectionType
from savant.gstreamer import Gst  # noqa:F401
from savant.deepstream.utils import (
    nvds_frame_meta_iterator,
    nvds_obj_meta_iterator,
    nvds_clf_meta_iterator,
    nvds_label_info_iterator,
    nvds_tensor_output_iterator,
    nvds_infer_tensor_meta_to_outputs,
    nvds_add_obj_meta_to_frame,
    nvds_add_attr_meta_to_obj,
    nvds_set_obj_selection_type,
    nvds_set_obj_uid,
)
from savant.deepstream.nvinfer.model import NvInferDetector, NvInferComplexModel
from savant.deepstream.pyfunc import NvDsPyFuncPlugin


class NvInferPostprocessor(NvDsPyFuncPlugin):
    """NvInfer PyFunc postprocessor element."""

    def process_buffer(self, buffer: Gst.Buffer):
        """Process Gst.Buffer with NvInfer model output.

        :param buffer: Gstreamer buffer.
        """
        element = self.gst_element.pyobject

        if self.is_debug():
            self.logger.debug(
                'Preparing "%s" element output for buffer with PTS %s.',
                element.name,
                buffer.pts,
            )

        model_uid = get_model_id(element.name)

        is_detector = isinstance(element.model, NvInferDetector)

        input_model_uid, input_class_id = get_object_id(
            *parse_compound_key(element.model.input.object)
        )

        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):

                # custom model
                if element.model.output.converter:
                    if (
                        nvds_obj_meta.unique_component_id == input_model_uid
                        and nvds_obj_meta.class_id == input_class_id
                    ):
                        self.process_custom_model_output(
                            element=element,
                            model_uid=model_uid,
                            nvds_batch_meta=nvds_batch_meta,
                            nvds_frame_meta=nvds_frame_meta,
                            nvds_obj_meta=nvds_obj_meta,
                        )

                # regular detector
                elif is_detector:
                    if nvds_obj_meta.unique_component_id == model_uid:
                        self.process_regular_detector_output(
                            element=element,
                            model_uid=model_uid,
                            nvds_batch_meta=nvds_batch_meta,
                            nvds_frame_meta=nvds_frame_meta,
                            nvds_obj_meta=nvds_obj_meta,
                        )

                # regular classifier
                else:
                    self.process_regular_classifier_output(
                        element=element,
                        model_uid=model_uid,
                        nvds_batch_meta=nvds_batch_meta,
                        nvds_frame_meta=nvds_frame_meta,
                        nvds_obj_meta=nvds_obj_meta,
                    )

    @staticmethod
    def process_custom_model_output(
        element: ModelElement,
        model_uid: int,
        nvds_batch_meta: pyds.NvDsBatchMeta,
        nvds_frame_meta: pyds.NvDsFrameMeta,
        nvds_obj_meta: pyds.NvDsObjectMeta,
    ):
        """Converts custom model output and save meta."""
        is_complex_model = isinstance(element.model, NvInferComplexModel)
        is_detector = isinstance(element.model, NvInferDetector)

        parent_nvds_obj_meta = nvds_obj_meta
        for tensor_meta in nvds_tensor_output_iterator(
            parent_nvds_obj_meta, gie_uid=model_uid
        ):
            # parse and post-process model output
            output_layers = nvds_infer_tensor_meta_to_outputs(
                tensor_meta=tensor_meta,
                layer_names=element.model.output.layer_names,
            )
            outputs = element.model.output.converter(
                *output_layers,
                model=element.model,
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

            # detector
            elif is_detector:
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
                    selection_type = ObjectSelectionType.REGULAR_BBOX
                    # add 0 angle
                    bbox_tensor = np.concatenate(
                        [
                            bbox_tensor,
                            np.zeros((bbox_tensor.shape[0], 1), dtype=np.float32),
                        ],
                        axis=1,
                    )
                else:
                    selection_type = ObjectSelectionType.ROTATED_BBOX

                # add index column to further filter attribute values
                bbox_tensor = np.concatenate(
                    [
                        bbox_tensor,
                        np.arange(bbox_tensor.shape[0], dtype=np.float32).reshape(
                            -1, 1
                        ),
                    ],
                    axis=1,
                )

                selected_bboxes = []
                for obj in element.model.output.objects:
                    cls_bbox_tensor = bbox_tensor[bbox_tensor[:, 0] == obj.class_id]
                    if cls_bbox_tensor.shape[0] == 0:
                        continue
                    if obj.selector:
                        cls_bbox_tensor = obj.selector(cls_bbox_tensor)

                    obj_label = build_model_object_key(element.name, obj.label)
                    for bbox in cls_bbox_tensor:
                        # add NvDsObjectMeta
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
                        selected_bboxes.append((int(bbox[7]), _nvds_obj_meta))

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
                for (_, _nvds_obj_meta), _values in zip(selected_bboxes, values):
                    for attr_name, value, confidence in _values:
                        nvds_add_attr_meta_to_obj(
                            frame_meta=nvds_frame_meta,
                            obj_meta=_nvds_obj_meta,
                            element_name=element.name,
                            name=attr_name,
                            value=value,
                            confidence=confidence,
                        )

    @staticmethod
    def process_regular_detector_output(  # pylint: disable=unused-argument
        element: ModelElement,
        model_uid: int,
        nvds_batch_meta: pyds.NvDsBatchMeta,
        nvds_frame_meta: pyds.NvDsFrameMeta,
        nvds_obj_meta: pyds.NvDsObjectMeta,
    ):
        """Correct nvds_obj_meta.obj_label."""
        for obj in element.model.output.objects:
            if nvds_obj_meta.class_id == obj.class_id:
                nvds_set_obj_selection_type(
                    obj_meta=nvds_obj_meta,
                    selection_type=ObjectSelectionType.REGULAR_BBOX,
                )
                nvds_set_obj_uid(frame_meta=nvds_frame_meta, obj_meta=nvds_obj_meta)
                nvds_obj_meta.obj_label = build_model_object_key(
                    element.name, obj.label
                )
                break

    @staticmethod
    def process_regular_classifier_output(  # pylint: disable=unused-argument
        element: ModelElement,
        model_uid: int,
        nvds_batch_meta: pyds.NvDsBatchMeta,
        nvds_frame_meta: pyds.NvDsFrameMeta,
        nvds_obj_meta: pyds.NvDsObjectMeta,
    ):
        """Converts nvds_clf_meta to attr_meta."""
        for nvds_clf_meta in nvds_clf_meta_iterator(nvds_obj_meta):
            if nvds_clf_meta.unique_component_id != model_uid:
                continue
            for attr, label_info in zip(
                element.model.output.attributes,
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

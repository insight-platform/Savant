"""Analytics module."""
from typing import Dict, Tuple

from savant.deepstream.utils import nvds_obj_meta_iterator
from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.meta.constants import PRIMARY_OBJECT_LABEL  # , PRIMARY_OBJECT_KEY
from savant.parameter_storage import param_storage


class Check(NvDsPyFuncPlugin):
    """Age and gender smoothing pyfunc.
    On each frame

    :key history_length: int, Length of the vector of historical values
    """

    def __init__(self, name_output: str, **kwargs):
        super().__init__(**kwargs)
        self.name_output = name_output

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        print("!!!!!!!!!!!!!!!!!!!!!!1")
        for obj in frame_meta.objects:
            print(obj.label)
            print(obj.bbox.width)

        # for nvds_obj_meta in nvds_obj_meta_iterator(frame_meta.frame_meta):
        #     if (
        #         nvds_obj_meta.parent is not None
        #         and nvds_obj_meta.parent.obj_label != PRIMARY_OBJECT_KEY
        #     ):
        #         print(self.name_output)
        #         print(f"frame num: {frame_meta.frame_num}")
        #         print("-------------------------")
        #         print(
        #             "!!!! nvds_obj_meta.parent.obj_label=",
        #             nvds_obj_meta.parent.obj_label,
        #         )
        #         print(
        #             "!!!!! nvds_obj_meta.parent.unique_component_id",
        #             nvds_obj_meta.parent.unique_component_id,
        #         )
        #         print(
        #             "!!!! nvds_obj_meta.parent.object_id",
        #             nvds_obj_meta.parent.object_id,
        #         )
        #         print(
        #             "(!!!! nvds_obj_meta.parent.class_id", nvds_obj_meta.parent.class_id
        #         )

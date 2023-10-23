from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from ctypes import *
import pyds


STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080

skeleton = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]


class KeyPoint(NvDsPyFuncPlugin):
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     with open(self.config_path, 'r', encoding='utf8') as stream:
    #         self.line_config = yaml.safe_load(stream)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        for obj in frame_meta.objects:
            if obj.label == "person":
                print(obj.object_meta_impl.ds_object_meta.obj_label)
                print(
                    "mask_params.size=",
                    obj.object_meta_impl.ds_object_meta.mask_params.size,
                )
                print(
                    "mask_params.height=",
                    obj.object_meta_impl.ds_object_meta.mask_params.height,
                )
                print(
                    "mask_params.width=",
                    obj.object_meta_impl.ds_object_meta.mask_params.width,
                )
                np_mask = (
                    obj.object_meta_impl.ds_object_meta.mask_params.get_mask_array()
                )
                print("np_mask.shape=", np_mask.shape)
                print("np_mask=", np_mask)
                print("np_mask.dtype=", np_mask.dtype)
                obj_meta = obj.object_meta_impl.ds_object_meta
                num_joints = int(obj_meta.mask_params.size / (sizeof(c_float) * 3))

                gain = min(
                    obj_meta.mask_params.width / STREAMMUX_WIDTH,
                    obj_meta.mask_params.height / STREAMMUX_HEIGHT,
                )
                pad_x = (obj_meta.mask_params.width - STREAMMUX_WIDTH * gain) / 2.0
                pad_y = (obj_meta.mask_params.height - STREAMMUX_HEIGHT * gain) / 2.0

                # batch_meta = frame_meta.batch_meta
                # display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                # pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

                for i in range(num_joints):
                    data = obj_meta.mask_params.get_mask_array()
                    xc = int((data[i * 3 + 0] - pad_x) / gain)
                    yc = int((data[i * 3 + 1] - pad_y) / gain)
                    confidence = data[i * 3 + 2]
                    print("xc=", xc, "yc=", yc, "confidence=", confidence)

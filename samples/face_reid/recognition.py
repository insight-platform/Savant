"""Recognition module."""
from typing import Tuple
import hnswlib
import numpy as np

from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.parameter_storage import param_storage

from samples.face_reid.utils import unpack_person_id_img_n

MODEL_NAME = param_storage()['reid_model_name']


class Recognition(NvDsPyFuncPlugin):
    """
    """

    def on_start(self) -> bool:
        """Do on plugin start."""
        self.index = hnswlib.Index(space=self.index_space, dim=self.index_dim)
        self.index.load_index('/index/index.bin', max_elements=self.index_max_elements)
        return True


    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """

        for obj_meta in frame_meta.objects:
            if obj_meta.label == 'face':
                feature = obj_meta.get_attr_meta(MODEL_NAME, 'feature').value

                label, distance = self.index.knn_query(feature, k=1)

                person_id, img_n = unpack_person_id_img_n(label.item())

                obj_meta.add_attr_meta(
                    'recognition', 'person_id', person_id
                )



"""Recognition module."""

import hnswlib


from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.parameter_storage import param_storage

from samples.face_reid.utils import unpack_person_id_img_n

MODEL_NAME = param_storage()['reid_model_name']
INDEX_PATH = '/opt/savant/samples/face_reid/person_index/index.bin'

class Recognition(NvDsPyFuncPlugin):
    """ """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.track_person_ids = {}


    def on_start(self) -> bool:
        """Do on plugin start."""
        self.index = hnswlib.Index(space=self.index_space, dim=self.index_dim)
        self.index.load_index(INDEX_PATH, max_elements=self.index_max_elements)

        return True

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """

        features = []
        for obj_meta in frame_meta.objects:
            if obj_meta.label == 'face':
                features.append(obj_meta.get_attr_meta(MODEL_NAME, 'feature').value)

        if len(features) < 1:
            return

        labels, distances = self.index.knn_query(features, k=1)


        person_idx = 0
        for obj_meta in frame_meta.objects:
            if obj_meta.label == 'face':
                distance = distances[person_idx]
                if distance > self.dist_threshold:
                    obj_meta.add_attr_meta('recognition', 'person_id', -1)
                    obj_meta.add_attr_meta('recognition', 'image_n', -1)
                else:
                    label = labels[person_idx]
                    person_id, img_n = unpack_person_id_img_n(label)
                    obj_meta.add_attr_meta('recognition', 'person_id', person_id)
                    obj_meta.add_attr_meta('recognition', 'image_n', img_n)
                person_idx += 1

"""Recognition module."""
import os
import hnswlib

from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.parameter_storage import param_storage
from savant.meta.constants import UNTRACKED_OBJECT_ID
from samples.face_reid.utils import unpack_person_id_img_n

MODEL_NAME = param_storage()['reid_model_name']


class Recognition(NvDsPyFuncPlugin):
    """Uses HNSW index to find nearest neighbor for each face feature vector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_match_for_track = {}
        self.index_file_path = os.path.join(self.index_dir, 'index.bin')

    def on_start(self) -> bool:
        """Do on plugin start."""
        self.index = hnswlib.Index(space=self.index_space, dim=self.index_dim)
        self.index.load_index(
            self.index_file_path, max_elements=self.index_max_elements
        )
        return True

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """

        # gather all face feature vectors from this frame
        features = []
        for obj_meta in frame_meta.objects:
            if obj_meta.label == 'face':
                features.append(obj_meta.get_attr_meta(MODEL_NAME, 'feature').value)

        if len(features) < 1:
            return

        # search for nearest neighbor for each feature vector
        labels, distances = self.index.knn_query(features, k=1)

        person_idx = 0
        for obj_meta in frame_meta.objects:
            if obj_meta.label == 'face':
                distance = distances[person_idx].item()
                label = labels[person_idx].item()
                person_id, img_n = unpack_person_id_img_n(label)

                if distance > self.dist_threshold:
                    # current frame match unsuccessful
                    if (
                        obj_meta.track_id != UNTRACKED_OBJECT_ID
                        and obj_meta.track_id in self.last_match_for_track
                    ):
                        # fallback on last successful match
                        person_id, img_n = self.last_match_for_track[obj_meta.track_id]
                    else:
                        # no match found
                        person_id = -1
                        img_n = -1
                elif obj_meta.track_id != UNTRACKED_OBJECT_ID:
                    # current frame match successful
                    # save match for future frames
                    self.last_match_for_track[obj_meta.track_id] = (person_id, img_n)

                obj_meta.add_attr_meta('recognition', 'person_id', person_id)
                obj_meta.add_attr_meta('recognition', 'image_n', img_n)
                obj_meta.add_attr_meta('recognition', 'distance', distance)
                person_idx += 1

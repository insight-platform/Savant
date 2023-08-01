"""Index builder module."""
import os
import hnswlib

from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.parameter_storage import param_storage


from samples.face_reid.utils import pack_person_id_img_n

MODEL_NAME = param_storage()['reid_model_name']


class IndexBuilder(NvDsPyFuncPlugin):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = hnswlib.Index(space=self.index_space, dim=self.index_dim)
        self.index.init_index(max_elements=self.index_max_elements, ef_construction = 200, M = 16)
        self.person_names = []

    def on_stop(self) -> bool:
        """Do on plugin stop."""
        self.index.save_index('/index/index.bin')
        return True

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """

        if frame_meta.objects_number != 2:
            self.logger.warn('%s faces detected on %s, 1 is expected. Not adding features to index.', frame_meta.objects_number - 1, location)
        else:
            location = frame_meta.tags['location']
            root, _ = os.path.splitext(location)
            file_name = os.path.basename(root)
            person_name, img_n = file_name.rsplit('_', maxsplit=1)
            try:
                person_id = self.person_names.index(person_name)
            except ValueError:
                person_id = len(self.person_names)
                self.person_names.append(person_name)

            img_n = int(img_n)
            for obj_meta in frame_meta.objects:
                if obj_meta.label == 'face':
                    feature = obj_meta.get_attr_meta(MODEL_NAME, 'feature').value
                    feature_id = pack_person_id_img_n(person_id, img_n)
                    self.index.add_items(feature, feature_id)

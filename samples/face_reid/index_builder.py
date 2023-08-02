"""Index builder module."""
import os
import shutil
import hnswlib
import cv2
from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.parameter_storage import param_storage
from savant.utils.artist import Artist
from savant.deepstream.opencv_utils import (
    nvds_to_gpu_mat,
)
from samples.face_reid.utils import pack_person_id_img_n

MODEL_NAME = param_storage()['reid_model_name']
INDEX_DIR = '/opt/savant/samples/face_reid/person_index/'
INDEX_FILE = os.path.join(INDEX_DIR, 'index.bin')
PROCESSED_GALLERY_DIR = '/opt/savant/samples/face_reid/processed_gallery/'

class IndexBuilder(NvDsPyFuncPlugin):
    """Index builder plugin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = hnswlib.Index(space=self.index_space, dim=self.index_dim)
        self.index.init_index(
            max_elements=self.index_max_elements, ef_construction=200, M=16
        )
        self.person_names = []
        shutil.rmtree(os.path.dirname(PROCESSED_GALLERY_DIR), ignore_errors=True)
        os.makedirs(os.path.dirname(PROCESSED_GALLERY_DIR))

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        location = frame_meta.tags['location']
        if frame_meta.objects_number != 2:
            self.logger.warn(
                '%s faces detected on %s, 1 is expected. Not adding features to index.',
                frame_meta.objects_number - 1,
                location,
            )
        else:
            stream = self.get_cuda_stream(frame_meta)
            with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
                with Artist(frame_mat, stream) as artist:



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

                            face_img = artist.copy_frame_region(obj_meta.bbox).download()
                            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                            img_path = os.path.join(PROCESSED_GALLERY_DIR, f'{person_name}_{person_id:03d}_{img_n:03d}.jpeg')
                            cv2.imwrite(img_path, face_img)
                    self.write_index()

    def write_index(self, index_path: str = INDEX_FILE):
        """Write index to disk."""
        shutil.rmtree(os.path.dirname(index_path), ignore_errors=True)
        os.makedirs(os.path.dirname(index_path))
        self.index.save_index(index_path)

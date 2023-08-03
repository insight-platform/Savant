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

REID_MODEL_NAME = param_storage()['reid_model_name']


class IndexBuilder(NvDsPyFuncPlugin):
    """Index builder plugin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = hnswlib.Index(space=self.index_space, dim=self.index_dim)
        self.index.init_index(
            max_elements=self.index_max_elements,
            ef_construction=self.index_ef_construction,
            M=self.index_m,
        )
        self.person_names = []

        self.index_file_path = os.path.join(self.index_dir, 'index.bin')
        self.processed_gallery_dir = os.path.join(self.index_dir, 'processed_gallery')

        shutil.rmtree(self.processed_gallery_dir, ignore_errors=True)
        os.makedirs(self.processed_gallery_dir)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        # image files source adapter adds location tag
        # which contains image file name
        location = frame_meta.tags['location']

        if frame_meta.objects_number < 2:
            # no faces detected and
            # only primary object is present
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
                    # gallery images are named as <person_name>_<img_n>.jpeg
                    person_name, img_n = file_name.rsplit('_', maxsplit=1)
                    # assign person_id to person_name in the order of appearance
                    try:
                        person_id = self.person_names.index(person_name)
                    except ValueError:
                        person_id = len(self.person_names)
                        self.person_names.append(person_name)

                    img_n = int(img_n)
                    for obj_meta in frame_meta.objects:
                        if obj_meta.label == 'face':
                            # get face feature vector from metadata
                            feature = obj_meta.get_attr_meta(
                                REID_MODEL_NAME, 'feature'
                            ).value
                            # index stores single int label for each feature
                            # we assume that there are no more than 2^32 people or images for each person
                            # and pack both person_id and img_n into single int
                            feature_id = pack_person_id_img_n(person_id, img_n)
                            self.index.add_items(feature, feature_id)

                            # save face image to disk
                            # it will be used to visualize face matches to gallery
                            face_img = artist.copy_frame_region(
                                obj_meta.bbox
                            ).download()
                            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2BGR)
                            img_path = os.path.join(
                                self.processed_gallery_dir,
                                f'{person_name}_{person_id:03d}_{img_n:03d}.jpeg',
                            )
                            cv2.imwrite(img_path, face_img)
            # refresh index on disk
            self.write_index()

    def write_index(self):
        """Write index to disk."""
        shutil.rmtree(self.index_file_path, ignore_errors=True)
        os.makedirs(os.path.dirname(self.index_file_path), exist_ok=True)
        self.index.save_index(self.index_file_path)

"""Index builder module."""

import sys
import os
import time
import pathlib
import shutil
import cv2
import numpy as np
import hnswlib

from samples.face_reid.utils import pack_person_id_img_n
from savant.client import  JpegSource, SinkBuilder, SourceBuilder


print('Starting Savant client...')

try:
    zmq_src_socket = os.environ['ZMQ_SRC_ENDPOINT']
    zmq_sink_socket = os.environ['ZMQ_SINK_ENDPOINT']
except KeyError:
    print('ZMQ_SRC_ENDPOINT and ZMQ_SINK_ENDPOINT environment variables must be set.')
    sys.exit(1)


source_id = 'test-source'
shutdown_auth = 'shutdown'
index_dir = '/index'

processed_gallery_dir = os.path.join(index_dir, 'processed_gallery')

shutil.rmtree(processed_gallery_dir, ignore_errors=True)
os.makedirs(processed_gallery_dir)

# Build the source
source = (
    SourceBuilder()
    .with_socket(zmq_src_socket)
    .build()
)

sink = (
    SinkBuilder()
    .with_socket(zmq_sink_socket)
    .with_idle_timeout(10)
    .build()
)

src_jpegs = [
    JpegSource(source_id, str(img_path))
    for img_path in sorted(pathlib.Path('/gallery').glob('*.jpeg'))
]

for src_jpeg in src_jpegs:
    source(src_jpeg, send_eos=False)
source.send_eos(source_id)

time.sleep(1)  # Wait for the module to process the frame

index_space='cosine'
# index_dim is set according to the reid model output dimensions
index_dim = 512
# hnswlib index parameter
index_max_elements = 100
index_ef_construction = 200
index_m = 16

index = hnswlib.Index(space=index_space, dim=index_dim)
index.init_index(
    max_elements=index_max_elements,
    ef_construction=index_ef_construction,
    M=index_m,
)


print('Receiving results from the module...')

feature_namespace = 'adaface_ir50_webface4m_90fb74c'
feature_name = 'feature'

person_names = []
results_count = 0
for result in sink:

    if result.eos:
        continue

    img = np.frombuffer(result.frame_content, dtype=np.uint8)
    img = img.reshape(result.frame_meta.height, result.frame_meta.width, 4)
    print(img.shape)

    loc_attr = result.frame_meta.get_attribute('default', 'location')
    location = loc_attr.values[0].as_string()

    root, _ = os.path.splitext(location)
    file_name = os.path.basename(root)
    # gallery images are named as <person_name>_<img_n>.jpeg
    person_name, img_n = file_name.rsplit('_', maxsplit=1)
    # assign person_id to person_name in the order of appearance
    try:
        person_id = person_names.index(person_name)
    except ValueError:
        person_id = len(person_names)
        person_names.append(person_name)

    img_n = int(img_n)

    # get the face feature vector from metadata
    for obj in result.frame_meta.get_all_objects():
        if obj.label == 'face':
            feature_attr = obj.get_attribute(feature_namespace, feature_name)
            feature = feature_attr.values[0].as_floats()

            # index stores single int label for each feature
            # we assume that the gallery holds no more than 2^32 people or images per person
            # and pack both person_id and img_n into single int64
            feature_id = pack_person_id_img_n(person_id, img_n)
            index.add_items(feature, feature_id)

            left, top, right, bottom = obj.detection_box.as_ltrb_int()

            try:
                face_img = img[top:bottom, left:right]
            except:
                print('Error')
            else:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2BGR)
                img_path = os.path.join(
                    processed_gallery_dir,
                    f'{person_name}_{person_id:03d}_{img_n:03d}.jpeg',
                )
                cv2.imwrite(img_path, face_img)

            # assume only one face per image
            break


    results_count += 1
    if results_count == len(src_jpegs):
        break


    # save the result image
    # cv2.imwrite(result_img_path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))

source.send_shutdown(source_id, shutdown_auth)
print('Done.')



# class IndexBuilder(NvDsPyFuncPlugin):
#     """Index builder plugin."""

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.index = hnswlib.Index(space=self.index_space, dim=self.index_dim)
#         self.index.init_index(
#             max_elements=self.index_max_elements,
#             ef_construction=self.index_ef_construction,
#             M=self.index_m,
#         )
#         self.person_names = []

#         self.index_file_path = os.path.join(self.index_dir, 'index.bin')
#         self.processed_gallery_dir = os.path.join(self.index_dir, 'processed_gallery')

#         shutil.rmtree(self.processed_gallery_dir, ignore_errors=True)
#         os.makedirs(self.processed_gallery_dir)

#     def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
#         """Process frame metadata.

#         :param buffer: Gstreamer buffer with this frame's data.
#         :param frame_meta: This frame's metadata.
#         """
#         # image files source adapter adds location tag
#         # which contains image file name
#         location = frame_meta.get_tag('location')

#         if frame_meta.objects_number != 2:
#             # no faces detected and only primary object is present
#             # or more than 1 face detected
#             self.logger.warn(
#                 '%s faces detected on %s, 1 is expected. Not adding features to index.',
#                 frame_meta.objects_number - 1,
#                 location,
#             )
#         else:
#             stream = self.get_cuda_stream(frame_meta)
#             with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
#                 with Artist(frame_mat, stream) as artist:
#                     root, _ = os.path.splitext(location)
#                     file_name = os.path.basename(root)
#                     # gallery images are named as <person_name>_<img_n>.jpeg
#                     person_name, img_n = file_name.rsplit('_', maxsplit=1)
#                     # assign person_id to person_name in the order of appearance
#                     try:
#                         person_id = self.person_names.index(person_name)
#                     except ValueError:
#                         person_id = len(self.person_names)
#                         self.person_names.append(person_name)

#                     img_n = int(img_n)
#                     for obj_meta in frame_meta.objects:
#                         if obj_meta.label == 'face':
#                             # get face feature vector from metadata
#                             feature = obj_meta.get_attr_meta(
#                                 REID_MODEL_NAME, 'feature'
#                             ).value
#                             # index stores single int label for each feature
#                             # we assume that there are no more than 2^32 people or images for each person
#                             # and pack both person_id and img_n into single int
#                             feature_id = pack_person_id_img_n(person_id, img_n)
#                             self.index.add_items(feature, feature_id)

#                             # save face image to disk
#                             # it will be used to visualize face matches to gallery
#                             face_img = artist.copy_frame_region(
#                                 obj_meta.bbox
#                             ).download()
#                             face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2BGR)
#                             img_path = os.path.join(
#                                 self.processed_gallery_dir,
#                                 f'{person_name}_{person_id:03d}_{img_n:03d}.jpeg',
#                             )
#                             cv2.imwrite(img_path, face_img)
#             # refresh index on disk
#             self.write_index()

#     def write_index(self):
#         """Write index to disk."""
#         shutil.rmtree(self.index_file_path, ignore_errors=True)
#         os.makedirs(os.path.dirname(self.index_file_path), exist_ok=True)
#         self.index.save_index(self.index_file_path)
#         self.logger.info('Face processed, index file refreshed.')


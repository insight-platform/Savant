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
from savant.utils.logging import init_logging, get_logger

init_logging()
logger = get_logger('index_builder_client')

logger.info('Starting Savant client...')

try:
    zmq_src_socket = os.environ['ZMQ_SRC_ENDPOINT']
    zmq_sink_socket = os.environ['ZMQ_SINK_ENDPOINT']
except KeyError:
    logger.error('ZMQ_SRC_ENDPOINT and ZMQ_SINK_ENDPOINT env vars must be set.')
    sys.exit(1)


source_id = 'test-source'
shutdown_auth = 'shutdown'
index_dir = '/index'

processed_gallery_dir = os.path.join(index_dir, 'processed_gallery')
index_file_path = os.path.join(index_dir, 'index.bin')

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
    source(src_jpeg)

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

logger.info('Receiving results from the module...')

feature_namespace = 'adaface_ir50_webface4m_90fb74c'
feature_name = 'feature'

person_names = []
results_count = 0
for result in sink:

    if result.eos:
        continue

    # get location tag from metadata
    loc_attr = result.frame_meta.get_attribute('default', 'location')
    location = loc_attr.values[0].as_string()
    root, _ = os.path.splitext(location)
    file_name = os.path.basename(root)
    logger.info('Processing %s...', file_name)

    # get the face feature vector from metadata
    objs = result.frame_meta.get_all_objects()
    if len(objs) != 1:
        logger.warn('%s: expected 1 object, got %s, not adding any faces to index.', file_name, len(objs))
        continue

    # gallery images are named as <person_name>_<img_n>.jpeg
    person_name, img_n = file_name.rsplit('_', maxsplit=1)
    img_n = int(img_n)
    # assign person_id to person_name in the order of appearance
    try:
        person_id = person_names.index(person_name)
    except ValueError:
        person_id = len(person_names)
        person_names.append(person_name)

    img = np.frombuffer(result.frame_content, dtype=np.uint8)
    img = img.reshape(result.frame_meta.height, result.frame_meta.width, 4)

    for obj in objs:

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
        except IndexError:
            logger.error('%s: detection box out of image bounds.', file_name)
        else:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2BGR)
            img_path = os.path.join(
                processed_gallery_dir,
                f'{person_name}_{person_id:03d}_{img_n:03d}.jpeg',
            )
            cv2.imwrite(img_path, face_img)

    results_count += 1

    # don't wait for sink timeout if all images were processed
    if results_count == len(src_jpegs):
        logger.info('All images processed, stopping the module.')
        break

source.send_shutdown(source_id, shutdown_auth)

shutil.rmtree(index_file_path, ignore_errors=True)
os.makedirs(os.path.dirname(index_file_path), exist_ok=True)
index.save_index(index_file_path)
logger.info('Index build finished: %s faces processed and added.', results_count)

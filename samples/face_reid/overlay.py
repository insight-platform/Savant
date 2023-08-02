from itertools import chain
from collections import defaultdict
from typing import Tuple
import pathlib
import numpy as np
import cv2
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta, BBox
from savant.utils.artist import Position, Artist


FACE_SIZE = (112,112)
LEFT_OFFSET = 25
TOP_OFFSET = 25

GALLERY_PATH = '/opt/savant/samples/face_reid/processed_gallery'

class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unknown_face = np.ones(FACE_SIZE + (4,), dtype=np.uint8) * 255
        self.unknown_face = cv2.cuda.GpuMat(self.unknown_face)
        self.processed_gallery = {}
        for img_path in pathlib.Path(GALLERY_PATH).glob('*.jpeg'):
            person_name, person_id, img_n = img_path.stem.rsplit('_', maxsplit=2)
            person_id = int(person_id)
            img_n = int(img_n)

            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = cv2.resize(img, FACE_SIZE)
            img = cv2.cuda.GpuMat(img)
            self.processed_gallery[(person_id, img_n)] = img


    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        # manually refresh (by filling with black) frame padding used for drawing
        # this workaround avoids rendering problem where drawings from previous frames
        # are persisted on the padding area in the next frame
        frame_w, frame_h = artist.frame_wh
        artist.add_bbox(
            BBox(
                frame_w - self.padding_width // 2,
                frame_h // 2,
                self.padding_width,
                frame_h,
            ),
            border_width=0,
            bg_color=(0, 0, 0, 255),
        )

        # sort faces by person_id
        # so that they are displayed in the same order as defined in the gallery
        persons = []
        face_idx = 0
        for obj_meta in frame_meta.objects:
            if obj_meta.label == 'face':
                person_id = obj_meta.get_attr_meta('recognition', 'person_id').value
                persons.append((face_idx, person_id))
                face_idx += 1
        persons.sort(key=lambda x: x[1])
        face_rows = [0] * len(persons)
        for i, (face_idx, person_id) in enumerate(persons):
            face_rows[face_idx] = i

        face_idx = 0
        for obj_meta in frame_meta.objects:
            if obj_meta.label == 'face':
                person_id = obj_meta.get_attr_meta('recognition', 'person_id').value
                image_n = obj_meta.get_attr_meta('recognition', 'image_n').value

                if person_id > 0 and image_n > 0:
                    artist.add_bbox(
                        obj_meta.bbox,
                        border_color=(0, 255, 0, 255),
                    )
                else:
                    artist.add_bbox(
                        obj_meta.bbox,
                        border_color=(255, 0, 0, 255),
                    )

                face_img = artist.copy_frame_region(obj_meta.bbox)
                face_img = cv2.cuda.resize(face_img, FACE_SIZE, stream=artist.stream)

                tile_y = face_rows[face_idx]

                face_top = TOP_OFFSET * (tile_y + 1) + tile_y * FACE_SIZE[1]
                face_left = frame_w - self.padding_width + LEFT_OFFSET
                artist.add_graphic(face_img, (face_left, face_top))

                face_left += FACE_SIZE[0] + LEFT_OFFSET
                if person_id > 0 and image_n > 0:
                    gallery_img = self.processed_gallery[(person_id, image_n)]
                    artist.add_graphic(gallery_img, (face_left, face_top))
                else:
                    artist.add_graphic(self.unknown_face, (face_left, face_top))
                face_idx += 1

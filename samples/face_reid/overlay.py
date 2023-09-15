import pathlib

import cv2
import numpy as np

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import BBox, NvDsFrameMeta
from savant.meta.constants import UNTRACKED_OBJECT_ID
from savant.utils.artist import Artist


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.face_size = self.face_width, self.face_height
        self.unknown_face = np.zeros(self.face_size + (4,), dtype=np.uint8)
        cv2.drawMarker(
            self.unknown_face,
            (self.face_width // 2, self.face_height // 2),
            (255, 255, 255, 255),
            cv2.MARKER_TILTED_CROSS,
            75,
            3,
        )
        self.unknown_face = cv2.cuda.GpuMat(self.unknown_face)
        self.processed_gallery = {}
        self.person_id_to_name = {}
        self.prev_used_gallery = {}
        self.prev_used_counter = {}
        # load gallery images to display face matches
        for img_path in pathlib.Path(self.gallery_path).glob('*.jpeg'):
            person_name, person_id, img_n = img_path.stem.rsplit('_', maxsplit=2)
            person_id = int(person_id)
            img_n = int(img_n)
            if person_id not in self.person_id_to_name:
                self.person_id_to_name[person_id] = person_name
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = cv2.resize(img, self.face_size)
            img = cv2.cuda.GpuMat(img)
            self.processed_gallery[(person_id, img_n)] = img

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        # manually refresh (by filling with black) frame padding used for drawing
        # this workaround avoids rendering problem where drawings from previous frames
        # are persisted on the padding area in the next frame
        frame_w, frame_h = artist.frame_wh
        artist.add_bbox(
            BBox(
                frame_w - self.frame_padding_width // 2,
                frame_h // 2,
                self.frame_padding_width,
                frame_h,
            ),
            border_width=0,
            bg_color=(0, 0, 0, 255),
        )

        # sort faces by bbox left coordinate
        # so that their display order is consistent across frames
        persons = []
        face_idx = 0
        for obj_meta in frame_meta.objects:
            if obj_meta.label == 'face':
                persons.append((face_idx, obj_meta.bbox.left))
                face_idx += 1
        persons.sort(key=lambda x: x[1])
        face_rows = [0] * len(persons)
        for i, (face_idx, _) in enumerate(persons):
            face_rows[face_idx] = i

        face_idx = 0
        for obj_meta in frame_meta.objects:
            if obj_meta.label == 'face':
                person_id = obj_meta.get_attr_meta('recognition', 'person_id').value
                image_n = obj_meta.get_attr_meta('recognition', 'image_n').value
                distance = obj_meta.get_attr_meta('recognition', 'distance').value

                face_img = artist.copy_frame_region(obj_meta.bbox)
                face_img = cv2.cuda.resize(
                    face_img, self.face_size, stream=artist.stream
                )

                tile_y = face_rows[face_idx]

                face_top = (
                    self.face_tile_padding * (tile_y + 1) + tile_y * self.face_size[1]
                )
                face_left = frame_w - self.frame_padding_width + self.face_tile_padding
                artist.add_graphic(face_img, (face_left, face_top))

                face_left += self.face_size[0] + self.face_tile_padding
                if person_id >= 0 and image_n >= 0:
                    # person was recognized
                    # get a face image from the gallery
                    track_id = obj_meta.track_id
                    if track_id != UNTRACKED_OBJECT_ID:
                        # do not use the same image for less than `match_linger_frames` frames
                        if track_id not in self.prev_used_gallery:
                            # no previous image used for this track
                            self.prev_used_gallery[track_id] = (person_id, image_n)
                            self.prev_used_counter[track_id] = 1

                        elif self.prev_used_gallery[track_id] != (person_id, image_n):
                            # previous image used for this track is different
                            prev_person_id, prev_img_n = self.prev_used_gallery[
                                track_id
                            ]
                            prev_used_count = self.prev_used_counter[track_id]
                            if (
                                prev_person_id != person_id
                                or prev_used_count > self.match_linger_frames
                            ):
                                # if the track switched to a different person
                                # or the previous image was used for more than `match_linger_frames` frames
                                # use the new image
                                self.prev_used_gallery[track_id] = (person_id, image_n)
                                self.prev_used_counter[track_id] = 1
                            else:
                                # use the previous image
                                image_n = prev_img_n
                                person_id = prev_person_id
                                self.prev_used_counter[track_id] += 1

                    try:
                        face_img = self.processed_gallery[(person_id, image_n)]
                    except KeyError:
                        # gallery image not found
                        face_img = self.unknown_face
                    try:
                        person_name = self.person_id_to_name[person_id]
                    except KeyError:
                        # person name not found
                        person_name = 'no gallery'
                    text = f'{person_name} {distance:.2f}'
                    color = (0, 255, 0, 255)
                else:
                    face_img = self.unknown_face
                    text = 'Unknown'
                    color = (255, 0, 0, 255)

                artist.add_bbox(
                    obj_meta.bbox,
                    border_color=color,
                )
                artist.add_graphic(face_img, (face_left, face_top))
                artist.add_text(text, (face_left, face_top + self.face_size[1] + 10))

                face_idx += 1

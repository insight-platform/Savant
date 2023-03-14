import numpy as np
import cv2

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta, BBox
from savant.utils.artist import Artist
from savant.utils.artist import Position, Artist
from samples.peoplenet_detector.animation import Animation
from samples.peoplenet_detector.utils import load_sprite, get_font_scale
from samples.peoplenet_detector.person_face_matching import match_person_faces
from samples.peoplenet_detector.smoothed_counter import SmoothedCounter


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Colors are BGR
        self.face_bbox_border_color = (1, 0.5, 0.5)
        self.person_with_face_bbox_color = (0, 1, 0)
        self.person_no_face_bbox_color = (1, 0, 0)

        self.person_label_bg_color = (1, 0.9, 0.85)
        self.person_label_font_color = (0, 0, 0)

        self.bbox_border_width = 3
        self.overlay_height = 180
        self.logo_height = 120
        self.sprite_heigth = 120
        self.letter_height = 85
        self.counters_smoothing_period = 0.25

        self.font_thickness = 5
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = get_font_scale(
            self.letter_height, self.font_thickness, self.font_face
        )

        self.persons_with_face_counter = SmoothedCounter(self.counters_smoothing_period)
        self.persons_no_face_counter = SmoothedCounter(self.counters_smoothing_period)

        self.logo = load_sprite(
            '/opt/app/samples/peoplenet_detector/sprites/logo_insight.png',
            self.logo_height,
        )
        self.green_man = Animation(
            '/opt/app/samples/peoplenet_detector/sprites/green_man/',
            10,
            self.sprite_heigth,
        )
        self.blue_man = Animation(
            '/opt/app/samples/peoplenet_detector/sprites/blue_man/',
            10,
            self.sprite_heigth,
        )

        narrow_sep_w = 15
        sep_w = 60
        text_w = cv2.getTextSize(
            '99', self.font_face, self.font_scale, self.font_thickness
        )[0][0]
        logo_w = self.logo.size()[0]
        sprite_w = self.green_man.width

        cum_left = 0
        self.logo_pos = cum_left, (self.overlay_height - self.logo_height) // 2

        cum_left += logo_w + sep_w
        self.green_sprite_tl = cum_left, (self.overlay_height - self.sprite_heigth) // 2

        cum_left += sprite_w + narrow_sep_w
        self.green_text_tl = cum_left, (self.overlay_height - self.letter_height) // 2

        cum_left += text_w + sep_w
        self.blue_sprite_tl = cum_left, (self.overlay_height - self.sprite_heigth) // 2

        cum_left += sprite_w + narrow_sep_w
        self.blue_text_tl = cum_left, (self.overlay_height - self.letter_height) // 2

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """ """
        person_bboxes = []
        person_track_ids = []
        face_bboxes = []
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                continue
            if obj_meta.label == 'person':
                person_bboxes.append(obj_meta.bbox)
                person_track_ids.append(obj_meta.track_id)
            elif obj_meta.label == 'face':
                face_bboxes.append(obj_meta.bbox)
                artist.blur(obj_meta.bbox)

        if len(person_bboxes) > 0 and len(face_bboxes) > 0:
            person_with_face_idxs = match_person_faces(
                np.array([bbox.tlbr for bbox in person_bboxes]),
                np.array([bbox.tlbr for bbox in face_bboxes]),
            )
        else:
            person_with_face_idxs = []

        for i, (bbox, track_id) in enumerate(zip(person_bboxes, person_track_ids)):
            if i in person_with_face_idxs:
                color = self.person_with_face_bbox_color
            else:
                color = self.person_no_face_bbox_color
            artist.add_bbox(
                bbox=bbox,
                border_color=color,
                border_width=self.bbox_border_width,
                padding=0,
            )
            artist.add_text(
                text=f'#{track_id}',
                anchor_x=int(bbox.left) - self.bbox_border_width,
                anchor_y=int(bbox.top) - self.bbox_border_width,
                bg_color=self.person_label_bg_color,
                font_color=self.person_label_font_color,
                anchor_point=Position.LEFT_BOTTOM,
                padding=0,
            )

        pts = frame_meta.pts

        n_persons_with_face = self.persons_with_face_counter.get_value(
            pts, len(person_with_face_idxs)
        )
        n_persons_no_face = self.persons_no_face_counter.get_value(
            pts, len(person_bboxes) - len(person_with_face_idxs)
        )

        frame_w, _ = artist.frame_wh
        # manually refresh frame padding used for drawing
        # this workaround avoids rendering problem where drawings from previous frames
        # are persisted on the padding area in the next frame
        artist.add_bbox(
            BBox(
                x_center=frame_w // 2,
                y_center=self.overlay_height // 2,
                width=frame_w,
                height=self.overlay_height,
            ),
            border_width=0,
            bg_color=(0, 0, 0),
            padding=0,
        )
        artist.add_graphic(self.logo, self.logo_pos)
        artist.add_graphic(self.green_man.get_frame(pts), self.green_sprite_tl)
        artist.add_graphic(self.blue_man.get_frame(pts), self.blue_sprite_tl)
        artist.add_text(
            f'{n_persons_with_face}',
            self.green_text_tl[0],
            self.green_text_tl[1],
            self.font_scale,
            self.font_thickness,
            (1, 1, 1),
            padding=0,
            anchor_point=Position.LEFT_TOP,
        )
        artist.add_text(
            f'{n_persons_no_face}',
            self.blue_text_tl[0],
            self.blue_text_tl[1],
            self.font_scale,
            self.font_thickness,
            (1, 1, 1),
            padding=0,
            anchor_point=Position.LEFT_TOP,
        )

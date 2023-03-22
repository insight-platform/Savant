"""Overlay dashboard module."""
import cv2

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta, BBox
from savant.utils.artist import Position, Artist
from samples.peoplenet_detector.animation import Animation
from samples.peoplenet_detector.utils import load_sprite, get_font_scale


class Overlay(NvDsDrawFunc):
    """Overlay dashboard drawfunc.
    Uses previously attached analytics metadata to draw a dashboard with insight logo
    and animated sprites and counters for two types of detected persons
    (person with visible face, person with no visible face).

    :key person_with_face_bbox_color: the BGR color value for
        bboxes of persons with face associated.
    :key person_no_face_bbox_color: the BGR color value for
        bboxes of persons with no face associated.
    :key person_label_bg_color: the BGR color value for person label's background.
    :key person_label_font_color: the BGR color value for person label's font.
    :key bbox_border_width: the width of person's bboxes in pixels.
    :key overlay_height: the height of overlay dashboard in pixels.
        Keep in mind no to exceed module frame top padding value.
    :key logo_height: the height of logo image in pixels.
    :key sprite_height: the height of animation's sprites in pixels.
    :key counters_height: the height of counter's numbers in pixels.
    :key counters_font_thickness: the thickness of counter's font.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = get_font_scale(
            self.counters_height, self.counters_font_thickness, self.font_face
        )

        self.logo = load_sprite(
            '/opt/app/samples/peoplenet_detector/sprites/logo_insight.png',
            self.logo_height,
        )
        self.green_man = Animation(
            '/opt/app/samples/peoplenet_detector/sprites/green_man/',
            10,
            self.sprite_height,
        )
        self.blue_man = Animation(
            '/opt/app/samples/peoplenet_detector/sprites/blue_man/',
            10,
            self.sprite_height,
        )

        narrow_sep_w = 15
        sep_w = 60
        text_w = cv2.getTextSize(
            '99', self.font_face, self.font_scale, self.counters_font_thickness
        )[0][0]
        logo_w = self.logo.size()[0]
        sprite_w = self.green_man.width

        cum_left = 0
        self.logo_pos = cum_left, (self.overlay_height - self.logo_height) // 2

        cum_left += logo_w + sep_w
        self.green_sprite_tl = cum_left, (self.overlay_height - self.sprite_height) // 2

        cum_left += sprite_w + narrow_sep_w
        self.green_text_tl = cum_left, (self.overlay_height - self.counters_height) // 2

        cum_left += text_w + sep_w
        self.blue_sprite_tl = cum_left, (self.overlay_height - self.sprite_height) // 2

        cum_left += sprite_w + narrow_sep_w
        self.blue_text_tl = cum_left, (self.overlay_height - self.counters_height) // 2

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """DrawFunc's method where the drawing happens.
        Use artist's methods to add custom graphics to the frame.

        :param frame_meta: This frame's metadata.
        :artist: Artist object, provides high-level interface to drawing funcitons.
        """
        person_bboxes = []
        person_track_ids = []

        person_w_face_idxs = []
        n_persons_w_face = 0
        n_persons_no_face = 0

        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                person_w_face_idxs = obj_meta.get_attr_meta(
                    'analytics', 'person_w_face_idxs'
                ).value
                n_persons_w_face = obj_meta.get_attr_meta(
                    'analytics', 'n_persons_w_face'
                ).value
                n_persons_no_face = obj_meta.get_attr_meta(
                    'analytics', 'n_persons_no_face'
                ).value

            elif obj_meta.label == 'person':
                person_bboxes.append(obj_meta.bbox)
                person_track_ids.append(obj_meta.track_id)

            elif obj_meta.label == 'face':
                artist.blur(obj_meta.bbox)

        for i, (bbox, track_id) in enumerate(zip(person_bboxes, person_track_ids)):
            if i in person_w_face_idxs:
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

        frame_w, _ = artist.frame_wh
        # manually refresh (by filling with black) frame padding used for drawing
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
            f'{n_persons_w_face}',
            self.green_text_tl[0],
            self.green_text_tl[1],
            self.font_scale,
            self.counters_font_thickness,
            (1, 1, 1),
            padding=0,
            anchor_point=Position.LEFT_TOP,
        )
        artist.add_text(
            f'{n_persons_no_face}',
            self.blue_text_tl[0],
            self.blue_text_tl[1],
            self.font_scale,
            self.counters_font_thickness,
            (1, 1, 1),
            padding=0,
            anchor_point=Position.LEFT_TOP,
        )

import cv2

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta, BBox
from savant.utils.artist import Artist
from savant.utils.artist import Position, Artist
from samples.peoplenet_detector.animation import Animation
from samples.peoplenet_detector.utils import load_sprite, get_font_scale


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.face_bbox_border_color = (1, 0.5, 0.5)
        self.person_bbox_border_color = (0.75, 0.75, 0.5)
        self.person_label_bg_color = (1, 0.9, 0.85)
        self.person_label_font_color = (0, 0, 0)
        self.bbox_border_width = 3
        self.overlay_height = 180
        self.logo_height = 120
        self.sprite_heigth = 120
        self.letter_height = 85

        self.font_thickness = 5
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = get_font_scale(
            self.letter_height, self.font_thickness, self.font_face
        )

        text_w = cv2.getTextSize(
            '99', self.font_face, self.font_scale, self.font_thickness
        )[0][0]
        narrow_sep_w = 15
        sep_w = 60

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
        frame_w, _ = artist.frame_wh
        # manually refresh padding used for drawing
        # workaround, avoids rendering problem where drawings from previous frames
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
        artist.add_overlay(self.logo, self.logo_pos)
        artist.add_overlay(
            self.green_man.get_frame(frame_meta.pts), self.green_sprite_tl
        )
        artist.add_overlay(self.blue_man.get_frame(frame_meta.pts), self.blue_sprite_tl)
        artist.add_text(
            '7',
            self.green_text_tl[0],
            self.green_text_tl[1],
            self.font_scale,
            self.font_thickness,
            (1, 1, 1),
            padding=0,
            anchor_point=Position.LEFT_TOP,
        )
        artist.add_text(
            '12',
            self.blue_text_tl[0],
            self.blue_text_tl[1],
            self.font_scale,
            self.font_thickness,
            (1, 1, 1),
            padding=0,
            anchor_point=Position.LEFT_TOP,
        )

        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                continue

            if obj_meta.label == 'person':
                bbox_color = self.person_bbox_border_color
                artist.add_text(
                    text=f'#{obj_meta.track_id}',
                    anchor_x=int(obj_meta.bbox.left) - self.bbox_border_width,
                    anchor_y=int(obj_meta.bbox.top) - self.bbox_border_width,
                    bg_color=self.person_label_bg_color,
                    font_color=self.person_label_font_color,
                    anchor_point=Position.LEFT_BOTTOM,
                    padding=0,
                )
            else:
                bbox_color = self.face_bbox_border_color
                # artist.blur(obj_meta.bbox)

            artist.add_bbox(
                bbox=obj_meta.bbox,
                border_color=bbox_color,
                border_width=self.bbox_border_width,
                padding=0,
            )

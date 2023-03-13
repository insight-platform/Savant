import numpy as np
import cv2

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta, BBox
from savant.utils.artist import Artist
from savant.utils.artist import Position, Artist


def load_sprite(path:str, target_height:int) -> cv2.cuda.GpuMat:
    sprite = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    sprite = cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA)
    resize_coeff = target_height / sprite.shape[0]
    sprite = cv2.resize(sprite, dsize=None, fx=resize_coeff, fy=resize_coeff, interpolation=cv2.INTER_CUBIC)
    return cv2.cuda.GpuMat(sprite)


def get_font_scale(target_height_px: int, font_thickness:float=1, font_face:int = cv2.FONT_HERSHEY_SIMPLEX, sample_text='0123456789') -> float:
    font_scale = 0.5
    text_size, _ = cv2.getTextSize(
        sample_text, font_face, font_scale, font_thickness
    )
    min_delta = abs(target_height_px - text_size[1])
    for scale in np.linspace(0.6, 5, 45):
        text_size, baseline = cv2.getTextSize(
            sample_text, font_face, scale, font_thickness
        )
        delta = abs(target_height_px - text_size[1])
        if delta < min_delta:
            min_delta = delta
            font_scale = scale
            _ = baseline
    return font_scale


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
        self.font_scale = get_font_scale(self.letter_height, self.font_thickness, self.font_face)

        text_w = cv2.getTextSize(
            '99', self.font_face, self.font_scale, self.font_thickness
        )[0][0]
        narrow_sep_w = 15
        sep_w = 60

        self.logo = load_sprite('/opt/app/samples/peoplenet_detector/logo_insight.png', self.logo_height)
        self.green_sprite = load_sprite('/opt/app/samples/peoplenet_detector/animation/1/1_001.png', self.sprite_heigth)
        self.blue_sprite = load_sprite('/opt/app/samples/peoplenet_detector/animation/2/2_001.png', self.sprite_heigth)

        logo_w = self.logo.size()[0]
        sprite_w = self.green_sprite.size()[0]

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

        anim_fps = 10
        self.anim_frames_n = 10
        self.frame_period = 10**9 / anim_fps
        self.current_frame_i = 1
        self.ts = 0


    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """
        """
        # manually refresh padding used for drawing
        # workaround, avoids layered rendering problem because of padding memory being shared for all frames
        artist.add_bbox(
            BBox(x_center=640,y_center=90,width=1280,height=180),
            border_width=0,
            bg_color=(0,0,0),
            padding=0
        )
        artist.add_overlay(self.logo, self.logo_pos)
        artist.add_overlay(self.green_sprite, self.green_sprite_tl)
        artist.add_overlay(self.blue_sprite, self.blue_sprite_tl)
        artist.add_text(
            f'{self.current_frame_i}',
            self.green_text_tl[0],
            self.green_text_tl[1],
            self.font_scale,
            self.font_thickness,
            (1,1,1),
            padding=0,
            anchor_point=Position.LEFT_TOP
        )
        artist.add_text(
            f'{self.current_frame_i}',
            self.blue_text_tl[0],
            self.blue_text_tl[1],
            self.font_scale,
            self.font_thickness,
            (1,1,1),
            padding=0,
            anchor_point=Position.LEFT_TOP
        )
        if frame_meta.pts - self.ts >= self.frame_period:
            self.current_frame_i += 1
            if self.current_frame_i > self.anim_frames_n:
                self.current_frame_i = 1
            self.ts = frame_meta.pts

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

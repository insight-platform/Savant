import cv2

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.meta.bbox import BBox
from savant.utils.artist import Artist
from savant.meta.constants import UNTRACKED_OBJECT_ID
from savant.utils.artist import Position, Artist, COLOR


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
        
        logo = cv2.imread(
            '/opt/app/samples/peoplenet_detector/logo_insight.png',
            cv2.IMREAD_UNCHANGED,
        )
        logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2RGBA)
        logo_resize_coeff = self.logo_height / logo.shape[0]
        logo = cv2.resize(logo, dsize=None, fx=logo_resize_coeff, fy=logo_resize_coeff, interpolation=cv2.INTER_CUBIC)
        self.logo = cv2.cuda.GpuMat(logo)

        self.logo_top = (self.overlay_height - self.logo_height) // 2

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """
        """
        artist.add_overlay(self.logo, (0,self.logo_top))
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

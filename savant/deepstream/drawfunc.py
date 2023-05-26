"""Default implementation PyFunc for drawing on frame."""
from typing import Any, Dict, Optional, Tuple
import pyds
import cv2

from savant_rs.primitives import BoundingBoxDraw, ColorDraw, LabelDraw, DotDraw, PaddingDraw, ObjectDraw

from savant.meta.object import ObjectMeta
from savant.deepstream.base_drawfunc import BaseNvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.meta.bbox import BBox, RBBox
from savant.meta.constants import UNTRACKED_OBJECT_ID
from savant.utils.artist import Position, Artist, COLOR
from savant.gstreamer import Gst  # noqa: F401
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
import pprint

class NvDsDrawFunc(BaseNvDsDrawFunc):
    """Default implementation of PyFunc for drawing on frame.
    Uses OpenCV GpuMat to work with frame data without mapping to CPU
    through OpenCV-based Artist.

    PyFunc implementations are defined in and instantiated by a
    :py:class:`.PyFunc` structure.
    """

    def __init__(self, **kwargs):
        self.rendered_objects: Optional[Dict[str, Dict[str, Any]]] = None
        super().__init__(**kwargs)
        self.draw_spec = {}
        if self.rendered_objects:

            
            pprint.pprint(self.rendered_objects)
            for unit, objects in self.rendered_objects.items():
                # print(unit, objects)
                for obj, obj_draw_spec_cfg in objects.items():
                    self.draw_spec[(unit, obj)] = self._get_obj_draw_spec(obj_draw_spec_cfg)

                    # for key, value in draw_spec.items():
                    #     print(key, value)
                    #     if key == "color":
                    #         draw_spec[key] = COLOR[value]


            
        # 

        self.frame_streams = []

    def _get_obj_draw_spec(self, config:dict) -> ObjectDraw:
        bbox_draw = None

        if 'bbox' in config:
            blur = False
            if 'blur' in config['bbox']:
                blur = config['bbox']['blur']
    
            padding_draw = None
            if 'padding' in config['bbox']:
                padding_draw = PaddingDraw(**config['bbox']['padding'])

            bbox_draw = BoundingBoxDraw(
                color=ColorDraw(*self._convert_hex_to_rgb(config['bbox']['color'])),
                padding=padding_draw,
                thickness=config['bbox']['thickness'],
                blur=blur,
            )

        central_dot_draw = None
        if 'central_dot' in config:
            central_dot_draw = DotDraw(
                color=ColorDraw(*self._convert_hex_to_rgb(config['central_dot']['color'])),
                radius=config['central_dot']['radius'],
            )

        label_draw = None
        if 'label' in config:

            label_draw = LabelDraw(
                color = ColorDraw(*self._convert_hex_to_rgb(config['label']['color'])),
                font_scale=2.5,
                thickness=2,
                format=["{model}", "{label}", "{confidence}", "{track_id}"]
            )


        return ObjectDraw(bounding_box=bbox_draw, label=label_draw, central_dot=central_dot_draw)
    
    def _convert_hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int, int]:
        """Convert hex color to RGBA.

        :param hex_color: Hex color string
        :return: RGBA color tuple
        """
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4, 6))

    def __call__(self, nvds_frame_meta: pyds.NvDsFrameMeta, buffer: Gst.Buffer):
        with nvds_to_gpu_mat(buffer, nvds_frame_meta) as frame_mat:
            stream = cv2.cuda.Stream()
            self.frame_streams.append(stream)
            with Artist(frame_mat, stream) as artist:
                self.draw_on_frame(NvDsFrameMeta(nvds_frame_meta), artist)

    def finalize(self):
        """Finalize batch processing. Wait for all frame CUDA streams to finish."""
        for stream in self.frame_streams:
            stream.waitForCompletion()
        self.frame_streams = []

    def get_bbox_border_color(
        self, obj_meta: ObjectMeta
    ) -> Optional[Tuple[float, float, float]]:
        """Get object's bbox color.
        Draw only objects in rendered_objects if set.

        :param obj_meta: Object's meta
        :return: None, if there is no need to draw the object, otherwise color in BGR
        """
        if self.rendered_objects is None:
            return 0.0, 1.0, 0.0  # BGR
        # draw only rendered_objects if set
        if (
            obj_meta.element_name in self.rendered_objects
            and obj_meta.label in self.rendered_objects[obj_meta.element_name]
        ):
            return self.rendered_objects[obj_meta.element_name][obj_meta.label]

    def override_draw_spec(object_meta, specification: ObjectDraw) -> ObjectDraw:
        """Override draw specification for an object
        based on dynamically changning object properties.

        :param object_meta: Object's meta
        :param specification: Draw specification
        :return: Overridden draw specification
        """
        # make sure default draw spec is not modified
        return specification

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """Draws bounding boxes and labels for all objects in the frame's metadata.

        :param frame_meta: Frame metadata.
        :param artist: Artist to draw on the frame.
        """
        for obj_meta in frame_meta.objects:
            
            if obj_meta.is_primary:
                continue

            spec = self.override_draw_spec(obj_meta, self.draw_spec[(obj_meta.element_name, obj_meta.label)])

            # draw according to the specification

            if spec.bounding_box:
                self._draw_bounding_box(obj_meta, artist, spec.bounding_box)
            if spec.label:
                self._draw_label(obj_meta, artist, spec.label)
            if spec.central_dot:
                self._draw_central_dot(obj_meta, artist, spec.central_dot)

            # bbox_border_color = self.get_bbox_border_color(obj_meta)
            # if bbox_border_color:
            #     artist.add_bbox(
            #         bbox=obj_meta.bbox,
            #         border_color=bbox_border_color,
            #     )

            #     label = obj_meta.label
            #     if obj_meta.track_id != UNTRACKED_OBJECT_ID:
            #         label += f' #{obj_meta.track_id}'

            #     if isinstance(obj_meta.bbox, BBox):
            #         artist.add_text(
            #             text=label,
            #             anchor_x=int(obj_meta.bbox.left),
            #             anchor_y=int(obj_meta.bbox.top),
            #             bg_color=(0.0, 0.0, 0.0),
            #             anchor_point=Position.LEFT_TOP,
            #         )

            #     elif isinstance(obj_meta.bbox, RBBox):
            #         artist.add_text(
            #             text=label,
            #             anchor_x=int(obj_meta.bbox.x_center),
            #             anchor_y=int(obj_meta.bbox.y_center),
            #             bg_color=(0.0, 0.0, 0.0),
            #             anchor_point=Position.CENTER,
            #         )

    def _draw_bounding_box(obj_meta, artist: Artist, spec: BoundingBoxDraw):
        artist.add_bbox(
            bbox=obj_meta.bbox,
            border_color=bbox_border_color,
        )

    def _draw_label(obj_meta, artist: Artist, spec: LabelDraw):
        pass

    def _draw_central_dot(obj_meta, artist: Artist, spec: DotDraw):
        pass
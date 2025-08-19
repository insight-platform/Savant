from savant_rs.primitives.geometry import BBox

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.meta.object import ObjectMeta


class Downsampler(NvDsPyFuncPlugin):
    def __init__(
        self,
        sampling_fps: int,
        roi_left: int,
        roi_top: int,
        roi_width: int,
        roi_height: int,
        **kwargs,
    ):
        self.sampling_fps = sampling_fps
        self.roi_left = roi_left
        self.roi_top = roi_top
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.last_pts: dict[str, int] = {}
        super().__init__(**kwargs)

    def on_source_add(self, source_id: str):
        self.last_pts[source_id] = 0

    def on_source_eos(self, source_id: str):
        del self.last_pts[source_id]

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        source_id = frame_meta.source_id
        time_base = frame_meta.time_base
        current_ts = frame_meta.pts
        last_ts = self.last_pts[source_id]
        sampling_period = time_base[1] / time_base[0] / self.sampling_fps

        # when the condition is true, we need to add our custom ROI to the frame
        # to process only the marked frames (downsampled frames)
        time_condition = current_ts - last_ts > sampling_period

        # remove default ROI because we add our custom ROI to the frame
        # removal is not required but may help in case of tracker to reduce amount of tracked objects
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                frame_meta.remove_obj_meta(obj_meta)
                break

        if time_condition:
            self.last_pts[source_id] = current_ts
            custom_roi = BBox(
                self.roi_left + self.roi_width / 2,
                self.roi_top + self.roi_height / 2,
                self.roi_width,
                self.roi_height,
            )
            custom_roi = ObjectMeta(
                element_name='rate_limiter', label='roi', bbox=custom_roi
            )
            frame_meta.add_obj_meta(custom_roi)
            frame_meta.set_tag('visualize', True)

"""Adds a ROI object to a frame."""

from typing import Tuple

from savant_rs.primitives.geometry import BBox
from savant_rs.utils import eval_expr

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst  # noqa: F401
from savant.meta.object import ObjectMeta
from savant.parameter_storage import param_storage

ELEMENT_NAME = 'roi_injector'
LABEL = 'roi'
ROI_CACHE_TTL = 1000
MIN_ROI_SIZE = 32


class RoiInjector(NvDsPyFuncPlugin):
    """PyFunc implementing roi injector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_width = param_storage()['frame']['width']
        self.frame_height = param_storage()['frame']['height']
        self.default_roi = BBox.ltrb(
            *self._parse_ltrb(str(param_storage()['roi_default']))
        )
        self.per_source_roi = {}

    def _parse_ltrb(self, raw: str) -> Tuple[int, int, int, int]:
        """Parse the ROI string as (left, top, right, bottom)."""

        left, top, right, bottom = (int(v) for v in raw.split(','))
        if not (
            0 <= left < right <= self.frame_width
            and 0 <= top < bottom <= self.frame_height
        ):
            raise ValueError(
                f'ROI {raw} is out of frame bounds '
                f'({self.frame_width}x{self.frame_height}).'
            )
        if right - left < MIN_ROI_SIZE or bottom - top < MIN_ROI_SIZE:
            raise ValueError(
                f'ROI {raw} size must be at least {MIN_ROI_SIZE} in each dimension.'
            )

        return left, top, right, bottom

    def _read_roi(self, source_id: str) -> BBox:
        """Read the current ROI for the source from Etcd."""

        val, is_cached = eval_expr(
            f'etcd("roi/{source_id}", "")',
            ttl=ROI_CACHE_TTL,
            no_gil=True,
        )
        if not is_cached:
            if val:
                try:
                    self.per_source_roi[source_id] = BBox.ltrb(*self._parse_ltrb(val))
                except Exception as e:
                    self.logger.warning(
                        'Failed to parse ROI %r for source %s: %s.', val, source_id, e
                    )
                    self.per_source_roi.pop(source_id, None)
            else:
                self.per_source_roi.pop(source_id, None)

        return self.per_source_roi.get(source_id, self.default_roi)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Callback on each frame in a Deepstream pipeline batch."""

        obj_meta = ObjectMeta(
            element_name=ELEMENT_NAME,
            label=LABEL,
            bbox=self._read_roi(frame_meta.source_id),
        )
        frame_meta.add_obj_meta(object_meta=obj_meta)

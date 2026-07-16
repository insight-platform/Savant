"""Adds a ROI object to a frame."""

from savant_rs.primitives.geometry import BBox
from savant_rs.utils import eval_expr

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst  # noqa: F401
from savant.meta.object import ObjectMeta
from savant.parameter_storage import param_storage

ELEMENT_NAME = 'roi_injector'
LABEL = 'roi'
ROI_CACHE_TTL = 5


class RoiInjector(NvDsPyFuncPlugin):
    """PyFunc implementing roi injector."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_roi = self._parse_default_roi()
        self._roi_cache = {}

    def _parse_default_roi(self):
        default_roi_str = str(param_storage()['roi_default'])
        left, top, right, bottom = (float(v) for v in default_roi_str.split(','))
        return BBox.ltrb(left, top, right, bottom)

    def _parse_roi(self, raw: str, source_id: str) -> BBox:
        """Parse a ROI "left,top,right,bottom"."""

        if not raw:
            return self.default_roi
        try:
            left, top, right, bottom = (float(v) for v in raw.split(','))
            return BBox.ltrb(left, top, right, bottom)
        except Exception as e:
            self.logger.warning(
                'Failed to parse ROI %r for source %s: %s.', raw, source_id, e
            )
            return self.default_roi

    def _read_roi(self, source_id: str) -> BBox:
        """Read the current ROI for the source from Etcd."""
        val, is_cached = eval_expr(
            f'etcd("roi/{source_id}", "")',
            ttl=ROI_CACHE_TTL,
            no_gil=True,
        )
        if not is_cached or source_id not in self._roi_cache:
            self._roi_cache[source_id] = self._parse_roi(val, source_id)
        return self._roi_cache[source_id]

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Callback on each frame in a Deepstream pipeline batch."""
        obj_meta = ObjectMeta(
            element_name=ELEMENT_NAME,
            label=LABEL,
            bbox=self._read_roi(frame_meta.source_id),
        )
        frame_meta.add_obj_meta(object_meta=obj_meta)

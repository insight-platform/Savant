"""Ingress filter that removes the excessive ROI for this module instance."""

from savant.base.frame_filter import BaseFrameFilter
from savant_rs.match_query import MatchQuery, StringExpression

from samples.meta_merge.src.meta_merge.roi_constants import (
    LEFT_ROI_LABEL,
    RIGHT_ROI_LABEL,
    ROI_LEFT,
    ROI_NAMESPACE,
    ROI_RIGHT,
    VALID_ROIS,
)


class ROIIngressFilter(BaseFrameFilter):
    """Removes the ROI that this module instance does not process.

    Left module: removes router.right_roi.
    Right module: removes router.left_roi.
    """

    def __init__(self, roi: str, **kwargs):
        super().__init__(**kwargs)
        self.roi = roi.lower()
        if self.roi not in VALID_ROIS:
            raise ValueError(f"roi must be {ROI_LEFT!r} or {ROI_RIGHT!r}, got {roi!r}")

    def __call__(self, video_frame) -> bool:
        """Remove the other ROI and pass the frame."""
        remove_label = RIGHT_ROI_LABEL if self.roi == ROI_LEFT else LEFT_ROI_LABEL

        query = MatchQuery.and_(
            MatchQuery.namespace(StringExpression.eq(ROI_NAMESPACE)),
            MatchQuery.label(StringExpression.eq(remove_label)),
        )
        video_frame.delete_objects(query)
        return True

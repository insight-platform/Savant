"""Shared ROI constants for meta_merge sample (router, detector, meta_merge)."""

ROI_NAMESPACE = "router"
LEFT_ROI_LABEL = "left_roi"
RIGHT_ROI_LABEL = "right_roi"
ROI_LEFT = "left"
ROI_RIGHT = "right"
VALID_ROIS = (ROI_LEFT, ROI_RIGHT)

# Expected number of ingress streams (left and right ROI modules)
EXPECTED_INGRESS_COUNT = 2

"""Detector output converter that pulls per-source config from Etcd."""

import json
from typing import Optional, Tuple

import numpy as np
from savant_rs.utils import eval_expr

from savant.base.model import ObjectModel
from savant.converter.yolo import TensorToBBoxConverter
from savant.deepstream.meta.frame import NvDsFrameMeta

# how long a fetched Etcd value stays cached locally (seconds)
CONFIG_CACHE_TTL = 5


class EtcdConfigurableConverter(TensorToBBoxConverter):
    """YOLO bbox converter whose thresholds are overridden per source_id from Etcd."""

    def __init__(self, **kwargs):
        self._default_confidence_threshold = kwargs.get('confidence_threshold', 0.25)
        self._default_nms_iou_threshold = kwargs.get('nms_iou_threshold', 0.0)
        self._configs = {}
        super().__init__(**kwargs)

    def _load_source_config(self, source_id: str) -> dict:
        expr = f'etcd("source/{source_id}", "")'
        val, is_cached = eval_expr(expr, ttl=CONFIG_CACHE_TTL, no_gil=True)
        if not is_cached:
            if val:
                try:
                    parsed_config = json.loads(val)
                    self._configs[source_id] = (
                        parsed_config if isinstance(parsed_config, dict) else {}
                    )
                except json.JSONDecodeError:
                    self.logger.warning(
                        'Invalid JSON in Etcd config for source %s: %r', source_id, val
                    )
                    self._configs[source_id] = {}
            else:
                self._configs[source_id] = {}

        return self._configs.get(source_id, {})

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ObjectModel,
        roi: Tuple[float, float, float, float],
        metadata: Optional[NvDsFrameMeta] = None,
    ) -> Optional[np.ndarray]:
        """Converts detector output layer tensor to bbox tensor.

        :param output_layers: Output layer tensor
        :param model: Model definition, required parameters: input tensor shape,
            maintain_aspect_ratio
        :param roi: [left, top, width, height] of the rectangle
            on which the model infers
        :param metadata: Frame metadata.
        :return: BBox tensor, see the base converter.
        """

        config = {}
        if metadata is not None:
            config = self._load_source_config(metadata.source_id)
            self.logger.debug(
                'Source %s converter config: %s', metadata.source_id, config
            )

        # per-source override with fallback to construction-time defaults
        self.confidence_threshold = config.get(
            'confidence_threshold', self._default_confidence_threshold
        )
        self.nms_iou_threshold = config.get(
            'nms_iou_threshold', self._default_nms_iou_threshold
        )

        # reuse the parent's YOLO tensor decoding / NMS / coordinate transform
        return super().__call__(*output_layers, model=model, roi=roi)

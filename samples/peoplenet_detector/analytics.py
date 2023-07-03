"""Analytics module."""
from collections import defaultdict
import numpy as np
from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from samples.peoplenet_detector.smoothed_counter import SmoothedCounter
from samples.peoplenet_detector.person_face_matching import match_person_faces


class Analytics(NvDsPyFuncPlugin):
    """Analytics pyfunc.
    On each frame associates detected persons to detected faces, keeps track of
    source-specific amount of persons with and without a faces on a frame.
    Attaches these counters as meta information to be used in drawing overlay.

    :key counters_smoothing_period: float, smoothing period in seconds.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.person_counters = defaultdict(
            lambda: SmoothedCounter(self.counters_smoothing_period)
        )

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        del self.person_counters[(source_id, 'face')]
        del self.person_counters[(source_id, 'noface')]

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        person_bboxes = []
        face_bboxes = []
        primary_meta_object = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta
            if obj_meta.label == 'person':
                person_bboxes.append(obj_meta.bbox)
            elif obj_meta.label == 'face':
                face_bboxes.append(obj_meta.bbox)

        if len(person_bboxes) > 0 and len(face_bboxes) > 0:
            person_w_face_idxs = match_person_faces(
                np.array([bbox.as_ltrb() for bbox in person_bboxes]),
                np.array([bbox.as_ltrb() for bbox in face_bboxes]),
            )
        else:
            person_w_face_idxs = []

        # assign draw labels to persons based on whether a face was matched to them
        # these draw labels are used in the config to define different draw specs
        # for objects of the same class
        person_idx = 0
        for obj_meta in frame_meta.objects:
            if obj_meta.label == 'person':
                if person_idx in person_w_face_idxs:
                    obj_meta.draw_label = 'person_face'
                else:
                    obj_meta.draw_label = 'person_noface'
                person_idx += 1

        pts = frame_meta.pts
        src_id = frame_meta.source_id

        n_persons_w_face = self.person_counters[(src_id, 'face')].get_value(
            pts, len(person_w_face_idxs)
        )
        n_persons_no_face = self.person_counters[(src_id, 'noface')].get_value(
            pts, len(person_bboxes) - len(person_w_face_idxs)
        )

        # there can be no objects due to the track validation period
        # (the tracker removes all objects it considers invalid),
        # see tracker config probationAge parameter
        if primary_meta_object:
            primary_meta_object.add_attr_meta(
                'analytics', 'n_persons_w_face', n_persons_w_face
            )
            primary_meta_object.add_attr_meta(
                'analytics', 'n_persons_no_face', n_persons_no_face
            )

from collections import defaultdict
import numpy as np
from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from samples.peoplenet_detector.smoothed_counter import SmoothedCounter
from samples.peoplenet_detector.person_face_matching import match_person_faces


class Analytics(NvDsPyFuncPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.person_counters = defaultdict(lambda:SmoothedCounter(self.counters_smoothing_period))

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        del self.person_counters[(source_id, 'face')]
        del self.person_counters[(source_id, 'noface')]

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        person_bboxes = []
        face_bboxes = []
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                continue
            if obj_meta.label == 'person':
                person_bboxes.append(obj_meta.bbox)
            elif obj_meta.label == 'face':
                face_bboxes.append(obj_meta.bbox)

        if len(person_bboxes) > 0 and len(face_bboxes) > 0:
            person_w_face_idxs = match_person_faces(
                np.array([bbox.tlbr for bbox in person_bboxes]),
                np.array([bbox.tlbr for bbox in face_bboxes]),
            )
        else:
            person_w_face_idxs = []

        pts = frame_meta.pts
        src_id = frame_meta.source_id

        n_persons_w_face = self.person_counters[(src_id, 'face')].get_value(
            pts, len(person_w_face_idxs)
        )
        n_persons_no_face = self.person_counters[(src_id, 'noface')].get_value(
            pts, len(person_bboxes) - len(person_w_face_idxs)
        )

        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                obj_meta.add_attr_meta('analytics', 'person_w_face_idxs', person_w_face_idxs)
                obj_meta.add_attr_meta('analytics', 'n_persons_w_face', n_persons_w_face)
                obj_meta.add_attr_meta('analytics', 'n_persons_no_face', n_persons_no_face)
                break

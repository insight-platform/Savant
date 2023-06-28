"""Analytics module."""
from typing import Dict, Tuple

from samples.age_gender_recognition.smoothed_counter import SmoothedCounter
from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.parameter_storage import param_storage

MODEL_NAME = param_storage()['detection_model_name']


class AgeGenderSmoothing(NvDsPyFuncPlugin):
    """Age and gender smoothing pyfunc.
    On each frame

    :key history_length: int, Length of the vector of historical values
    """

    def __init__(self, history_length=24, **kwargs):
        super().__init__(**kwargs)
        self.history_length = history_length
        self.smoother: Dict[Tuple[str, str]] = {}

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        del self.smoother[(source_id, 'age')]
        del self.smoother[(source_id, 'gender')]

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        if (frame_meta.source_id, 'age') not in self.smoother:
            self.smoother[(frame_meta.source_id, 'age')] = SmoothedCounter(
                history_len=self.history_length,
                smoothed_type='mean',
            )
        if (frame_meta.source_id, 'gender') not in self.smoother:
            self.smoother[(frame_meta.source_id, 'gender')] = SmoothedCounter(
                history_len=self.history_length,
                smoothed_type='vote',
            )

        for obj_meta in frame_meta.objects:
            if obj_meta.element_name == MODEL_NAME:
                age = obj_meta.get_attr_meta('age_gender', 'age').value
                gender = obj_meta.get_attr_meta('age_gender', 'gender').value
                new_age = self.smoother[(frame_meta.source_id, 'age')].get_value(
                    new_value=age, frame_num=frame_meta.frame_num, key=obj_meta.track_id
                )
                new_gender = self.smoother[(frame_meta.source_id, 'gender')].get_value(
                    new_value=gender,
                    frame_num=frame_meta.frame_num,
                    key=obj_meta.track_id,
                )
                obj_meta.add_attr_meta('smoothed_value', 'age', new_age)
                obj_meta.add_attr_meta('smoothed_value', 'gender', new_gender)

        if (frame_meta.frame_num + 1) % 100 == 0:
            self.smoother[(frame_meta.source_id, 'age')].clean(frame_meta.frame_num)
            self.smoother[(frame_meta.source_id, 'gender')].clean(frame_meta.frame_num)

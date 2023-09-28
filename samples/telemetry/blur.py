import cv2

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import apply_cuda_filter, nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst


class Blur(NvDsPyFuncPlugin):
    """Apply gaussian blur to the frame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC4, cv2.CV_8UC4, (31, 31), 0
        )

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        # the call of this method is wrapped with `process-frame` span
        # the span is available in frame_meta, `frame_meta.telemetry_span`

        # use attributes to enrich span with some useful information
        frame_meta.telemetry_span.set_int_attribute('frame_num', frame_meta.frame_num)

        # logger messages will be added to span automatically
        self.logger.info('Try to blur frame #%d.', frame_meta.frame_num)

        stream = self.get_cuda_stream(frame_meta)
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            # create a new span for an important code section
            # to track the time spent on its execution
            with frame_meta.telemetry_span.nested_span('blur-filter'):
                apply_cuda_filter(
                    self._gaussian_filter,
                    frame_mat,
                    frame_meta.roi.as_ltwh_int(),
                    stream,
                )

        # wrap the code with try/catch and get exceptions in span
        try:
            # some error code for example
            with frame_meta.telemetry_span.nested_span('error-code') as span:
                span.set_string_attribute('section', 'try division by zero')
                # raise Exception('Some exception.')
                _ = 2 / 0

        except:
            pass

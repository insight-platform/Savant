import logging
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst

logger = logging.getLogger(__name__)


class Dummy(NvDsPyFuncPlugin):

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """
        Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        pts = frame_meta.pts
        source_id = frame_meta.source_id
        try:
            _ = self.get_cuda_stream(frame_meta)

            with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
                # Extract objects and embeddings
                elements = set()
                for obj_meta in frame_meta.objects:
                    if obj_meta.element_name not in elements:
                        elements.add(obj_meta.element_name)
                if "peoplenet" not in elements:
                    print(f"No peoplenet element found in frame {pts} from source {source_id}")
        except Exception as e:
            logger.error(
                f"Source {source_id}, PTS {pts}: EXCEPTION in process_frame: {e}",
                exc_info=True,
            )

"""
gst-launch-1.0 videotestsrc num-buffers=100 ! nvvideoconvert ! pyfunc module=samples.test.pyfunc class=PyFunc ! queue leaky=upstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! pyfunc module=samples.test.pyfunc class=PyFunc ! fpsdisplaysink video-sink=fakesink sync=0 -v
"""
import random
import time

from savant.base.pyfunc import BasePyFuncPlugin
from savant.gstreamer import Gst  # noqa: F401


class PyFunc(BasePyFuncPlugin):
    def __init__(self, **kwargs):
        self.sleep = 0
        self.random = 0
        super().__init__(**kwargs)
        self._counter = 0

    def process_buffer(self, buffer: Gst.Buffer):
        if self.sleep:
            sleep = self.sleep
            if self.random:
                sleep = self.sleep * random.randint(1, self.random)
            time.sleep(sleep)
        self._counter += 1

    def on_start(self) -> bool:
        sleep = self.sleep
        if sleep:
            if self.random:
                sleep = [self.sleep, self.random * self.sleep]
            self.logger.info(f'{self.gst_element.name} will sleep {sleep} seconds.')
        return True

    def on_stop(self) -> bool:
        self.logger.info(f'{self.__class__.__name__} counter: {self._counter}')
        return True

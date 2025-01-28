import time

import requests
import savant_rs.webserver.kvs as kvs
from savant_rs.primitives import Attribute, AttributeValue

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst


class First(NvDsPyFuncPlugin):
    """Apply gaussian blur to the frame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._counter = 0
        self._subscription = kvs.KvsSubscription('events', 100)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        self._counter += 1
        now = time.time()
        attr = Attribute(
            namespace='counter',
            name='frame_counter',
            hint='This attribute is set on every frame change',
            values=[
                AttributeValue.integer(self._counter),
            ],
        )
        kvs.set_attributes([attr], ttl=None)

        events = [self._subscription.recv()]

        maybe_second = self._subscription.try_recv()
        if maybe_second is not None:
            events.append(maybe_second)

        if len(events) == 2:  # we have an event from the downstream pipeline
            for e in events:
                attributes = e.attributes
                # filter only attributes with namespace 'second'
                second_attributes = [a for a in attributes if a.namespace == 'second']
                if len(second_attributes) > 0:
                    elapsed_time = float(int((time.time() - now) * 100_000) / 100)
                    for a in second_attributes:
                        self.logger.info(
                            f'Downstream attribute value (second): {a.values[0].as_integer()}, Elapsed time: {elapsed_time} ms'
                        )


class Second(NvDsPyFuncPlugin):
    """Apply gaussian blur to the frame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._counter = 0

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        self._counter += 1
        if self._counter % 100 == 0:
            now = time.time()
            attr = Attribute(
                namespace='second',
                name='frame_counter',
                hint='This attribute is set on every 1000th frame processing',
                values=[
                    AttributeValue.integer(self._counter),
                ],
            )
            binary_attributes = kvs.serialize_attributes([attr])
            response = requests.post(
                f'http://first:8080/kvs/set', data=binary_attributes
            )
            assert response.status_code == 200
            elapsed = float(int((time.time() - now) * 100_000) / 100)
            self.logger.info(
                f'Sent event to the upstream (first) module. Elapsed time: {elapsed} ms'
            )

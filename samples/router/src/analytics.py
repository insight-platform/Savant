import asyncio
import os
import time

from savant_rs.logging import LogLevel, log
from savant_rs.match_query import MatchQuery as Q
from savant_rs.primitives import BorrowedVideoObject, VideoFrame
from savant_rs.zmq import NonBlockingReader, ReaderConfigBuilder, ReaderResultMessage

socket_url = os.getenv('ZMQ_SRC_ENDPOINT')


async def reader():
    reader_config = ReaderConfigBuilder(socket_url).build()
    reader = NonBlockingReader(reader_config, 1_000)
    reader.start()

    while True:
        m = reader.receive()
        if m is None:
            # await asyncio.sleep(0.001)
            continue
        else:
            if isinstance(m, ReaderResultMessage):
                m = m.message
                if m.is_video_frame():
                    frame: VideoFrame = m.as_video_frame()
                    created_at = frame.creation_timestamp_ns / 1e6
                    objects = frame.access_objects(Q.idle())
                    object_info = []
                    for obj in objects:
                        o: BorrowedVideoObject = obj
                        object_info.append((o.namespace, o.label, o.id))
                    object_info.sort()
                    processed_at = time.time() * 1000
                    log(
                        LogLevel.Info,
                        'analytics::time_delta',
                        f'{frame.source_id} {frame.pts} {int(processed_at - created_at)} {object_info}',
                    )
                    log(LogLevel.Info, 'analytics::frame::json', f'{frame.json}')


loop = asyncio.new_event_loop()
try:
    loop.run_until_complete(reader())
finally:
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()

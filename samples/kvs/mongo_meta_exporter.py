from queue import Empty, Queue
from threading import Thread
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from savant_rs.primitives import VideoFrame

from savant.api.parser import parse_video_frame
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst


class MongoMetaExporter(NvDsPyFuncPlugin):
    def __init__(
        self,
        uri: str,
        collection: str,
        db: str,
        queue_size: int = 1000,
        max_batch_size: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mongo_client = MongoClient(uri)
        self.db_name = db
        self.collection_name = collection
        self.max_batch_size = max_batch_size

        self.db: Optional[Database] = None
        self.collection: Optional[Collection] = None
        self.thread: Optional[Thread] = None
        self.is_running = False
        self.queue = Queue(maxsize=queue_size)

    def on_start(self) -> bool:
        super().on_start()
        self.logger.info('Connecting to MongoDB')
        self.db: Database = self.mongo_client[self.db_name]
        self.collection: Collection = self.db[self.collection_name]
        self.thread = Thread(target=self.thread_workload)
        self.is_running = True
        self.thread.start()
        self.logger.info('Started MongoDB exporter thread')

        return True

    def on_stop(self) -> bool:
        super().on_stop()
        self.logger.info('Stopping MongoDB exporter thread')
        self.is_running = False
        self.thread.join()
        self.logger.info('Closing MongoDB connection')
        self.mongo_client.close()

        return True

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        doc = frame_to_document(frame_meta.video_frame)
        self.logger.debug(
            'Inserting document for frame %s/%s into queue',
            frame_meta.video_frame.source_id,
            frame_meta.video_frame.pts,
        )
        self.queue.put(doc)

    def thread_workload(self):
        while self.is_running or not self.queue.empty():
            batch = []
            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self.queue.get(timeout=1))
                except Empty:
                    break

            if not batch:
                continue

            while True:
                try:
                    self.logger.info(
                        'Inserting batch of %s documents into MongoDB',
                        len(batch),
                    )
                    self.collection.insert_many(batch)
                    break
                except Exception as e:
                    self.logger.error('Error inserting batch: %s', e)


def frame_to_document(frame: VideoFrame):
    doc = parse_video_frame(frame)
    doc['uuid'] = frame.uuid

    return doc

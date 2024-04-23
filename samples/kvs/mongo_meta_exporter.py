from datetime import datetime, timedelta
from queue import Empty, Queue
from threading import Thread
from typing import Optional

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from savant_rs.primitives.geometry import RBBox

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.meta.constants import UNTRACKED_OBJECT_ID
from savant.meta.object import ObjectMeta


class MongoMetaExporter(NvDsPyFuncPlugin):
    def __init__(
        self,
        uri: str,
        collection: str,
        db: str,
        queue_size: int = 1000,
        max_batch_size: int = 25,
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
        self.db.list_collections()  # Check if connection is working

        self.collection: Collection = self.db[self.collection_name]
        if 'source_id_uuid_index' not in self.collection.index_information():
            self.logger.info('Creating index for source_id and uuid')
            self.collection.create_index(
                [('source_id', ASCENDING), ('uuid', ASCENDING)],
                unique=True,
                name='source_id_uuid_index',
            )

        if 'ttl_index' not in self.collection.index_information():
            self.logger.info('Creating TTL index')
            self.collection.create_index(
                'created_at',
                expireAfterSeconds=int(timedelta(days=1).total_seconds()),
                name='ttl_index',
            )

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
        doc = frame_to_document(frame_meta)
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


def frame_to_document(frame_meta: NvDsFrameMeta):
    doc = {
        'source_id': frame_meta.video_frame.source_id,
        'uuid': frame_meta.video_frame.uuid,
        'pts': frame_meta.video_frame.pts,
        'objects': [
            object_to_document(obj) for obj in frame_meta.objects if not obj.is_primary
        ],
        'created_at': datetime.now(),
    }

    return doc


def object_to_document(obj: ObjectMeta):
    return {
        'uid': int(obj.uid),
        'element_name': obj.element_name,
        'label': obj.label,
        'track_id': obj.track_id if obj.track_id != UNTRACKED_OBJECT_ID else None,
        'confidence': float(obj.confidence),
        'bbox': {
            'type': 'rbbox' if isinstance(obj.bbox, RBBox) else 'bbox',
            'xc': float(obj.bbox.xc),
            'yc': float(obj.bbox.yc),
            'width': float(obj.bbox.width),
            'height': float(obj.bbox.height),
            'angle': float(obj.bbox.angle) if isinstance(obj.bbox, RBBox) else None,
        },
    }

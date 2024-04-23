from collections import deque
from typing import Dict, Optional, Tuple

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from savant_rs.primitives.geometry import BBox, RBBox

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.meta.constants import UNTRACKED_OBJECT_ID
from savant.meta.object import ObjectMeta


class MongoMetaImporter(NvDsPyFuncPlugin):
    def __init__(
        self,
        uri: str,
        collection: str,
        db: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mongo_client = MongoClient(uri)
        self.db_name = db
        self.collection_name = collection

        self.db: Optional[Database] = None
        self.collection: Optional[Collection] = None
        self.last_fragment_uuids: Dict[str, Tuple[str, str]] = {}
        self.pending_meta: Dict[str, deque[Dict]] = {}

    def on_start(self) -> bool:
        super().on_start()
        self.logger.info('Connecting to MongoDB')
        self.db: Database = self.mongo_client[self.db_name]
        self.collection: Collection = self.db[self.collection_name]
        self.logger.info('Started MongoDB importer')

        return True

    def on_stop(self) -> bool:
        super().on_stop()
        self.logger.info('Closing MongoDB connection')
        self.mongo_client.close()

        return True

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        self.load_fragment_meta(frame_meta)
        pending_meta = self.pending_meta.get(frame_meta.source_id)
        if not pending_meta:
            return
        next_meta = pending_meta.popleft()
        for obj in next_meta['objects']:
            if obj['bbox']['angle'] is not None:
                bbox = RBBox(
                    xc=obj['bbox']['xc'],
                    yc=obj['bbox']['yc'],
                    width=obj['bbox']['width'],
                    height=obj['bbox']['height'],
                    angle=obj['bbox']['angle'],
                )
            else:
                bbox = BBox(
                    xc=obj['bbox']['xc'],
                    yc=obj['bbox']['yc'],
                    width=obj['bbox']['width'],
                    height=obj['bbox']['height'],
                )
            track_id = obj['track_id']
            obj_meta = ObjectMeta(
                element_name=obj['element_name'],
                label=obj['label'],
                bbox=bbox,
                confidence=obj['confidence'],
                track_id=track_id if track_id is not None else UNTRACKED_OBJECT_ID,
            )
            frame_meta.add_obj_meta(obj_meta)

    def load_fragment_meta(self, frame_meta: NvDsFrameMeta):
        first_frame_uuid = frame_meta.get_tag('first-frame-uuid')
        last_frame_uuid = frame_meta.get_tag('last-frame-uuid')
        if first_frame_uuid is None or last_frame_uuid is None:
            return
        current_uuids = (first_frame_uuid, last_frame_uuid)
        if self.last_fragment_uuids.get(frame_meta.source_id) == current_uuids:
            return
        self.logger.info('New fragment: %s', current_uuids)
        self.last_fragment_uuids[frame_meta.source_id] = current_uuids
        query = {
            'source_id': frame_meta.source_id,
            'uuid': {'$gte': first_frame_uuid, '$lte': last_frame_uuid},
        }
        docs = list(self.collection.find(query))
        self.logger.info('Found %d documents', len(docs))
        self.pending_meta[frame_meta.source_id] = deque(docs)

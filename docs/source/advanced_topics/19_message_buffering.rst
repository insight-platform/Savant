Message Buffering
-----------------

Savant provides a special service (Buffer NG) for reliable message buffering. This service can be deployed as an intermediate unit between upstream and downstream elements, persistently storing messages on disk using RocksDB to prevent data loss during downstream outages, network interruptions, or temporary capacity shortages. Buffer NG supports Python-based extensibility for custom message processing logic before buffering and after retrieval, configurable buffer capacity with high watermark monitoring, and comprehensive telemetry for monitoring buffer utilization and message flow.

Buffer NG service documentation is available on `GitHub Pages <https://insight-platform.github.io/savant-rs/services/buffer_ng/index.html>`__.

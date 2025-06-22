Restreaming and Replaying
-------------------------

Savant provides a special service (Replay) for restreaming and replaying video streams. This service can be deployed as an intermediate unit bypassing the traffic from upstream to downstream elements and keeping the required period of last seconds in the RocksDB database. Replay clients use REST API to initiate re-streaming jobs specifying job parameters and destination.

Replay service documentation is available on `GitHub Pages <https://insight-platform.github.io/savant-rs/services/replay/index.html>`__.
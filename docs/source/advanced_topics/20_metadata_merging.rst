Metadata Merging
----------------

Savant provides a special service (Meta Merge) for merging metadata from multiple parallel processing pipelines back into a single stream. When the same video frame is processed by several parallel pipelines (e.g. detection, classification, pose estimation), each pipeline attaches its own metadata and attributes to the frame. Meta Merge collects these partial results, merges them into a single consolidated frame using developer-defined Python callbacks, and forwards the result downstream in the correct temporal order.

The service supports configurable expiration so that frames are forwarded even if not all pipelines complete in time, handles late-arriving frames via a dedicated callback, and provides per-ingress EOS (end-of-stream) management with allow/deny policies.

Meta Merge service documentation is available on `GitHub Pages <https://insight-platform.github.io/savant-rs/services/meta_merge/index.html>`__.

Stream Routing
--------------

Savant provides a special service (Router) for routing video streams to different destinations. This service can be deployed as an intermediate unit bypassing the traffic from upstream to downstream elements. It operates over multiple ingress streams and routes messages to multiple egress streams based on the assigned labels. The routing logic is implemented in Python and allows not only to route messages but also to modify them.

Router service documentation is available on `GitHub Pages <https://insight-platform.github.io/savant-rs/services/router/index.html>`__.
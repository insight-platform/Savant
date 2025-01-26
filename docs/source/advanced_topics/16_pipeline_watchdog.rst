Pipeline Watchdog
-----------------

Pipeline watchdog is a service which monitors the buffer adapter placed before the pipeline and restarts required
containers through the Docker API. The watchdog is implemented as a separate service and is not a part of the pipeline.

The watchdog can monitor various conditions and restart the containers based on the conditions. It uses assigned Docker
container labels to identify the containers to restart.

The watchdog can monitor the following conditions:

- queue length;
- egress idle time;
- ingress idle time.

The watchdog documentation is available on `GitHub <https://github.com/insight-platform/Savant/tree/develop/services/watchdog>`__.

The samples directory contains a `sample <https://github.com/insight-platform/Savant/tree/develop/samples/pipeline_watchdog>`__ showing how to use the watchdog service with the pipeline.

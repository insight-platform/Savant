# Watchdog

This service watches the health of pipeline by monitoring one or more buffers in parallel. It will stop or restart designated pipeline services if the buffer queue length exceeds a threshold value or the time since the last output or input message exceeds a specified time.

Queue monitoring helps detect the slow processing of messages, and ingress and egress monitoring is helpful in detecting how pipeline services are processing messages. In other words, the service can detect if the pipeline is not processing messages at the expected rate or if the pipeline is not processing messages at all.

The watchdog parses metrics in the [OpenMetrics](https://openmetrics.io/) text format exposed by the buffer service. When a metric has multiple labeled samples (e.g. different `reason` values), the maximum value across all samples is used by default. This can be customized with [label filters](#label-filters). Transient errors such as missing metrics or HTTP failures are logged and the current polling cycle is skipped without crashing.

## Configuration

The watchdog service is configured using the following environment variables:

* `CONFIG_FILE_PATH` - The path to the configuration file. Required.
* `LOGLEVEL` - The log level for the service. Default is `INFO`.

Configuration file is YAML file with the following structure:

```yaml
watch:
    - buffer: <str>
      queue:
        action: <restart|stop>
        length: <int>
        cooldown: <int>
        polling_interval: <int>
        container:
          - labels: [<str>]
          # other labels
        label_filters:  # optional
          <metric_name>:
            <label_key>: <label_value>
      egress:
        action: <restart|stop>
        idle: <int>
        cooldown: <int>
        polling_interval: <int>
        container:
          - labels: [<str>]
        label_filters:  # optional
          <metric_name>:
            <label_key>: <label_value>
      ingress:
        action: <restart|stop>
        idle: <int>
        cooldown: <int>
        polling_interval: <int>
        container:
          - labels: [<str>]
          # other labels
        label_filters:  # optional
          <metric_name>:
            <label_key>: <label_value>
      pyfunc:  # optional
        action: <restart|stop>
        cooldown: <int>
        polling_interval: <int>
        container:
          - labels: [<str>]
        module: <str>
        class_name: <str>
        kwargs:  # optional
          <key>: <value>
    # other buffers
```

Where:

* `buffer` - url of the buffer to watch.
* `queue` - configuration for the buffer queue. Optional.
  * `action` - action to take when the queue length exceeds the length threshold. It can be `restart` or `stop`.
  * `length` - threshold length for the queue.
  * `cooldown` - interval in seconds to wait after applying the action.
  * `polling_interval` - interval in seconds to check the queue length.
  * `container` - list of labels to match for the action. Actions are performed on containers that match any of the label sets.
    * `labels` - one or more labels to match on the same container, i.e. the container must have all labels.
  * `label_filters` - optional mapping of metric name to label key-value pairs used to select specific metric samples. See [Label filters](#label-filters) below.
* `ingress` or `egress` - configuration for the input or output traffic of the buffer. Optional.
  * `action` - action to take when the time since the last input or output message exceeds the idle threshold. It can be `restart` or `stop`.
  * `idle` - threshold time in seconds since the last input or output message.
  * `cooldown` - interval in seconds to wait after applying the action.
  * `polling_interval` - interval in seconds between buffer traffic checks. Optional. Default equals to `idle`.
  * `container` - list of labels to match for the action. Actions are performed on containers that match any of the label sets.
    * `labels` - one or more labels to match on the same container, i.e. the container must have all labels.
  * `label_filters` - optional mapping of metric name to label key-value pairs used to select specific metric samples. See [Label filters](#label-filters) below.
* `pyfunc` - configuration for a custom Python trigger. Optional.
  * `action` - action to take when the trigger returns `True`. It can be `restart` or `stop`.
  * `cooldown` - interval in seconds to wait after applying the action.
  * `polling_interval` - interval in seconds between trigger calls.
  * `container` - list of labels to match for the action. Actions are performed on containers that match any of the label sets.
    * `labels` - one or more labels to match on the same container, i.e. the container must have all labels.
  * `module` - Python module path to import (e.g. `watchdog.triggers.discrepancy`).
  * `class_name` - class name within the module. The class must be callable (implement `__call__`). It is instantiated once at startup with `buffer_url` (from the `buffer` field) and `kwargs`, and called on each polling cycle. Both sync and async callables are supported.
  * `kwargs` - optional additional keyword arguments passed to the class constructor.

**Note**: For each buffer, at least one of the `queue`, `ingress`, `egress`, or `pyfunc` sections must be present.

### Label filters

Buffer metrics may expose multiple samples for the same metric name, distinguished by labels (e.g. `last_sent_message{reason="send_success"}` and `last_sent_message{reason="ack_success"}`). By default, the watchdog aggregates all samples for a metric using `max()`. This works well in most cases, but when you need to watch a specific label variant you can use `label_filters`.

`label_filters` is a mapping where each key is a metric name and the value is a dictionary of label key-value pairs. Only samples whose labels match **all** specified pairs are considered. Metrics not mentioned in `label_filters` continue to use `max()` aggregation.

Example:

```yaml
egress:
  action: restart
  cooldown: 60s
  idle: 100s
  container:
    - labels: egress-client-label=egress-client-value
  label_filters:
    last_sent_message:
      reason: send_success
```

In this example, only the `last_sent_message` sample with `reason="send_success"` is used for the idle check; the `ack_success` sample is ignored.

You can find an example configuration file in the [samples](../../samples/pipeline_watchdog/config.yml) folder.

### Custom triggers (pyfunc)

The `pyfunc` watch type allows you to define arbitrary restart triggers as Python classes. The watchdog imports the class at startup, instantiates it with `buffer_url` (from the `buffer` field) and any additional `kwargs`, and calls it on each polling cycle. If the call returns `True`, the configured action is executed on the matched containers.

The trigger class contract:

```python
class MyTrigger:
    def __init__(self, buffer_url: str, **kwargs):
        # buffer_url is passed automatically from the watch config's `buffer` field.
        # Additional kwargs come from the config's `kwargs` section.
        ...

    async def __call__(self) -> bool:
        # Return True to trigger the action.
        # Both sync and async __call__ are supported.
        ...
```

A built-in example trigger is provided in `watchdog.triggers.discrepancy`. It detects stuck modules by checking whether egress is idle while ingress is still active — frames are going in but not coming out.

```yaml
watch:
  - buffer: ${oc.env:BUFFER_URL}
    pyfunc:
      action: restart
      cooldown: 600s
      polling_interval: 10s
      container:
        - labels: [com.savant.module=detector]
      module: watchdog.triggers.discrepancy
      class_name: DiscrepancyCheck
      kwargs:
        egress_idle: 60
        ingress_idle: 30
```

The `DiscrepancyCheck` trigger accepts (via `kwargs`):

* `egress_idle` - seconds of egress idle time before considering it stalled.
* `ingress_idle` - seconds within which ingress must have been active for the trigger to fire. If ingress is also idle, the problem is upstream (no input), not a stuck module.

### Interpolation

The configuration file supports variable interpolation. You can use a path to another node or environment variable in the configuration file by wrapping it in `${}`. For example:

* `${oc.env:BUFFER_URL}` - will be replaced with the value of the `BUFFER_URL` environment variable.
* `${.idle}` - will be replaced with the value of the `idle` key in the same section.

For more information, refer to the [OmegaConf documentation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation).

## Usage

You can find the watchdog service image on:

* for x86: [GitHub Packages](https://github.com/insight-platform/Savant/pkgs/container/savant-watchdog)
* for l4t: [GitHub Packages](https://github.com/insight-platform/Savant/pkgs/container/savant-watchdog-l4t)

Configuration of a docker service might be as follows:

```yaml
  pipeline-watchdog:
    image: ghcr.io/insight-platform/savant-watchdog-l4t:0.5.0
    restart: unless-stopped
    network_mode: host
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./config.yml:/app/config.yml
    environment:
      - LOGLEVEL=INFO
      - CONFIG_FILE_PATH=/app/config.yml
```


## Sample

The sample demonstrates how to start the watchdog service with an example pipeline to watch the buffer and restart the SDK client based on configuration and buffer state. The client randomly pauses message processing, triggering the watchdog to restart it when the egress idle threshold is exceeded.

### Run

This sample is designed to run on x86 architecture only.

```bash
docker compose -f samples/pipeline_watchdog/docker-compose.x86.yml up --build -d
```

### Check

After starting the pipeline, you can check the logs of the client container:

```bash
docker logs -f pipeline_watchdog-client-1
```

When the client stops processing messages for more than `egress.idle` seconds (see [config](../../samples/pipeline_watchdog/config.yml)) you will see the following logs in the client container, and the container itself will be restarted:

```
Traceback (most recent call last):
  File "/opt/savant/src/client.py", line 52, in <module>
    main()
  File "/opt/savant/src/client.py", line 37, in main
    time.sleep(sleep_duration)
KeyboardInterrupt
```

### Stop

```bash
docker compose -f samples/pipeline_watchdog/docker-compose.x86.yml down
```

## Tests

Run the tests from the watchdog service directory:

```bash
cd services/watchdog
pytest
```

Dependencies: `pytest`, `pytest-asyncio`, `aiohttp`, `aiodocker`, `prometheus_client`, `omegaconf`.

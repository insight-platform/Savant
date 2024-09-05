# Savant module template

This sample is intended to help users to start developing a custom Savant module. It provides:

* A [module](src/module/module.yml) with a simple pipeline that includes basic Savant elements and dev features.
* Scripts with [pyfunc](src/module/custom_pyfunc.py) and [drawfunc](src/module/overlay_draw_spec.py) templates.
* A [script](src/client/run.py) demonstrating basic Client SDK usage.
* Easy dev environment setup with either Docker Compose or devcontainer configuration files.
* Jaeger tracing platform service that allows to collect and inspect module's pipeline traces.
* Supporting Savant services such as Always On Sink adapter and Uri Input script that allow to send a video to the module and receive stream output.
* All of the above is set up to be ready to run, with no additional configuration needed.

See [documentation](https://insight-platform.github.io/Savant/) for more information.

## Development Quick Start

Instructions below assume current platform is x86 and current directory is `template`.

In case the `template` sample was copied into custom directory, `devcontainer.json` config will need to be updated. E.g. directory name is `my-module`, then

1. Update `--network` value in `runArgs`

```
"runArgs": [ "--gpus=all", "--network=my-module_network" ],
```

2. Update zmq sockets volume source in `mounts`

```
{
    "source": "my-module_zmq_sockets",
    "target": "/tmp/zmq-sockets",
    "type": "volume"
},
```

Subsections below describe alternative ways to realize the general development workflow:

1. Start jaeger and module processes
1. The client sends input image to the module and receives module results (image, metadata, logs)
1. Uri Input script + Always On sink allow to send a video file to be processed by the module and receive a stream output
1. Changes in pyfuncs, drawfunc, pre- and postprocessing code are loaded at runtime when module receives input from the client
1. Changes in module configuration require module container restart
1. Changes in module docker image, e.g. changes in `requirements.txt`, require module image rebuild

### First time start

Note that starting module for the first time involves downloading the model files and building TRT engine which may take several minutes. Additionally, client's wait for module status has a timeout which may be exceeded if TRT engine build process takes long enough. Wait for TRT build finish and restart the client script if so.

### Download sample video

URI-Input script demonstration requires a sample video.

```bash
# you are expected to be in Savant/samples/template/ directory

curl -o assets/test_data/elon_musk_perf.mp4 https://eu-central-1.linodeobjects.com/savant-data/demo/elon_musk_perf.mp4
```

### Docker Compose

1. Start Jaeger container

```bash
docker compose -f docker-compose.x86.yml up jaeger -d
```

This is required by the sample module since it is configured to use Jaeger telemetry.

Visit `http://127.0.0.1:16686` to access the Jaeger UI.

2. Start module container

```bash
docker compose -f docker-compose.x86.yml up module -d
```

3. Run and re-run the client script (result image written into `src/output` directory)

```bash
docker compose -f docker-compose.x86.yml up client
```

4. Send a video to the module and receive stream output

Start Always On Sink container

```bash
docker compose -f docker-compose.x86.yml up always-on-sink -d
```

Start Uri Input container

```bash
docker compose -f docker-compose.x86.yml up uri-input
```

Open `rtsp://127.0.0.1:554/stream/test` in your player, or visit `http://127.0.0.1:888/stream/test` in a browser.

4. Restart module container

E.g., in case [module.yml](src/module/module.yml) was modified.

```bash
docker compose -f docker-compose.x86.yml restart module
```

5. Rebuild module image (optionally, use `--no-cache` and `--pull` flags for docker build to force full rebuild)

E.g., in case [Dockerfile.x86](docker/Dockerfile.x86) or [requirements.txt](./requirements.txt) were modified.

```bash
docker compose -f docker-compose.x86.yml down module
docker compose -f docker-compose.x86.yml build module
```

### VS Code dev container

1. Open the module directory on host in IDE

File -> Open Folder -> enter path

2. Start Jaeger container

```bash
docker compose -f docker-compose.x86.yml up jaeger -d
```

This is required by the sample module since it is configured to use Jaeger telemetry.

Visit `http://127.0.0.1:16686` to access the Jaeger UI.

3. Reopen the directory in dev container

Command Palette (F1) -> "Dev Containers: Reopen in container" -> Select `devcontainer.json` appropriate for the platform

4. Start and restart module

```bash
python module/run.py
```

5. Run and re-run the client script (result image written into `/opt/savant/src/output` directory)

```bash
python client/run.py
```

6. Send a video to the module and receive stream output

Open the module directory on host, start Always On Sink container

```bash
docker compose -f docker-compose.x86.yml up always-on-sink -d
```

Run URI Input script inside the dev container

```bash
cd /opt/savant
python scripts/uri-input.py /test_data/elon_musk_perf.mp4 --socket pub+connect:ipc:///tmp/zmq-sockets/input-video.ipc --sync
```

Open `rtsp://127.0.0.1:554/stream/test` in your player, or visit `http://127.0.0.1:888/stream/test/` in a browser.

7. Rebuild module image

E.g., in case [Dockerfile.x86](docker/Dockerfile.x86) or [requirements.txt](./requirements.txt) were modified.

Command Palette (F1) -> "Dev Containers: Rebuild container"

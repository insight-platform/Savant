# Savant module template

Start your own project with this template.

See [documentation](https://insight-platform.github.io/Savant/) for more information.

## Development Quick Start

Instructions below assume current current platform is x86 and current directory is `template`.

Subsections below describe alternative ways to realize the general development workflow:

1. Start jaeger and module processes
2. The client sends input image to the module and receives module results (image, metadata, logs)
3. Changes in pyfuncs, drawfunc, pre- and postprocessing code are loaded at runtime when module receives input from the client
4. Changes in module configuration require module container restart
5. Changes in module image, e.g. changes in `requirements.txt`, require module image rebuild

### Download sample video

URI-Input script demonstration requires a sample video.

```
curl -o assets/test_data/elon_musk_perf.mp4 https://eu-central-1.linodeobjects.com/savant-data/demo/elon_musk_perf.mp4
```

### Docker Compose

1. Start Jaeger container

```bash
docker compose -f docker-compose.x86.yml up jaeger -d
```

2. Start module container

```bash
docker compose -f docker-compose.x86.yml up module -d
```

3. Run and re-run the client script (result image written into `src/output` directory)

```bash
docker compose -f docker-compose.x86.yml up client
```

4. Restart module container

```bash
docker compose -f docker-compose.x86.yml restart module
```

5. Rebuild module image (optionally, use `--no-cache` and `--pull` flags for docker build to force full rebuild)

```bash
docker compose -f docker-compose.x86.yml down module
docker compose -f docker-compose.x86.yml build module
```

### VS Code dev container

1. Start Jaeger container

```bash
docker compose -f docker-compose.x86.yml up jaeger -d
```

2. Reopen the directory in dev container

Command Palette (F1) -> "Dev Containers: Reopen in container" -> Select `devcontainer.json` appropriate for the platform

3. Start and restart module

```bash
python module/run.py
```

3. Run and re-run the client script (result image written into `/opt/savant/src/output` directory)

```bash
python client/run.py
```

5. Rebuild module image

Command Palette (F1) -> "Dev Containers: Rebuild container"

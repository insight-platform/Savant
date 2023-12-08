#!/usr/bin/env python3
"""Run sink adapter."""
import os
import uuid
from typing import Optional

import click
from common import (
    adapter_docker_image_option,
    build_common_envs,
    build_docker_run_command,
    build_zmq_endpoint_envs,
    fps_meter_options,
    run_command,
    source_id_option,
    validate_source_id,
    validate_source_id_list,
)


@click.group()
def cli():
    """Click command line group callback."""


def common_options(func):
    """Common Click sink adapter options."""
    func = click.option(
        '--in-endpoint',
        default='ipc:///tmp/zmq-sockets/output-video.ipc',
        help='Adapter input (module output) ZeroMQ socket endpoint.',
        show_default=True,
    )(func)
    func = click.option(
        '--in-type',
        default='SUB',
        help='Adapter input (module output) ZeroMQ socket type.',
        show_default=True,
    )(func)
    func = click.option(
        '--in-bind',
        default=False,
        help=(
            'Adapter input (module output) ZeroMQ socket bind/connect mode '
            '(bind if True).'
        ),
        show_default=True,
    )(func)
    return func


def source_id_prefix_option(func):
    return click.option(
        '--source-id-prefix',
        callback=validate_source_id,
        help='Filter frames by source ID prefix.',
    )(func)


def build_common_sink_envs(
    source_id: Optional[str],
    source_id_prefix: Optional[str],
    zmq_endpoint: str,
    zmq_type: Optional[str],
    zmq_bind: Optional[bool],
):
    """Generate env var run options."""
    envs = build_zmq_endpoint_envs(
        zmq_endpoint=zmq_endpoint,
        zmq_type=zmq_type,
        zmq_bind=zmq_bind,
    )
    if source_id:
        envs.append(f'SOURCE_ID={source_id}')
    if source_id_prefix:
        envs.append(f'SOURCE_ID_PREFIX={source_id_prefix}')
    return envs


skip_frames_without_objects_option = click.option(
    '--skip-frames-without-objects',
    is_flag=True,
    default=False,
    help='Skip frames without detected objects.',
)


def chunk_size_option(default=10000):
    """Chunk size option."""
    return click.option(
        '--chunk-size',
        default=default,
        help='Chunk size in frames.',
        show_default=True,
    )


@cli.command('display')
@click.option(
    '--closing-delay',
    type=click.INT,
    default=0,
    help='Delay in seconds before closing window when stream from source finished.',
    show_default=True,
)
@click.option(
    '--sync',
    is_flag=True,
    default=False,
    help='Show frames on sink synchronously (i.e. at the source file rate).',
)
@common_options
@source_id_option(required=False)
@source_id_prefix_option
@adapter_docker_image_option('deepstream')
def display_sink(
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    sync: bool,
    docker_image: str,
    closing_delay: int,
    source_id: Optional[str],
    source_id_prefix: Optional[str],
):
    """Show video on display, one window per source."""

    print('Setting environment variables for X')
    run_command(['xhost', '+local:docker'])
    xsock = '/tmp/.X11-unix'
    xauth = '/tmp/.docker.xauth'
    run_command(
        [
            'bash',
            '-c',
            (
                f'xauth nlist $DISPLAY | '
                "sed -e 's/^..../ffff/' | "
                f'xauth -f {xauth} nmerge -'
            ),
        ]
    )

    envs = build_common_sink_envs(
        source_id=source_id,
        source_id_prefix=source_id_prefix,
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
    ) + [
        'DISPLAY',
        f'XAUTHORITY={xauth}',
        f'CLOSING_DELAY={closing_delay}',
    ]

    cmd = build_docker_run_command(
        'sink-display',
        zmq_endpoints=[in_endpoint],
        sync=sync,
        entrypoint='/opt/savant/adapters/ds/sinks/display.sh',
        envs=envs,
        volumes=[f'{xsock}:{xsock}', f'{xauth}:{xauth}'],
        with_gpu=True,
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('meta-json')
@skip_frames_without_objects_option
@chunk_size_option(0)
@common_options
@source_id_option(required=False)
@source_id_prefix_option
@adapter_docker_image_option('py')
@click.argument('location', required=True)
def meta_json_sink(
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    docker_image: str,
    skip_frames_without_objects: bool,
    chunk_size: int,
    location: str,
    source_id: Optional[str],
    source_id_prefix: Optional[str],
):
    """Write metadata to streaming JSON files.

    LOCATION - location of the file to write. Can be plain location or pattern.
    (e.g. `/data/meta-%source_id-%src_filename`).
    Allowed substitution parameters: `%source_id`, `%src_filename`.
    """

    location = os.path.abspath(location)
    target_dir = os.path.dirname(location.split('%', 1)[0])

    envs = build_common_sink_envs(
        source_id=source_id,
        source_id_prefix=source_id_prefix,
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
    ) + [
        f'LOCATION={location}',
        f'SKIP_FRAMES_WITHOUT_OBJECTS={skip_frames_without_objects}',
        f'CHUNK_SIZE={chunk_size}',
    ]

    cmd = build_docker_run_command(
        f'sink-meta-json-{uuid.uuid4().hex}',
        zmq_endpoints=[in_endpoint],
        entrypoint='/opt/savant/adapters/python/sinks/metadata_json.py',
        envs=envs,
        volumes=[f'{target_dir}:{target_dir}'],
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('image-files')
@skip_frames_without_objects_option
@chunk_size_option()
@common_options
@source_id_option(required=False)
@source_id_prefix_option
@adapter_docker_image_option('py')
@click.argument('location', required=True)
def image_files_sink(
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    docker_image: str,
    skip_frames_without_objects: bool,
    chunk_size: int,
    location: str,
    source_id: Optional[str],
    source_id_prefix: Optional[str],
):
    """Write metadata to image files.

    LOCATION - target directory for the files to write.
    """

    location = os.path.abspath(location)
    target_dir = os.path.dirname(location.split('%', 1)[0])

    envs = build_common_sink_envs(
        source_id=source_id,
        source_id_prefix=source_id_prefix,
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
    ) + [
        f'DIR_LOCATION={location}',
        f'SKIP_FRAMES_WITHOUT_OBJECTS={skip_frames_without_objects}',
        f'CHUNK_SIZE={chunk_size}',
    ]

    cmd = build_docker_run_command(
        f'sink-image-files-{uuid.uuid4().hex}',
        zmq_endpoints=[in_endpoint],
        entrypoint='/opt/savant/adapters/python/sinks/image_files.py',
        envs=envs,
        volumes=[f'{target_dir}:{target_dir}'],
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('video-files')
@chunk_size_option()
@common_options
@source_id_option(required=False)
@source_id_prefix_option
@adapter_docker_image_option('gstreamer')
@click.argument('location', required=True)
def video_files_sink(
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    docker_image: str,
    chunk_size: int,
    location: str,
    source_id: Optional[str],
    source_id_prefix: Optional[str],
):
    """Write video to files at directory LOCATION."""

    location = os.path.abspath(location)

    envs = build_common_sink_envs(
        source_id=source_id,
        source_id_prefix=source_id_prefix,
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
    ) + [
        f'DIR_LOCATION={location}',
        f'CHUNK_SIZE={chunk_size}',
    ]

    cmd = build_docker_run_command(
        f'sink-video-files-{uuid.uuid4().hex}',
        zmq_endpoints=[in_endpoint],
        entrypoint='/opt/savant/adapters/gst/sinks/video_files.sh',
        envs=envs,
        volumes=[f'{location}:{location}'],
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('always-on-rtsp')
@click.option(
    '--source-id',
    callback=validate_source_id,
    help='Source ID, e.g. "camera1". The sink works in single-stream mode when this option is specified.',
)
@click.option(
    '--source-ids',
    callback=validate_source_id_list,
    help='Comma-separated source ID list, e.g. "camera1,camera2". The sink works in multi-stream mode when this option is specified.',
)
@click.option(
    '--stub-file-location',
    required=True,
    help='Location of the stub image file. Image file must be in JPEG format.',
)
@click.option(
    '--max-delay-ms',
    type=click.INT,
    default=1000,
    help='Maximum delay for the last frame in milliseconds.',
    show_default=True,
)
@click.option(
    '--transfer-mode',
    default='scale-to-fit',
    help='Transfer mode. One of: "scale-to-fit", "crop-to-fit".',
    show_default=True,
)
@click.option(
    '--protocols',
    default='tcp',
    help='Allowed lower transport protocols, e.g. "tcp+udp-mcast+udp".',
    show_default=True,
)
@click.option(
    '--latency-ms',
    type=click.INT,
    default=100,
    help='Amount of ms to buffer RTSP stream.',
    show_default=True,
)
@click.option(
    '--keep-alive',
    type=click.BOOL,
    default=True,
    help='Send RTSP keep alive packets, disable for old incompatible server.',
    show_default=True,
)
@click.option(
    '--profile',
    default='High',
    help='H264 encoding profile. One of: "Baseline", "Main", "High".',
    show_default=True,
)
@click.option(
    '--bitrate',
    type=click.INT,
    default=4000000,
    help='H264 encoding bitrate.',
    show_default=True,
)
@click.option(
    '--framerate',
    default='30/1',
    help='Frame rate of the output stream.',
    show_default=True,
)
@click.option(
    '--metadata-output',
    help='Where to dump metadata (stdout or logger).',
)
@click.option(
    '--sync',
    is_flag=True,
    default=False,
    help=(
        'Show frames on sink synchronously (i.e. at the source file rate). '
        'Note: inbound stream is not stable with this flag, try to avoid it.'
    ),
)
@click.option(
    '--dev-mode',
    default=False,
    is_flag=True,
    help='Use embedded MediaMTX to publish RTSP stream.',
)
@click.option(
    '--publish-ports',
    default=False,
    is_flag=True,
    help=(
        'Publish container ports for embedded MediaMTX to the host. '
        'Published ports: 554 (RTSP), 1935 (RTMP), 888 (HLS), 8889 (WebRTC). '
        'Ignored when --dev-mode is not set.'
    ),
)
@click.option(
    '--cpu',
    default=False,
    is_flag=True,
    help='Use CPU for transcoding and scaling.',
)
@fps_meter_options
@common_options
@adapter_docker_image_option('deepstream')
@click.argument('rtsp_uri', required=False)
def always_on_rtsp_sink(
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    docker_image: str,
    source_id: Optional[str],
    source_ids: Optional[str],
    stub_file_location: str,
    max_delay_ms: int,
    transfer_mode: str,
    protocols: str,
    latency_ms: int,
    keep_alive: bool,
    profile: str,
    bitrate: int,
    framerate: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: Optional[str],
    metadata_output: Optional[str],
    sync: bool,
    dev_mode: bool,
    publish_ports: bool,
    cpu: bool,
    rtsp_uri: Optional[str],
):
    """Send video stream from specific source to RTSP server.

    RTSP_URI - URI of the RTSP server. The sink sends video stream to RTSP_URI
    in single-stream mode and to RTSP_URI/{source-id} in multi-stream mode.
    Exactly one of --dev-mode flag and RTSP_URI argument must be used.

    When --dev-mode flag is used the stream is available at:

        - RTSP: rtsp://<container-host>:554/stream (single-stream),
        rtsp://<container-host>:554/stream/{source-id} (multi-stream)

        - RTMP: rtmp://<container-host>:1935/stream (single-stream),
        rtmp://<container-host>:1935/stream/{source-id} (multi-stream)

        - HLS: http://<container-host>:888/stream (single-stream),
        http://<container-host>:888/stream/{source-id} (multi-stream)

        - WebRTC: http://<container-host>:8889/stream (single-stream),
        http://<container-host>:8889/stream/{source-id} (multi-stream)


    Note: it is advisable to use --sync flag on source adapter or use a live
    source adapter (e.g. rtsp or usb-cam).
    """

    assert (source_id is None) != (
        source_ids is None
    ), 'Must be specified one of "--source-id" flag or "--source-ids" argument.'
    assert os.path.exists(stub_file_location)
    assert dev_mode == (
        rtsp_uri is None
    ), 'Must be specified one of "--dev-mode" flag or "RTSP_URI" argument.'
    stub_file_location = os.path.abspath(stub_file_location)

    envs = build_common_envs(
        source_id=source_id,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
    ) + [
        f'STUB_FILE_LOCATION={stub_file_location}',
        f'MAX_DELAY_MS={max_delay_ms}',
        f'TRANSFER_MODE={transfer_mode}',
        f'RTSP_PROTOCOLS={protocols}',
        f'RTSP_LATENCY_MS={latency_ms}',
        f'RTSP_KEEP_ALIVE={keep_alive}',
        f'ENCODER_PROFILE={profile}',
        f'ENCODER_BITRATE={bitrate}',
        f'FRAMERATE={framerate}',
    ]
    if source_ids:
        envs.append(f'SOURCE_IDS={source_ids}')
    if metadata_output:
        envs.append(f'METADATA_OUTPUT={metadata_output}')
    if dev_mode:
        envs.append(f'DEV_MODE={dev_mode}')
    else:
        envs.append(f'RTSP_URI={rtsp_uri}')

    if publish_ports:
        ports = [(x, x) for x in [554, 1935, 888, 8889]]
    else:
        ports = None

    cmd = build_docker_run_command(
        f'sink-always-on-rtsp-{uuid.uuid4().hex}',
        zmq_endpoints=[in_endpoint],
        sync=sync,
        entrypoint='python',
        args=['-m', 'adapters.ds.sinks.always_on_rtsp'],
        envs=envs,
        volumes=[f'{stub_file_location}:{stub_file_location}:ro'],
        with_gpu=not cpu,
        docker_image=docker_image,
        ports=ports,
    )
    run_command(cmd)


@cli.command('kafka-redis')
@click.option(
    '--in-endpoint',
    default='sub+connect:ipc:///tmp/zmq-sockets/output-video.ipc',
    help='Adapter input (module output) ZeroMQ socket endpoint.',
    show_default=True,
)
@fps_meter_options
@click.option(
    '--brokers',
    required=True,
    help='Comma-separated list of Kafka brokers.',
)
@click.option(
    '--topic',
    required=True,
    help='Kafka topic to put messages to.',
)
@click.option(
    '--create-topic',
    is_flag=True,
    default=False,
    help='Create Kafka topic if it does not exist.',
)
@click.option(
    '--create-topic-num-partitions',
    type=click.INT,
    default=1,
    help='Number of partitions for a Kafka topic to create.',
    show_default=True,
)
@click.option(
    '--create-topic-replication-factor',
    type=click.INT,
    default=1,
    help='Replication factor for a Kafka topic to create.',
    show_default=True,
)
@click.option(
    '--create-topic-config',
    default='{}',
    help='JSON dict of a Kafka topic configuration.',
    show_default=True,
)
@click.option(
    '--redis-host',
    required=True,
    help='Redis host.',
)
@click.option(
    '--redis-port',
    type=click.INT,
    default=6379,
    help='Redis port.',
    show_default=True,
)
@click.option(
    '--redis-db',
    type=click.INT,
    default=0,
    help='Redis database.',
    show_default=True,
)
@click.option(
    '--redis-key-prefix',
    default='savant:frames',
    help=(
        'Prefix for Redis keys; frame content is put to Redis with a key '
        '"<redis-key-prefix>:<uuid>".'
    ),
    show_default=True,
)
@click.option(
    '--redis-ttl-seconds',
    type=click.INT,
    default=60,
    help='TTL for Redis keys.',
    show_default=True,
)
@click.option(
    '--queue-size',
    type=click.INT,
    default=50,
    help='Maximum amount of messages in the queue.',
    show_default=True,
)
@adapter_docker_image_option('py')
def kafka_redis_sink(
    in_endpoint: str,
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: Optional[str],
    brokers: str,
    topic: str,
    create_topic: bool,
    create_topic_num_partitions: int,
    create_topic_replication_factor: int,
    create_topic_config: str,
    redis_host: str,
    redis_port: int,
    redis_db: int,
    redis_key_prefix: str,
    redis_ttl_seconds: int,
    queue_size: int,
):
    """Sends video stream metadata to Kafka and frame content to Redis.

    Frame content location is encoded as <redis-host>:<redis-port>:<redis-db>/<redis-key>.
    """

    envs = build_common_envs(
        source_id=None,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
        zmq_endpoint=in_endpoint,
        zmq_type=None,
        zmq_bind=None,
    ) + [
        f'KAFKA_BROKERS={brokers}',
        f'KAFKA_TOPIC={topic}',
        f'KAFKA_CREATE_TOPIC={create_topic}',
        f'KAFKA_CREATE_TOPIC_NUM_PARTITIONS={create_topic_num_partitions}',
        f'KAFKA_CREATE_TOPIC_REPLICATION_FACTOR={create_topic_replication_factor}',
        f'KAFKA_CREATE_TOPIC_CONFIG={create_topic_config}',
        f'REDIS_HOST={redis_host}',
        f'REDIS_PORT={redis_port}',
        f'REDIS_DB={redis_db}',
        f'REDIS_KEY_PREFIX={redis_key_prefix}',
        f'REDIS_TTL_SECONDS={redis_ttl_seconds}',
        f'QUEUE_SIZE={queue_size}',
    ]
    cmd = build_docker_run_command(
        f'sink-kafka-redis-{uuid.uuid4().hex}',
        zmq_endpoints=[in_endpoint],
        entrypoint='python',
        args=['-m', 'adapters.python.sinks.kafka_redis'],
        envs=envs,
        docker_image=docker_image,
    )
    run_command(cmd)


if __name__ == '__main__':
    cli()

#!/usr/bin/env python3
"""Run source adapter."""
import os
import uuid
from typing import List, Optional

import click
from common import (
    adapter_docker_image_option,
    build_common_envs,
    build_docker_run_command,
    detach_option,
    fps_meter_options,
    run_command,
    source_id_option,
)


@click.group()
def cli():
    """Click command line group callback."""


sync_option = click.option(
    '--sync',
    is_flag=True,
    default=False,
    help='Send frames from source synchronously (i.e. at the source file rate).',
)
absolute_ts_option = click.option(
    '--use-absolute-timestamps',
    is_flag=True,
    default=False,
    help=(
        'Put absolute timestamps into the frames, i.e. the timestamps of the '
        'frames start from the time of adapter launch.'
    ),
)


def output_endpoint_options(func):
    """Click options for output endpoint."""
    func = click.option(
        '--out-endpoint',
        default='ipc:///tmp/zmq-sockets/input-video.ipc',
        help='Adapter output (module input) ZeroMQ socket endpoint.',
        show_default=True,
    )(func)
    func = click.option(
        '--out-type',
        default='DEALER',
        help='Adapter output (module input) ZeroMQ socket type.',
        show_default=True,
    )(func)
    func = click.option(
        '--out-bind',
        default=False,
        help=(
            'Adapter output (module input) ZeroMQ socket bind/connect mode '
            '(bind if True).'
        ),
        show_default=True,
    )(func)
    return func


def common_options(func):
    """Common Click source adapter options."""
    func = output_endpoint_options(func)
    func = fps_meter_options(func)
    func = source_id_option(required=True)(func)
    return func


def files_source(
    source_id: Optional[str],
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    sync: bool,
    use_absolute_timestamps: Optional[bool],
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    location: str,
    file_type: str,
    envs: List[str],
    entrypoint: str = '/opt/savant/adapters/gst/sources/media_files.sh',
    extra_volumes: List[str] = None,
    detach: bool = False,
):
    """Read image or video files from LOCATION.
    LOCATION can be single file, directory or HTTP URL.
    """
    if location.startswith('http://') or location.startswith('https://'):
        volumes = []
    else:
        assert os.path.exists(location)
        location = os.path.abspath(location)
        volumes = [f'{location}:{location}:ro']

    if extra_volumes:
        volumes.extend(extra_volumes)

    container_name = f'source-{file_type}-files'
    if source_id is not None:
        container_name = f'{container_name}-{source_id}'
    envs = (
        build_common_envs(
            source_id=source_id,
            fps_period_frames=fps_period_frames,
            fps_period_seconds=fps_period_seconds,
            fps_output=fps_output,
            zmq_endpoint=out_endpoint,
            zmq_type=out_type,
            zmq_bind=out_bind,
            use_absolute_timestamps=use_absolute_timestamps,
        )
        + [f'LOCATION={location}', f'FILE_TYPE={file_type}']
        + envs
    )
    cmd = build_docker_run_command(
        container_name,
        zmq_endpoints=[out_endpoint],
        sync_output=sync,
        entrypoint=entrypoint,
        envs=envs,
        volumes=volumes,
        docker_image=docker_image,
        detach=detach,
    )
    run_command(cmd)


@cli.command('videos')
@click.option(
    '--sort-by-time',
    default=False,
    is_flag=True,
    help='Sort files by modification time.',
)
@click.option(
    '--read-metadata',
    default=False,
    is_flag=True,
    help='Attempt to read the metadata of objects from the JSON file that has the identical name '
    'as the source file with `json` extension, and then send it to the module.',
)
@click.option(
    '--eos-on-file-end',
    help='Send EOS at the end of each file.',
    default=True,
    show_default=True,
)
@common_options
@sync_option
@absolute_ts_option
@adapter_docker_image_option('gstreamer')
@click.argument('location', required=True)
def videos_source(
    source_id: str,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    sync: bool,
    use_absolute_timestamps: Optional[bool],
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    location: str,
    sort_by_time: bool,
    read_metadata: bool,
    eos_on_file_end: bool,
):
    """Read video files from LOCATION.
    LOCATION can be single file, directory or HTTP URL.
    """
    files_source(
        source_id=source_id,
        out_endpoint=out_endpoint,
        out_type=out_type,
        out_bind=out_bind,
        sync=sync,
        docker_image=docker_image,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
        location=location,
        file_type='video',
        envs=[
            f'SORT_BY_TIME={sort_by_time}',
            f'READ_METADATA={read_metadata}',
            f'EOS_ON_FILE_END={eos_on_file_end}',
        ],
        use_absolute_timestamps=use_absolute_timestamps,
    )


@cli.command('video-loop')
@click.option(
    '--read-metadata',
    default=False,
    is_flag=True,
    help='Attempt to read the metadata of objects from the JSON file that has the identical name '
    'as the source file with `json` extension, and then send it to the module.',
)
@click.option(
    '--eos-on-loop-end',
    default=False,
    is_flag=True,
    help='Send EOS on a loop end.',
)
@click.option(
    '--measure-fps-per-loop',
    default=False,
    is_flag=True,
    help='Measure FPS per loop. FPS meter will dump statistics at the end of each loop.',
)
@click.option(
    '--download-path',
    default='/tmp/video-loop-source-downloads',
    help='Path to download files from remote storage.',
    show_default=True,
)
@click.option(
    '--mount-download-path',
    default=False,
    is_flag=True,
    help='Mount path to download files from remote storage to the container.',
)
@click.option(
    '--loss-rate',
    type=click.FLOAT,
    help='Probability to drop the frames.',
)
@common_options
@sync_option
@absolute_ts_option
@adapter_docker_image_option('gstreamer')
@click.argument('location', required=True)
def video_loop_source(
    source_id: str,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    sync: bool,
    use_absolute_timestamps: Optional[bool],
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    measure_fps_per_loop: bool,
    eos_on_loop_end: bool,
    download_path: str,
    mount_download_path: bool,
    loss_rate: float,
    location: str,
    read_metadata: bool,
):
    """Read a video file from LOCATION and loop it.
    LOCATION can be single file, directory or HTTP URL.
    """

    download_path = os.path.abspath(download_path)
    if mount_download_path:
        volumes = [f'{download_path}:{download_path}']
    else:
        volumes = []

    envs = [
        f'MEASURE_FPS_PER_LOOP={measure_fps_per_loop}',
        f'EOS_ON_LOOP_END={eos_on_loop_end}',
        f'READ_METADATA={read_metadata}',
        f'DOWNLOAD_PATH={download_path}',
    ]
    if loss_rate is not None:
        envs.append(f'LOSS_RATE={loss_rate}')

    files_source(
        source_id=source_id,
        out_endpoint=out_endpoint,
        out_type=out_type,
        out_bind=out_bind,
        sync=sync,
        docker_image=docker_image,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
        location=location,
        file_type='video',
        envs=envs,
        entrypoint='/opt/savant/adapters/gst/sources/video_loop.sh',
        extra_volumes=volumes,
        use_absolute_timestamps=use_absolute_timestamps,
    )


@cli.command('multi-stream')
@click.option(
    '--read-metadata',
    default=False,
    is_flag=True,
    help='Attempt to read the metadata of objects from the JSON file that has the identical name '
    'as the source file with `json` extension, and then send it to the module.',
)
@click.option(
    '--download-path',
    default='/tmp/video-loop-source-downloads',
    help='Path to download files from remote storage.',
    show_default=True,
)
@click.option(
    '--mount-download-path',
    default=False,
    is_flag=True,
    help='Mount path to download files from remote storage to the container.',
)
@click.option('--source-id-pattern', help='Pattern for source ID.')
@click.option(
    '--number-of-streams',
    default=1,
    help='Number of streams.',
    show_default=True,
)
@click.option(
    '--number-of-frames',
    type=click.INT,
    help='Number of frames per each source.',
)
@click.option(
    '--shutdown-auth',
    help=(
        'Authentication key for Shutdown message. When specified, a shutdown'
        'message will be sent at the end of the stream.'
    ),
)
@output_endpoint_options
@fps_meter_options
@sync_option
@absolute_ts_option
@adapter_docker_image_option('gstreamer')
@detach_option
@click.argument('location', required=True)
def multi_stream_source(
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    sync: bool,
    use_absolute_timestamps: Optional[bool],
    docker_image: str,
    detach: bool,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    download_path: str,
    mount_download_path: bool,
    source_id_pattern: Optional[str],
    number_of_streams: int,
    number_of_frames: Optional[int],
    shutdown_auth: Optional[str],
    location: str,
    read_metadata: bool,
):
    """Read a video file from LOCATION and sends it to with multiple source IDs.
    LOCATION can be single file or HTTP URL.
    """

    download_path = os.path.abspath(download_path)
    if mount_download_path:
        volumes = [f'{download_path}:{download_path}']
    else:
        volumes = []

    envs = [
        f'NUMBER_OF_STREAMS={number_of_streams}',
        f'READ_METADATA={read_metadata}',
        f'DOWNLOAD_PATH={download_path}',
    ]
    if number_of_frames is not None:
        envs.append(f'NUMBER_OF_FRAMES={number_of_frames}')
    if shutdown_auth is not None:
        envs.append(f'SHUTDOWN_AUTH={shutdown_auth}')
    if source_id_pattern is not None:
        envs.append(f'SOURCE_ID_PATTERN={source_id_pattern}')

    files_source(
        source_id=None,
        out_endpoint=out_endpoint,
        out_type=out_type,
        out_bind=out_bind,
        sync=sync,
        docker_image=docker_image,
        detach=detach,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
        location=location,
        file_type='video',
        envs=envs,
        entrypoint='/opt/savant/adapters/gst/sources/multi_stream.sh',
        extra_volumes=volumes,
        use_absolute_timestamps=use_absolute_timestamps,
    )


@cli.command('images')
@click.option(
    '--framerate',
    default='30/1',
    help='Frame rate of the images.',
    show_default=True,
)
@click.option(
    '--sort-by-time',
    default=False,
    is_flag=True,
    help='Sort files by modification time.',
)
@click.option(
    '--read-metadata',
    default=False,
    is_flag=True,
    help='Attempt to read the metadata of objects from the JSON file that has the identical name '
    'as the source file with `json` extension, and then send it to the module.',
)
@click.option(
    '--eos-on-file-end',
    help='Send EOS at the end of each file.',
    default=False,
    show_default=True,
)
@common_options
@sync_option
@absolute_ts_option
@adapter_docker_image_option('gstreamer')
@click.argument('location', required=True)
def images_source(
    source_id: str,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    sync: bool,
    use_absolute_timestamps: Optional[bool],
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    location: str,
    framerate: str,
    sort_by_time: bool,
    read_metadata: bool,
    eos_on_file_end: bool,
):
    """Read image files from LOCATION.
    LOCATION can be single file, directory or HTTP URL.
    """

    files_source(
        source_id=source_id,
        out_endpoint=out_endpoint,
        out_type=out_type,
        out_bind=out_bind,
        sync=sync,
        docker_image=docker_image,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
        location=location,
        file_type='image',
        envs=[
            f'FRAMERATE={framerate}',
            f'SORT_BY_TIME={sort_by_time}',
            f'READ_METADATA={read_metadata}',
            f'EOS_ON_FILE_END={eos_on_file_end}',
        ],
        use_absolute_timestamps=use_absolute_timestamps,
    )


@cli.command('rtsp')
@common_options
@sync_option
@click.option(
    '--sync-delay',
    type=click.INT,
    help=(
        'Delay in seconds before sending frames. '
        'Ignored when synchronous frames sending is turned off (i.e. no --sync flag).'
    ),
)
@click.option(
    '--rtsp-transport',
    default='tcp',
    help='RTSP transport protocol ("udp" or "tcp").',
    show_default=True,
)
@click.option(
    '--buffer-len',
    default=50,
    help='Maximum amount of frames in the buffer.',
    show_default=True,
)
@click.option(
    '--ffmpeg-loglevel',
    default='info',
    help='Log level for FFmpeg.',
    show_default=True,
)
@absolute_ts_option
@adapter_docker_image_option('gstreamer')
@click.argument('rtsp_uri', required=True)
def rtsp_source(
    source_id: str,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    sync: bool,
    sync_delay: Optional[int],
    buffer_len: int,
    ffmpeg_loglevel: str,
    rtsp_transport: str,
    use_absolute_timestamps: Optional[bool],
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    rtsp_uri: str,
):
    """Read video stream from RTSP_URI."""

    envs = build_common_envs(
        source_id=source_id,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
        zmq_endpoint=out_endpoint,
        zmq_type=out_type,
        zmq_bind=out_bind,
        use_absolute_timestamps=use_absolute_timestamps,
    ) + [
        f'RTSP_URI={rtsp_uri}',
        f'RTSP_TRANSPORT={rtsp_transport}',
        f'BUFFER_LEN={buffer_len}',
        f'FFMPEG_LOGLEVEL={ffmpeg_loglevel}',
    ]
    if sync and sync_delay is not None:
        envs.append(f'SYNC_DELAY={sync_delay}')

    cmd = build_docker_run_command(
        f'source-rtsp-{source_id}',
        zmq_endpoints=[out_endpoint],
        sync_output=sync,
        entrypoint='/opt/savant/adapters/gst/sources/rtsp.sh',
        envs=envs,
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('gige')
@click.option('--width', type=int, help='Width of streaming video')
@click.option('--height', type=int, help='Height of streaming video')
@click.option('--framerate', type=str, help='Framerate of streaming video')
@click.option(
    # TODO: replace with PixelFormat
    # https://github.com/AravisProject/aravis/blob/0.8.22/src/arvmisc.c#L656
    '--input-caps',
    type=str,
    help='Caps of input video (e.g. "video/x-raw,format=RGB"). Look '
    'https://github.com/AravisProject/aravis/blob/0.8.22/src/arvmisc.c#L656 '
    'for PixelFormat -> Caps mapping.',
)
@click.option('--packet-size', type=int, help='GigEVision streaming packet size')
@click.option(
    '--auto-packet-size', type=bool, help='Negotiate GigEVision streaming packet size'
)
@click.option('--exposure', type=float, help='Exposure time (µs)')
@click.option('--exposure-auto', help='Auto Exposure Mode, one of "off", "once", "on"')
@click.option('--gain', type=float, help='Gain (dB)')
@click.option('--gain-auto', help='Auto Gain Mode, one of "off", "once", "on"')
@click.option(
    '--features',
    help='Additional configuration parameters as a space separated list of feature assignations',
)
@click.option(
    '--host-network', default=False, is_flag=True, help='Use the host network.'
)
@click.option(
    '--encode',
    default=False,
    is_flag=True,
    help='Encode the video stream with H264.',
)
@click.option(
    '--encode-bitrate',
    default=2048,
    help='Bitrate of the encoded video stream, in kbit/sec',
    show_default=True,
)
@click.option(
    '--encode-key-int-max',
    default=30,
    help='Maximum interval between two keyframes, in frames',
    show_default=True,
)
@click.option(
    '--encode-speed-preset',
    default='medium',
    help='Speed preset of the encoder, one of "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo"',
    show_default=True,
)
@click.option(
    '--encode-tune',
    default='zerolatency',
    help='Tune of the encoder, one of "psnr", "ssim", "grain", "zerolatency", "psnr", "fastdecode", "animation"',
    show_default=True,
)
@common_options
@absolute_ts_option
@adapter_docker_image_option('gstreamer')
@click.argument('camera_name', required=False)
def gige_cam_source(
    source_id: str,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    use_absolute_timestamps: Optional[bool],
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    width: Optional[int],
    height: Optional[int],
    framerate: Optional[str],
    input_caps: Optional[str],
    packet_size: Optional[int],
    auto_packet_size: Optional[bool],
    exposure: Optional[float],
    exposure_auto: Optional[str],
    gain: Optional[float],
    gain_auto: Optional[str],
    features: Optional[str],
    host_network: bool,
    encode: bool,
    encode_bitrate: int,
    encode_key_int_max: int,
    encode_speed_preset: str,
    encode_tune: str,
    camera_name: Optional[str],
):
    """Read video stream from GigE camera CAMERA_NAME.

    If the camera is a GigEVision, CAMERA_NAME can be either:

      - <vendor>-<model>-<serial>

      - <vendor_alias>-<serial>

      - <vendor>-<serial>

      - <user_id>

      - <ip_address>

      - <mac_address>
    """

    envs = build_common_envs(
        source_id=source_id,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
        zmq_endpoint=out_endpoint,
        zmq_type=out_type,
        zmq_bind=out_bind,
        use_absolute_timestamps=use_absolute_timestamps,
    )

    envs_dict = {
        'CAMERA_NAME': camera_name,
        'WIDTH': width,
        'HEIGHT': height,
        'FRAMERATE': framerate,
        'INPUT_CAPS': input_caps,
        'PACKET_SIZE': packet_size,
        'AUTO_PACKET_SIZE': (
            int(auto_packet_size) if auto_packet_size is not None else None
        ),
        'EXPOSURE': exposure,
        'EXPOSURE_AUTO': exposure_auto,
        'GAIN': gain,
        'GAIN_AUTO': gain_auto,
        'FEATURES': features,
        'ENCODE': encode,
        'ENCODE_BITRATE': encode_bitrate,
        'ENCODE_KEY_INT_MAX': encode_key_int_max,
        'ENCODE_SPEED_PRESET': encode_speed_preset,
        'ENCODE_TUNE': encode_tune,
    }
    for k, v in envs_dict.items():
        if v is not None:
            envs.append(f'{k}={v}')

    cmd = build_docker_run_command(
        f'source-gige-{source_id}',
        zmq_endpoints=[out_endpoint],
        entrypoint='/opt/savant/adapters/gst/sources/gige_cam.sh',
        envs=envs,
        docker_image=docker_image,
        host_network=host_network,
    )
    run_command(cmd)


@cli.command('ffmpeg')
@common_options
@sync_option
@click.option(
    '--sync-delay',
    type=click.INT,
    help=(
        'Delay in seconds before sending frames. '
        'Ignored when synchronous frames sending is turned off (i.e. no --sync flag).'
    ),
)
@click.option(
    '--ffmpeg-params',
    help=(
        'A comma separated string "key=value" with parameters for FFmpeg '
        '(e.g. "rtsp_transport=tcp", "input_format=mjpeg,video_size=1280x720").'
    ),
)
@click.option(
    '--buffer-len',
    default=50,
    help='Maximum amount of frames in the buffer.',
    show_default=True,
)
@click.option(
    '--ffmpeg-loglevel',
    default='info',
    help='Log level for FFmpeg.',
    show_default=True,
)
@click.option(
    '--device',
    help='Device to mount to the container (e.g. "/dev/video0").',
)
@absolute_ts_option
@adapter_docker_image_option('gstreamer')
@click.argument('uri', required=True)
def ffmpeg_source(
    source_id: str,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    sync: bool,
    sync_delay: Optional[int],
    ffmpeg_params: Optional[str],
    buffer_len: int,
    ffmpeg_loglevel: str,
    device: Optional[str],
    use_absolute_timestamps: Optional[bool],
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    uri: str,
):
    """Read video stream from URI using FFmpeg library."""

    envs = build_common_envs(
        source_id=source_id,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
        zmq_endpoint=out_endpoint,
        zmq_type=out_type,
        zmq_bind=out_bind,
        use_absolute_timestamps=use_absolute_timestamps,
    ) + [
        f'URI={uri}',
        f'BUFFER_LEN={buffer_len}',
        f'FFMPEG_LOGLEVEL={ffmpeg_loglevel}',
    ]
    if sync and sync_delay is not None:
        envs.append(f'SYNC_DELAY={sync_delay}')
    if ffmpeg_params:
        envs.append(f'FFMPEG_PARAMS={ffmpeg_params}')
    devices = [device] if device is not None else []

    cmd = build_docker_run_command(
        f'source-rtsp-{source_id}',
        zmq_endpoints=[out_endpoint],
        sync_output=sync,
        entrypoint='/opt/savant/adapters/gst/sources/ffmpeg.sh',
        envs=envs,
        devices=devices,
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('kafka-redis')
@click.option(
    '--out-endpoint',
    default='pub+connect:ipc:///tmp/zmq-sockets/input-video.ipc',
    help='Adapter output (module input) ZeroMQ socket endpoint.',
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
    help='Kafka topic to read messages from.',
)
@click.option(
    '--group-id',
    required=True,
    help='Kafka consumer group ID.',
)
@click.option(
    '--poll-timeout',
    type=click.INT,
    default=1,
    help='Timeout for Kafka consumer poll, in seconds.',
    show_default=True,
)
@click.option(
    '--auto-commit-interval-ms',
    type=click.INT,
    default=1000,
    help='Frequency in milliseconds that the consumer offsets are auto-committed to Kafka.',
    show_default=True,
)
@click.option(
    '--auto-offset-reset',
    default='latest',
    help='Position to start reading messages from Kafka topic when the group is created.',
    show_default=True,
)
@click.option(
    '--partition-assignment-strategy',
    default='roundrobin',
    help='Strategy to assign partitions to consumers.',
    show_default=True,
)
@click.option(
    '--max-poll-interval-ms',
    type=click.INT,
    default=300000,
    help=(
        'Maximum delay in milliseconds between invocations of poll() '
        'when using consumer group management.'
    ),
    show_default=True,
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
    '--queue-size',
    type=click.INT,
    default=50,
    help='Maximum amount of messages in the queue.',
    show_default=True,
)
@adapter_docker_image_option('py')
def kafka_redis_source(
    out_endpoint: str,
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: Optional[str],
    brokers: str,
    topic: str,
    group_id: str,
    poll_timeout: int,
    auto_commit_interval_ms: int,
    auto_offset_reset: str,
    partition_assignment_strategy: str,
    max_poll_interval_ms: int,
    create_topic: bool,
    create_topic_num_partitions: int,
    create_topic_replication_factor: int,
    create_topic_config: str,
    queue_size: int,
):
    """Takes video stream metadata from Kafka and fetches frame content from Redis.

    Frame content location is encoded as <redis-host>:<redis-port>:<redis-db>/<redis-key>.
    """

    envs = build_common_envs(
        source_id=None,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
        zmq_endpoint=out_endpoint,
        zmq_type=None,
        zmq_bind=None,
    ) + [
        f'KAFKA_BROKERS={brokers}',
        f'KAFKA_TOPIC={topic}',
        f'KAFKA_GROUP_ID={group_id}',
        f'KAFKA_POLL_TIMEOUT={poll_timeout}',
        f'KAFKA_AUTO_COMMIT_INTERVAL_MS={auto_commit_interval_ms}',
        f'KAFKA_AUTO_OFFSET_RESET={auto_offset_reset}',
        f'KAFKA_PARTITION_ASSIGNMENT_STRATEGY={partition_assignment_strategy}',
        f'KAFKA_MAX_POLL_INTERVAL_MS={max_poll_interval_ms}',
        f'KAFKA_CREATE_TOPIC={create_topic}',
        f'KAFKA_CREATE_TOPIC_NUM_PARTITIONS={create_topic_num_partitions}',
        f'KAFKA_CREATE_TOPIC_REPLICATION_FACTOR={create_topic_replication_factor}',
        f'KAFKA_CREATE_TOPIC_CONFIG={create_topic_config}',
        f'QUEUE_SIZE={queue_size}',
    ]
    cmd = build_docker_run_command(
        f'source-kafka-redis-{uuid.uuid4().hex}',
        zmq_endpoints=[out_endpoint],
        entrypoint='python',
        args=['-m', 'adapters.python.sources.kafka_redis'],
        envs=envs,
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('kvs')
@click.option(
    '--out-endpoint',
    default='pub+connect:ipc:///tmp/zmq-sockets/input-video.ipc',
    help='Adapter output (module input) ZeroMQ socket endpoint.',
    show_default=True,
)
@click.option(
    '--aws-region',
    required=True,
    help='AWS region.',
)
@click.option(
    '--aws-access-key',
    required=True,
    help='AWS access key ID.',
)
@click.option(
    '--aws-secret-key',
    required=True,
    help='AWS secret access key.',
)
@click.option(
    '--stream-name',
    required=True,
    help='Name of the Kinesis Video Stream.',
)
@click.option(
    '--timestamp',
    required=False,
    help=(
        'Either timestamp in format "%Y-%m-%dT%H:%M:%S" or delay from current '
        'time in "-<delay>(s\|m)". E.g. "2024-03-12T06:57:00", "-30s", "-1m".'
    ),
)
@click.option(
    '--no-playing',
    is_flag=True,
    default=False,
    help='Do not start playing stream immediately.',
)
@click.option(
    '--api-port',
    default=18367,
    help='Port for the REST API. This port is always published.',
    show_default=True,
)
@click.option(
    '--save-state',
    is_flag=True,
    default=False,
    help='Save state to the state file.',
)
@click.option(
    '--state-path',
    default='/state/state.json',
    help='Path to the state file. Ignored if --save-state is not set.',
    show_default=True,
)
@click.option(
    '--mount-state-path',
    required=False,
    help=(
        'Where to mount the directory with the state file. '
        'Ignored if --save-state is not set.'
    ),
)
@source_id_option(required=True)
@sync_option
@fps_meter_options
@adapter_docker_image_option('gstreamer')
def kvs_source(
    source_id: str,
    out_endpoint: str,
    sync: bool,
    aws_region: str,
    aws_access_key: str,
    aws_secret_key: str,
    stream_name: str,
    timestamp: Optional[str],
    no_playing: bool,
    api_port: int,
    save_state: bool,
    state_path: str,
    mount_state_path: Optional[str],
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: Optional[str],
    docker_image: str,
):
    """Read video stream from Kinesis Video Stream.

    REST API is available at http://<container-host>:<api-port>.

    See http://<container-host>:<api-port>/docs for API documentation
    """

    envs = build_common_envs(
        source_id=source_id,
        zmq_endpoint=out_endpoint,
        zmq_type=None,
        zmq_bind=None,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
    ) + [
        f'AWS_REGION={aws_region}',
        f'AWS_ACCESS_KEY={aws_access_key}',
        f'AWS_SECRET_KEY={aws_secret_key}',
        f'STREAM_NAME={stream_name}',
        f'TIMESTAMP={timestamp}',
        f'PLAYING={not no_playing}',
        f'API_PORT={api_port}',
        f'SAVE_STATE={save_state}',
        f'STATE_PATH={state_path}',
    ]
    if save_state and mount_state_path:
        assert os.path.isabs(
            state_path
        ), 'State path must be absolute when mounting state path.'
        state_dir = os.path.dirname(state_path)
        assert state_dir != '/', 'State directory must not be root.'
        volumes = [f'{os.path.abspath(mount_state_path)}:{state_dir}']
    else:
        volumes = []

    cmd = build_docker_run_command(
        f'source-kvs-{source_id}',
        zmq_endpoints=[out_endpoint],
        sync_output=sync,
        entrypoint='python',
        args=['-m', 'adapters.gst.sources.kvs'],
        envs=envs,
        volumes=volumes,
        docker_image=docker_image,
        ports=[(api_port, api_port)],
    )
    run_command(cmd)


if __name__ == '__main__':
    cli()

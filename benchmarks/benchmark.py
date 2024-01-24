#!/usr/bin/env python3
"""Run benchmark for drawing on frames."""

import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from savant.deepstream.opencv_utils import (
    alpha_comp,
    apply_cuda_filter,
    draw_rect,
    nvds_to_gpu_mat,
)
from savant.deepstream.utils.iterator import nvds_frame_meta_iterator
from savant.deepstream.utils.surface import get_nvds_buf_surface

sys.path.append('../../')

import cv2
import gi

gi.require_version('Gst', '1.0')
import pyds
from gi.repository import GLib, Gst

scale = 10**6  # milliseconds
RECT_COLOR = (127, 127, 127, 255)  # gray
RECT_N = 20
RECT_WIDTH = 100
RECT_HEIGHT = 100
FACE_WIDTH = 30
FACE_HEIGHT = 40


@dataclass
class BenchmarkData:
    overlay: np.ndarray
    overlay_mat: cv2.cuda.GpuMat
    points: List[Tuple[int, int]]
    cuda_blur_filter: cv2.cuda.Filter


def benchmark_cpu_overlay(
    gst_buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
    data: BenchmarkData,
):
    with get_nvds_buf_surface(gst_buffer, nvds_frame_meta) as np_frame:
        height, width, _ = data.overlay.shape
        np_frame[:height, :width] = data.overlay


def benchmark_gpu_overlay(
    gst_buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
    data: BenchmarkData,
):
    with nvds_to_gpu_mat(gst_buffer, nvds_frame_meta) as frame_mat:
        alpha_comp(frame_mat, data.overlay, (0, 0))


def benchmark_gpu_overlay_single(
    gst_buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
    data: BenchmarkData,
):
    with nvds_to_gpu_mat(gst_buffer, nvds_frame_meta) as frame_mat:
        alpha_comp(frame_mat, data.overlay_mat, (0, 0))


def benchmark_cpu_draw_rectangles(
    gst_buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
    data: BenchmarkData,
):
    with get_nvds_buf_surface(gst_buffer, nvds_frame_meta) as np_frame:
        for x, y in data.points:
            cv2.rectangle(
                np_frame,
                (x, y),
                (x + RECT_WIDTH, y + RECT_HEIGHT),
                RECT_COLOR,
                4,
            )


def benchmark_gpu_draw_rectangles(
    gst_buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
    data: BenchmarkData,
):
    with nvds_to_gpu_mat(gst_buffer, nvds_frame_meta) as frame_mat:
        for x, y in data.points:
            draw_rect(
                frame_mat,
                (x, y, x + RECT_WIDTH, y + RECT_HEIGHT),
                RECT_COLOR,
                4,
            )


def benchmark_cpu_blur_faces(
    gst_buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
    data: BenchmarkData,
):
    with get_nvds_buf_surface(gst_buffer, nvds_frame_meta) as np_frame:
        for x, y in data.points:
            np_frame[y : y + FACE_HEIGHT, x : x + FACE_WIDTH] = cv2.GaussianBlur(
                np_frame[y : y + FACE_HEIGHT, x : x + FACE_WIDTH],
                (31, 31),
                100,
                100,
            )


def benchmark_gpu_blur_faces(
    gst_buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
    data: BenchmarkData,
):
    with nvds_to_gpu_mat(gst_buffer, nvds_frame_meta) as frame_mat:
        for x, y in data.points:
            apply_cuda_filter(
                data.cuda_blur_filter, frame_mat, (x, y, FACE_WIDTH, FACE_HEIGHT)
            )


def benchmark_gpu_blur_faces_parallel(
    gst_buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
    data: BenchmarkData,
):
    with nvds_to_gpu_mat(gst_buffer, nvds_frame_meta) as frame_mat:
        streams = []
        for x, y in data.points:
            streams.append(cv2.cuda.Stream())
            apply_cuda_filter(
                data.cuda_blur_filter,
                frame_mat,
                (x, y, FACE_WIDTH, FACE_HEIGHT),
                stream=streams[-1],
            )
        for i, stream in enumerate(streams):
            stream.waitForCompletion()


def benchmark_gpu_blur_faces_single_stream(
    gst_buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
    data: BenchmarkData,
):
    with nvds_to_gpu_mat(gst_buffer, nvds_frame_meta) as frame_mat:
        stream = cv2.cuda.Stream()
        for x, y in data.points:
            apply_cuda_filter(
                data.cuda_blur_filter,
                frame_mat,
                (x, y, FACE_WIDTH, FACE_HEIGHT),
                stream=stream,
            )
        stream.waitForCompletion()


def benchmark_gpu_blur_faces_in_cpu(
    gst_buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
    data: BenchmarkData,
):
    with nvds_to_gpu_mat(gst_buffer, nvds_frame_meta) as frame_mat:
        for x, y in data.points:
            roi = cv2.cuda.GpuMat(frame_mat, (x, y, FACE_WIDTH, FACE_HEIGHT))
            roi.upload(
                cv2.GaussianBlur(
                    roi.download(),
                    (31, 31),
                    100,
                    100,
                )
            )


def benchmark_gpu_download_upload(
    gst_buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
    data: BenchmarkData,
):
    with nvds_to_gpu_mat(gst_buffer, nvds_frame_meta) as frame_mat:
        for x, y in data.points:
            roi = cv2.cuda.GpuMat(frame_mat, (x, y, RECT_WIDTH, RECT_HEIGHT))
            part = roi.download()
            roi.upload(part)


BenchmarkFunc = Callable[[Gst.Buffer, pyds.NvDsFrameMeta, BenchmarkData], None]
BENCHMARK_FUNCS: Dict[str, Tuple[Optional[BenchmarkFunc], Optional[BenchmarkFunc]]] = {
    'overlay': (benchmark_cpu_overlay, benchmark_gpu_overlay),
    'overlay-single': (None, benchmark_gpu_overlay_single),
    'draw-rectangles': (benchmark_cpu_draw_rectangles, benchmark_gpu_draw_rectangles),
    'blur-faces': (benchmark_cpu_blur_faces, benchmark_gpu_blur_faces),
    'blur-faces-parallel': (None, benchmark_gpu_blur_faces_parallel),
    'blur-faces-single-stream': (None, benchmark_gpu_blur_faces_single_stream),
    'blur-faces-in-cpu': (None, benchmark_gpu_blur_faces_in_cpu),
    'download-upload': (None, benchmark_gpu_download_upload),
}


def pad_buffer_probe(
    pad: Gst.Pad,
    info: Gst.PadProbeInfo,
    benchmark_func: BenchmarkFunc,
    data: BenchmarkData,
    measurements: List[float],
):
    data.points = [
        (random.randint(0, 1900 - RECT_WIDTH), random.randint(0, 1000 - RECT_HEIGHT))
        for _ in range(RECT_N)
    ]
    gst_buffer: Gst.Buffer = info.get_buffer()
    nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
        ts1 = time.time()
        benchmark_func(gst_buffer, nvds_frame_meta, data)
        ts2 = time.time()
        measurements.append((ts2 - ts1) * scale)

    return Gst.PadProbeReturn.OK


def is_aarch64():
    return platform.uname()[4] == 'aarch64'


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True


def main(args):
    assert (
        len(args) > 2
    ), 'Usage: ./benchmark.py <benchmark-name> <cpu|gpu> [n-frames] [output-filename]'
    benchmark_name = args[1]
    is_gpu = args[2] == 'gpu'
    assert (
        benchmark_name in BENCHMARK_FUNCS
    ), f'Available benchmark names: {", ".join(BENCHMARK_FUNCS.keys())}'
    benchmark_func = BENCHMARK_FUNCS[benchmark_name][int(is_gpu)]
    assert benchmark_func is not None, 'Benchmark not implemented'

    output_filename = None
    if len(args) > 3:
        n_frames = int(args[3])
        if len(args) > 4:
            output_filename = args[4]
    else:
        n_frames = 1

    Gst.init(None)

    print("Creating Pipeline")
    pipeline = Gst.Pipeline()
    is_live = False

    print("Creating streammux")
    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    pipeline.add(streammux)

    print("Creating source")
    source = Gst.ElementFactory.make("videotestsrc", "source")
    pipeline.add(source)

    print("Creating source converter")
    source_converter = Gst.ElementFactory.make("nvvideoconvert", "source-converter")
    pipeline.add(source_converter)

    print("Creating source capsfilter")
    source_capsfilter = Gst.ElementFactory.make("capsfilter", "source-capsfilter")
    pipeline.add(source_capsfilter)

    print("Creating workload")
    workload = Gst.ElementFactory.make("identity", "workload")
    pipeline.add(workload)

    print("Creating streamdemux")
    streamdemux = Gst.ElementFactory.make("nvstreamdemux", "streamdemux")
    pipeline.add(streamdemux)

    print("Creating queue")
    queue = Gst.ElementFactory.make("queue", "queue")
    pipeline.add(queue)

    if output_filename:
        print("Creating converter")
        converter = Gst.ElementFactory.make("nvvideoconvert", "converter")
        pipeline.add(converter)

        print("Creating sink_capsfilter")
        sink_capsfilter = Gst.ElementFactory.make("capsfilter", "sink_capsfilter")
        pipeline.add(sink_capsfilter)

        print("Creating encoder")
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        pipeline.add(encoder)

        print("Creating parser")
        parser = Gst.ElementFactory.make("h264parse", "parser")
        pipeline.add(parser)

        print("Creating sink")
        sink = Gst.ElementFactory.make("filesink", "sink")
        pipeline.add(sink)
    else:
        print("Creating sink")
        sink = Gst.ElementFactory.make("fakesink", "sink")
        pipeline.add(sink)

    source.set_property('num-buffers', n_frames)

    if is_live:
        streammux.set_property('live-source', 1)
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)

    sink.set_property("sync", 0)
    sink.set_property("qos", 0)
    sink.set_property("enable-last-sample", 0)
    if output_filename:
        sink.set_property("location", output_filename)

    if not is_aarch64():
        nv_buf_memory_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        source_converter.set_property("nvbuf-memory-type", nv_buf_memory_type)
        streammux.set_property("nvbuf-memory-type", nv_buf_memory_type)
        if output_filename:
            converter.set_property("nvbuf-memory-type", nv_buf_memory_type)

    source_capsfilter.set_property(
        'caps',
        Gst.Caps.from_string(
            'video/x-raw(memory:NVMM), format=RGBA, width=1920, height=1080'
        ),
    )
    if output_filename:
        sink_capsfilter.set_property(
            'caps',
            Gst.Caps.from_string(
                'video/x-raw(memory:NVMM), format=RGBA, width=1920, height=1080'
            ),
        )

    print("Linking elements in the Pipeline")

    assert source.link(source_converter)
    assert source_converter.link(source_capsfilter)

    assert (
        source_capsfilter.get_static_pad('src').link(
            streammux.get_request_pad('sink_0')
        )
        == Gst.PadLinkReturn.OK
    )

    assert streammux.link(workload)
    assert workload.link(streamdemux)

    streamdemux_src_pad = streamdemux.get_request_pad('src_0')
    streamdemux.get_request_pad('src_1')
    streamdemux.get_request_pad('src_2')
    streamdemux.get_request_pad('src_3')
    queue_sink_pad = queue.get_static_pad('sink')
    assert streamdemux_src_pad.link(queue_sink_pad) == Gst.PadLinkReturn.OK

    if output_filename:
        assert queue.link(converter)
        assert converter.link(encoder)
        assert encoder.link(parser)
        assert parser.link(sink)
    else:
        assert queue.link(sink)

    # create an event loop and feed gstreamer bus messages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    sink_pad = workload.get_static_pad("sink")
    measurements = []
    if not sink_pad:
        sys.stderr.write("Unable to get sink pad")
    else:
        overlay = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)
        benchmark_data = BenchmarkData(
            overlay=overlay,
            overlay_mat=cv2.cuda.GpuMat(overlay),
            points=[],
            cuda_blur_filter=cv2.cuda.createGaussianFilter(
                cv2.CV_8UC4,
                cv2.CV_8UC4,
                (31, 31),
                100,
                100,
            ),
        )
        sink_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            pad_buffer_probe,
            benchmark_func,
            benchmark_data,
            measurements,
        )

    print("Starting pipeline")
    ts1 = time.time()
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)
    ts2 = time.time()
    elapsed = ts2 - ts1
    print(f"Elapsed: {elapsed:.2f}, framerate: {n_frames / elapsed:.2f}")
    metrics = [
        ('min', min(measurements)),
        ('max', max(measurements)),
        ('mean', statistics.mean(measurements)),
        ('median', statistics.median(measurements)),
        ('80%', statistics.quantiles(measurements, n=5)[-1]),
        ('90%', statistics.quantiles(measurements, n=10)[-1]),
        ('95%', statistics.quantiles(measurements, n=20)[-1]),
        ('99%', statistics.quantiles(measurements, n=100)[-1]),
        ('stdev', statistics.stdev(measurements)),
    ]
    for name, val in metrics:
        print(f'{name}: {val:.3f}')
    device_name = "gpu" if is_gpu else "cpu"
    with open('metrics.csv', 'a') as f:
        f.write(
            ','.join(
                [benchmark_name, device_name, str(n_frames)]
                + [f'{val:.3f}' for _, val in metrics]
            )
        )
        f.write('\n')
    measurements_filename = f'measurements-{benchmark_name}-{device_name}.txt'
    with open(measurements_filename, 'w') as f:
        for x in measurements:
            f.write(f'{x}\n')


if __name__ == '__main__':
    sys.exit(main(sys.argv))

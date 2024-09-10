import time
from typing import Tuple

from savant.deepstream.runner import NvDsPipelineRunner
from savant.gstreamer import Gst, GstApp
from savant.utils.logging import get_logger, init_logging

logger = get_logger(__name__)


def build_pipeline(n: int) -> Gst.Pipeline:
    pipeline: Gst.Pipeline = Gst.Pipeline.new()

    nvstreammux: Gst.Element = Gst.ElementFactory.make('nvstreammux', 'nvstreammux')
    nvstreammux.set_property('batch-size', 1)

    nvstreamdemux: Gst.Element = Gst.ElementFactory.make(
        'nvstreamdemux', 'nvstreamdemux'
    )
    for i in range(n):
        nvstreamdemux.request_pad_simple(f'src_{i}')

    # fakesink: Gst.Element = Gst.ElementFactory.make('fakesink', 'fakesink')
    # fakesink.set_property('sync', False)
    # fakesink.set_property('async', False)
    # fakesink.set_property('enable-last-sample', False)
    # fakesink.set_property('qos', False)

    pipeline.add(nvstreammux)
    pipeline.add(nvstreamdemux)

    assert nvstreammux.link(
        nvstreamdemux
    ), f'Failed to link {nvstreammux.get_name()} to {nvstreamdemux.get_name()}'

    nvstreammux.sync_state_with_parent()
    nvstreamdemux.sync_state_with_parent()

    return pipeline


def create_source(
    pipeline: Gst.Pipeline,
    idx: int,
    width: int,
    height: int,
    framerate: str,
) -> Tuple[GstApp.AppSrc, Gst.Pad]:
    logger.info('[%s] Adding source to pipeline', idx)
    appsrc: GstApp.AppSrc = Gst.ElementFactory.make('appsrc', f'appsrc-{idx}')
    appsrc.set_property(
        'caps',
        Gst.Caps.from_string(
            f'video/x-raw,format=RGBA,width={width},height={height},framerate={framerate}'
        ),
    )
    logger.info('[%s] Created %s', idx, appsrc.get_name())

    nvvideoconvert: Gst.Element = Gst.ElementFactory.make(
        'nvvideoconvert', f'nvvideoconvert-src-{idx}'
    )
    logger.info('[%s] Created %s', idx, nvvideoconvert.get_name())

    nvvideoconvert_capsfilter: Gst.Element = Gst.ElementFactory.make(
        'capsfilter', f'nvvideoconvert-src-capsfilter-{idx}'
    )
    nvvideoconvert_capsfilter.set_property(
        'caps',
        Gst.Caps.from_string(
            f'video/x-raw(memory:NVMM),format=NV12,width={width},height={height},framerate={framerate}'
        ),
    )
    logger.info('[%s] Created %s', idx, nvvideoconvert_capsfilter.get_name())

    pipeline.add(appsrc)
    logger.info('[%s] Added %s to pipeline', idx, appsrc.get_name())
    pipeline.add(nvvideoconvert)
    logger.info('[%s] Added %s to pipeline', idx, nvvideoconvert.get_name())
    pipeline.add(nvvideoconvert_capsfilter)
    logger.info('[%s] Added %s to pipeline', idx, nvvideoconvert_capsfilter.get_name())

    assert appsrc.link(
        nvvideoconvert
    ), f'Failed to link {appsrc.get_name()} to {nvvideoconvert.get_name()}'
    logger.info(
        '[%s] Linked %s to %s', idx, appsrc.get_name(), nvvideoconvert.get_name()
    )
    assert nvvideoconvert.link(
        nvvideoconvert_capsfilter
    ), f'Failed to link {nvvideoconvert.get_name()} to {nvvideoconvert_capsfilter.get_name()}'
    logger.info(
        '[%s] Linked %s to %s',
        idx,
        nvvideoconvert.get_name(),
        nvvideoconvert_capsfilter.get_name(),
    )

    appsrc.sync_state_with_parent()
    nvvideoconvert.sync_state_with_parent()
    nvvideoconvert_capsfilter.sync_state_with_parent()

    logger.info('[%s] Synced states', idx)

    return appsrc, nvvideoconvert_capsfilter.get_static_pad('src')


def link_source(
    pipeline: Gst.Pipeline,
    idx: int,
    src_pad: Gst.Pad,
    width: int,
    height: int,
):
    logger.info('[%s] Linking source to pipeline', idx)

    queue: Gst.Element = Gst.ElementFactory.make('queue', f'queue-mux-{idx}')
    logger.info('[%s] Created %s', idx, queue.get_name())

    nvvideoconvert: Gst.Element = Gst.ElementFactory.make(
        'nvvideoconvert', f'nvvideoconvert-mux-{idx}'
    )
    logger.info('[%s] Created %s', idx, nvvideoconvert.get_name())

    nvvideoconvert_capsfilter: Gst.Element = Gst.ElementFactory.make(
        'capsfilter', f'nvvideoconvert-mux-capsfilter-{idx}'
    )
    nvvideoconvert_capsfilter.set_property(
        'caps',
        Gst.Caps.from_string(
            f'video/x-raw(memory:NVMM),format=RGBA,width={width},height={height}'
        ),
    )
    logger.info('[%s] Created %s', idx, nvvideoconvert_capsfilter.get_name())

    nvstreammux: Gst.Element = pipeline.get_by_name('nvstreammux')

    pipeline.add(queue)
    logger.info('[%s] Added %s to pipeline', idx, queue.get_name())
    pipeline.add(nvvideoconvert)
    logger.info('[%s] Added %s to pipeline', idx, nvvideoconvert.get_name())
    pipeline.add(nvvideoconvert_capsfilter)
    logger.info('[%s] Added %s to pipeline', idx, nvvideoconvert_capsfilter.get_name())

    assert (
        src_pad.link(queue.get_static_pad('sink')) == Gst.PadLinkReturn.OK
    ), f'Failed to link {src_pad.get_name()} to {queue.get_name()}'
    logger.info('[%s] Linked %s to %s', idx, src_pad.get_name(), queue.get_name())
    assert queue.link(
        nvvideoconvert
    ), f'Failed to link {queue.get_name()} to {nvvideoconvert.get_name()}'
    logger.info(
        '[%s] Linked %s to %s', idx, queue.get_name(), nvvideoconvert.get_name()
    )

    assert nvvideoconvert.link(
        nvvideoconvert_capsfilter
    ), f'Failed to link {nvvideoconvert.get_name()} to {nvvideoconvert_capsfilter.get_name()}'
    logger.info(
        '[%s] Linked %s to %s',
        idx,
        nvvideoconvert.get_name(),
        nvvideoconvert_capsfilter.get_name(),
    )

    nvstreammux_pad: Gst.Pad = nvstreammux.request_pad_simple(f'sink_{idx}')
    assert (
        nvvideoconvert_capsfilter.get_static_pad('src').link(nvstreammux_pad)
        == Gst.PadLinkReturn.OK
    ), f'Failed to link {nvvideoconvert_capsfilter.get_name()} to {nvstreammux.get_name()}'
    logger.info(
        '[%s] Linked %s to %s',
        idx,
        nvvideoconvert_capsfilter.get_name(),
        nvstreammux.get_name(),
    )

    queue.sync_state_with_parent()
    nvvideoconvert.sync_state_with_parent()
    nvvideoconvert_capsfilter.sync_state_with_parent()
    logger.info('[%s] Synced states', idx)


def create_sink(pipeline: Gst.Pipeline, idx: int) -> Gst.Pad:
    logger.info('[%s] Adding sink to pipeline', idx)
    fakesink: Gst.Element = Gst.ElementFactory.make('fakesink', f'fakesink-{idx}')
    fakesink.set_property('sync', False)
    fakesink.set_property('async', False)
    fakesink.set_property('enable-last-sample', False)
    fakesink.set_property('qos', False)
    logger.info('[%s] Created %s', idx, fakesink.get_name())

    pipeline.add(fakesink)
    logger.info('[%s] Added %s to pipeline', idx, fakesink.get_name())

    fakesink.sync_state_with_parent()
    logger.info('[%s] Synced states', idx)

    return fakesink.get_static_pad('sink')


def link_sink(pipeline: Gst.Pipeline, idx: int, sink_pad: Gst.Pad):
    logger.info('[%s] Linking sink to pipeline', idx)

    nvstreamdemux: Gst.Element = pipeline.get_by_name('nvstreamdemux')

    assert (
        nvstreamdemux.get_static_pad(f'src_{idx}').link(sink_pad)
        == Gst.PadLinkReturn.OK
    ), f'Failed to link {sink_pad.get_name()} to {nvstreamdemux.get_name()}'
    logger.info(
        '[%s] Linked %s to %s', idx, sink_pad.get_name(), nvstreamdemux.get_name()
    )


def main():
    init_logging()
    logger.info('Starting...')
    Gst.init(None)

    pipeline = build_pipeline(2)
    logger.info('Pipeline built')

    with NvDsPipelineRunner(pipeline) as runner:
        sink1_pad = create_sink(pipeline, 0)
        link_sink(pipeline, 0, sink1_pad)

        appsrc_1, src1_pad = create_source(pipeline, 0, 640, 480, '30/1')
        link_source(pipeline, 0, src1_pad, 640, 480)

        sink2_pad = create_sink(pipeline, 1)
        link_sink(pipeline, 1, sink2_pad)

        appsrc_2, src2_pad = create_source(pipeline, 1, 1280, 720, '25/1')
        link_source(pipeline, 1, src2_pad, 1280, 720)

        time.sleep(1)

        for i in range(10):
            logger.info('Running iteration %s', i)
            appsrc_1: GstApp.AppSrc
            appsrc_1.push_buffer(Gst.Buffer.new_wrapped(b'\x00' * 640 * 480 * 4))
            time.sleep(0.5)
    logger.info('Exiting...')


if __name__ == '__main__':
    main()

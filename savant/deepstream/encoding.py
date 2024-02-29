import time
from typing import Any, Dict

from savant.config.schema import PipelineElement
from savant.deepstream.element_factory import NvDsElementFactory
from savant.deepstream.runner import NvDsPipelineRunner
from savant.gstreamer import Gst  # noqa:F401
from savant.gstreamer.codecs import CODEC_BY_NAME, Codec
from savant.gstreamer.element_factory import GstElementFactory
from savant.utils.logging import get_logger


def check_encoder_is_available(parameters: Dict[str, Any]) -> bool:
    """Check if encoder is available."""

    logger = get_logger(__name__)

    output_frame = parameters.get('output_frame')
    codec_name = output_frame['codec']
    if not output_frame or codec_name == 'copy':
        return True

    codec = CODEC_BY_NAME[codec_name]
    if codec not in [Codec.H264, Codec.HEVC]:
        return True

    logger.info('Checking if encoder for codec %r is available', codec_name)
    encoder = codec.value.encoder(output_frame.get('encoder'))
    output_caps = codec.value.caps_with_params
    if codec == Codec.H264 and encoder == codec.value.sw_encoder:
        profile = output_frame.get('profile')
        if profile is None:
            profile = 'baseline'
        output_caps = f'{output_caps},profile={profile}'

    pipeline: Gst.Pipeline = Gst.Pipeline.new()
    if codec in [Codec.H264, Codec.HEVC]:
        parser_props = {'config-interval': -1}
    else:
        parser_props = {}
    elements = [
        PipelineElement(
            'videotestsrc',
            properties={'num-buffers': 1},
        ),
        PipelineElement(
            'capsfilter',
            properties={'caps': 'video/x-raw,width=256,height=256'},
        ),
        PipelineElement('nvvideoconvert'),
        PipelineElement(
            encoder,
            properties=output_frame.get('encoder_params', {}),
        ),
        PipelineElement(
            codec.value.parser,
            properties=parser_props,
        ),
        PipelineElement(
            'capsfilter',
            properties={'caps': output_caps},
        ),
        PipelineElement('fakesink'),
    ]

    last_gst_element = None
    for element in elements:
        if element.element == 'capsfilter':
            # Cannot use NvDsElementFactory().create() since it creates videotestsrc as a bin.
            gst_element = GstElementFactory.create_caps_filter(element)
        elif element.element == 'nvvideoconvert':
            gst_element = NvDsElementFactory.create_nvvideoconvert(element)
        else:
            gst_element = GstElementFactory.create_element(element)
        logger.debug('Created element %r', gst_element.name)
        pipeline.add(gst_element)
        if last_gst_element is not None:
            logger.debug('Linking %r -> %r', last_gst_element.name, gst_element.name)
            if not last_gst_element.link(gst_element):
                logger.error(
                    'Failed to link %r -> %r', last_gst_element.name, gst_element.name
                )
                return False
        last_gst_element = gst_element

    with NvDsPipelineRunner(pipeline) as runner:
        while runner.is_running:
            time.sleep(0.1)
        if runner.error is not None:
            logger.error(
                'You have configured NVENC-accelerated encoding, '
                'but your device doesn\'t support NVENC for codec %r.',
                codec_name,
            )
            return False

    logger.info('Encoder for codec %r is available', codec_name)
    return True

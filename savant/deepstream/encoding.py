import time
from typing import Any, Dict

from savant.config.schema import PipelineElement
from savant.deepstream.runner import NvDsPipelineRunner
from savant.deepstream.utils.misc import get_nvvideoconvert_properties
from savant.gstreamer import Gst  # noqa:F401
from savant.gstreamer.codecs import CODEC_BY_NAME, Codec
from savant.gstreamer.element_factory import GstElementFactory
from savant.utils.logging import get_logger


def check_encoder_is_available(parameters: Dict[str, Any]) -> bool:
    """Check if encoder is available."""

    logger = get_logger(__name__)

    output_frame = parameters.get('output_frame')
    if not output_frame:
        return True

    codec = CODEC_BY_NAME[output_frame['codec']]
    if codec not in [Codec.H264, Codec.HEVC]:
        return True

    logger.info('Checking if encoder for codec %r is available', output_frame['codec'])
    pipeline: Gst.Pipeline = Gst.Pipeline.new()

    converter_props = get_nvvideoconvert_properties()
    encoder = codec.value.encoder(output_frame.get('encoder'))
    elements = [
        PipelineElement(
            'videotestsrc',
            properties={'num-buffers': 1},
        ),
        PipelineElement(
            'capsfilter',
            properties={'caps': 'video/x-raw,width=256,height=256'},
        ),
        PipelineElement(
            'nvvideoconvert',
            properties=converter_props,
        ),
        PipelineElement(
            encoder,
            properties=output_frame.get('encoder_params', {}),
        ),
    ]
    if codec in [Codec.H264, Codec.HEVC]:
        elements.append(
            PipelineElement(
                codec.value.parser,
                properties={'config-interval': -1},
            )
        )
    caps = codec.value.caps_with_params
    if codec == Codec.H264 and encoder == codec.value.sw_encoder:
        profile = output_frame.get('profile')
        if profile is None:
            profile = 'baseline'
        caps = f'{caps},profile={profile}'
    elements.extend(
        [
            PipelineElement(
                'capsfilter',
                properties={'caps': caps},
            ),
            PipelineElement('fakesink'),
        ]
    )

    last_gst_element = None
    for element in elements:
        if element.element == 'capsfilter':
            gst_element = GstElementFactory.create_caps_filter(element)
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
        while runner._is_running:
            time.sleep(0.1)
        if runner._error is not None:
            logger.error(
                'You have configured NVENC-accelerated encoding, '
                'but your device doesn\'t support NVENC.'
            )
            return False

    logger.info('Encoder for codec %r is available', output_frame['codec'])
    return True

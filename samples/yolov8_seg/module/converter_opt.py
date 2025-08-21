from typing import Any, Dict

from savant.utils.platform import is_aarch64


def converter_selector() -> str:
    converter = 'converter' if is_aarch64() else 'gpu_converter'
    return converter


def output_frame_selector() -> Dict[str, Any]:
    visualize_tag = {'tag': 'visualize'}
    codec = 'jpeg' if is_aarch64() else 'hevc'
    return {'codec': codec, 'condition': visualize_tag}

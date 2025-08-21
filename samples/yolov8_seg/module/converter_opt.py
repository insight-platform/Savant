from typing import Any, Dict

from savant.utils.platform import is_aarch64


def converter_selector() -> str:
    if is_aarch64():
        return 'converter'
    else:
        return 'gpu_converter'


def output_frame_selector() -> Dict[str, Any]:
    visualize_tag = {'tag': 'visualize'}
    if is_aarch64():
        return {'codec': 'jpeg', 'condition': visualize_tag}
    else:
        return {'codec': 'hevc', 'condition': visualize_tag}

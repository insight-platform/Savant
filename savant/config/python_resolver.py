"""JSON resolver for OmegaConf."""

import importlib
from typing import Any, Optional

from omegaconf import OmegaConf

from savant.utils.log import get_logger

logger = get_logger(__name__)


def python_resolver(  # pylint:disable=unused-argument
    module_name: str, function_name: str, *args
) -> Optional[Any]:
    """OmegaConf resolver that provides config variable value by parsing a Python
    module and function name."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.error('Python module "%s" not found', module_name)
        raise ValueError(f'Python module "{module_name}" not found')
    try:
        function = getattr(module, function_name)
    except AttributeError:
        logger.error(
            'Python configuration resolver function "%s" not found in module "%s".',
            function_name,
            module_name,
        )
        raise ValueError(
            f'Python configuration resolver function "{function_name}" not found in module "{module_name}"'
        )
    try:
        res = function(*args)
        if isinstance(res, (list, dict, tuple)):
            res = OmegaConf.create(res)
        logger.debug(
            'Python configuration resolver function "%s" in module "%s" returned %s',
            function_name,
            module_name,
            res,
        )
        return res
    except Exception as e:
        logger.error(
            'Python configuration resolver function "%s" in module "%s" raised an error: %s',
            function_name,
            module_name,
            e,
        )
        raise ValueError(
            f'Python configuration resolver function "{function_name}" in module "{module_name}" raised an error: {e}'
        )

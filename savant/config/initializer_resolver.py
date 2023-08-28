"""Initializer resolver for OmegaConf."""
import json
import os
from typing import Any

from savant.parameter_storage import STORAGE_TYPES, param_storage
from savant.utils.logging import get_logger
logger = get_logger(__name__)


def initializer_resolver(param_name: str, default_val: Any, _parent_, _root_) -> Any:
    """OmegaConf resolver that provides config variable value by polling a
    number of different value stores in order of their priority and returning
    result from the first available one.

    Example usage in config (higher number means lower priority):

    .. code-block:: yaml

        parameter_init_priority:
            environment: 20
            etcd: 10

        parameters:
            frame:
                width: ${initializer:frame_width,1280}

    where

    * ``initializer`` is the registered resolver name
    * ``frame_width`` is the example variable name
    * ``1280`` is the default value, used in case all value stores failed
      to return a result
    """
    priority_cfg_key = 'parameter_init_priority'
    if _root_[priority_cfg_key]:
        init_order = sorted(_root_[priority_cfg_key].items(), key=lambda item: item[1])

        for init_type, _ in init_order:
            try:
                if init_type == 'environment':
                    return _get_from_env(param_name)

                if init_type in STORAGE_TYPES:
                    return _get_from_storage(param_name)

            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                logger.warning(
                    'Parameter "%s" init with "%s" failed.', param_name, init_type
                )

    else:
        logger.warning(
            'Using initializer resolver, but "%s" config section is empty.',
            priority_cfg_key,
        )
    # default is last
    return default_val


def _get_from_env(param_name: str):
    """Use passed parameter name to get initial value for parameter from
    environment variables."""
    return json.loads(os.environ[param_name])


def _get_from_storage(param_name: str):
    # no default added when registering
    # so that init sources are clearly separated
    param_storage().register_parameter(param_name)
    return param_storage()[param_name]

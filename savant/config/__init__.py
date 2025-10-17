"""Module configuration."""

from omegaconf import OmegaConf

from .calc_resolver import calc_resolver
from .json_resolver import json_resolver
from .module_config import ModuleConfig
from .python_resolver import python_resolver

__all__ = ['ModuleConfig']

OmegaConf.register_new_resolver('calc', calc_resolver)
OmegaConf.register_new_resolver('json', json_resolver)
OmegaConf.register_new_resolver('py', python_resolver)

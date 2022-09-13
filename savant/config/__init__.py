"""Module configuration."""
from omegaconf import OmegaConf
from .module_config import ModuleConfig
from .initializer_resolver import initializer_resolver
from .calc_resolver import calc_resolver
from .json_resolver import json_resolver

OmegaConf.register_new_resolver('initializer', initializer_resolver)
OmegaConf.register_new_resolver('calc', calc_resolver)
OmegaConf.register_new_resolver('json', json_resolver)

"""JSON resolver for OmegaConf."""
import json
from typing import Optional, Union

from omegaconf import DictConfig, ListConfig, OmegaConf
from savant.utils.logging import get_logger
logger = get_logger(__name__)


def json_resolver(  # pylint:disable=unused-argument
    json_string: Optional[str], *args
) -> Optional[Union[DictConfig, ListConfig]]:
    """OmegaConf resolver that provides config variable value by parsing a JSON
    string."""
    logger.debug('Parsing param JSON "%s"', json_string)
    try:
        parsed_conf_node = json.loads(json_string)
    except TypeError:
        logger.warning('JSON loads fail, returning None for "%s".', json_string)
        return None

    dict_conf = OmegaConf.create(parsed_conf_node)
    logger.debug('Returning config node %s', dict_conf)
    return dict_conf

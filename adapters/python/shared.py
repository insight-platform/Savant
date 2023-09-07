import os
import time
from typing import Any, Dict

from confluent_kafka.admin import AdminClient, ClusterMetadata, NewTopic

from savant.utils.logging import get_logger

logger = get_logger(__name__)


def opt_config(name, default=None, convert=None):
    conf_str = os.environ.get(name)
    if conf_str:
        return convert(conf_str) if convert else conf_str
    return default


def kafka_topic_exists(
    brokers: str,
    topic: str,
    create_if_not_exists: bool,
    create_topic_config: Dict[str, Any],
    timeout: int = 10,
):
    admin_client = AdminClient({'bootstrap.servers': brokers})
    cluster_meta: ClusterMetadata = admin_client.list_topics()
    if topic in cluster_meta.topics:
        return True

    if not create_if_not_exists:
        raise False

    logger.info('Creating kafka topic %s with config %s', topic, create_topic_config)
    admin_client.create_topics([NewTopic(topic, **create_topic_config)])
    for _ in range(timeout):
        cluster_meta = admin_client.list_topics()
        if topic in cluster_meta.topics:
            return True
        time.sleep(1)

    logger.error('Failed to create kafka topic %s', topic)

    return False

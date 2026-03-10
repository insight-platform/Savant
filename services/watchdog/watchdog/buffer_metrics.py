from typing import Dict, Optional

import aiohttp
from prometheus_client.openmetrics.parser import text_string_to_metric_families


async def get_metrics(buffer_url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(f'http://{buffer_url}/metrics') as response:
            content = await response.text()
            return content


async def parse_metrics(
    content: str,
    label_filters: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, float]:
    """Parse Prometheus / OpenMetrics text into ``{metric_name: value}``.

    When a metric has multiple samples (e.g. different label sets), the
    behaviour depends on *label_filters*:

    * If *label_filters* contains an entry for the metric name, only
      samples whose labels are a superset of the specified key-value
      pairs are considered.
    * Otherwise **all** samples are collected and the **maximum** value
      is returned for that metric name.
    """

    collected: Dict[str, list] = {}
    for family in text_string_to_metric_families(content):
        for sample in family.samples:
            name = sample.name
            labels = sample.labels
            value = float(sample.value)

            if label_filters and name in label_filters:
                required = label_filters[name]
                if not all(labels.get(k) == v for k, v in required.items()):
                    continue

            collected.setdefault(name, []).append(value)

    return {name: max(values) for name, values in collected.items()}

from typing import Dict

import aiohttp
from prometheus_client.parser import text_string_to_metric_families


async def get_metrics(buffer_url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(f'http://{buffer_url}/metrics') as response:
            content = await response.text()
            return content


async def parse_metrics(content: str) -> Dict[str, float]:
    metrics = {}
    for family in text_string_to_metric_families(content):
        for sample in family.samples:
            metrics[sample[0]] = float(sample[2])
    return metrics

from typing import List, Union

from savant_rs.metrics import CounterFamily, GaugeFamily, delete_metric_family


def get_or_create_counter(
    name: str,
    description: str = None,
    label_names: Union[List[str], None] = None,
    unit: Union[str, None] = None,
) -> CounterFamily:
    if label_names is None:
        label_names = []
    return CounterFamily.get_or_create_counter_family(
        name, description, label_names, unit
    )


def get_or_create_gauge(
    name: str,
    description: str = None,
    label_names: Union[List[str], None] = None,
    unit: Union[str, None] = None,
) -> GaugeFamily:
    if label_names is None:
        label_names = []
    return GaugeFamily.get_or_create_gauge_family(name, description, label_names, unit)


def get_counter(name: str) -> CounterFamily:
    return CounterFamily.get_counter_family(name)


def get_gauge(name: str) -> GaugeFamily:
    return GaugeFamily.get_gauge_family(name)


def delete(name: str):
    delete_metric_family(name)

from .schema import Config


def validate(config: Config):
    if any(
        [not w.queue and not w.ingress and not w.egress for w in config.watch_configs]
    ):
        raise ValueError(
            'Watch config must include at least one of the following: '
            'queue, ingress, or egress.'
        )

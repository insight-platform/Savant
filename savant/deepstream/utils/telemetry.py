from savant_rs.telemetry import (
    ClientTlsConfig,
    ContextPropagationFormat,
    Identity,
    Protocol,
    TelemetryConfiguration,
    TracerConfiguration,
    init,
    shutdown,
)

from savant.config.schema import TracingParameters
from savant.utils.logging import get_logger

logger = get_logger(__name__)

PROTOCOLS = {
    'grpc': Protocol.Grpc,
    'http_binary': Protocol.HttpBinary,
    'http_json': Protocol.HttpJson,
}


def init_tracing(module_name: str, tracing: TracingParameters):
    """Initialize tracing provider."""

    provider_params = tracing.provider_params or {}
    if tracing.provider == 'jaeger':
        service_name = provider_params.get('service_name', module_name)

        endpoint = provider_params.get('endpoint')
        if not endpoint:
            raise ValueError(
                'Jaeger endpoint is not specified. Please specify it in the config file.'
            )

        protocol_name = provider_params.get('protocol') or 'grpc'
        try:
            protocol = PROTOCOLS[protocol_name]
        except KeyError:
            raise ValueError(f'Unknown tracing protocol: {protocol_name}')

        tls = provider_params.get('tls')
        if tls:
            tls_identity = tls.get('identity')
            if tls_identity:
                tls_identity_config = Identity(
                    key=tls_identity.get('key'),
                    certificate=tls_identity.get('certificate'),
                )
            else:
                tls_identity_config = None
            tls_config = ClientTlsConfig(
                certificate=tls.get('certificate'),
                identity=tls_identity_config,
            )
        else:
            tls_config = None

        timeout = provider_params.get('timeout')
        if timeout is not None:
            timeout = int(timeout)

        logger.info(
            'Initializing Jaeger tracer with service name %r and endpoint %r. Protocol: %s.',
            service_name,
            endpoint,
            protocol,
        )
        tracer_config = TracerConfiguration(
            service_name=service_name,
            protocol=protocol,
            endpoint=endpoint,
            tls=tls_config,
            timeout=timeout,
        )
        config = TelemetryConfiguration(
            # TODO: support W3C
            context_propagation_format=ContextPropagationFormat.Jaeger,
            tracer=tracer_config,
        )
        init(config)

    elif tracing.provider is not None:
        raise ValueError(f'Unknown tracing provider: {tracing.provider}')

    else:
        logger.info('No tracing provider specified. Using noop tracer.')
        init(TelemetryConfiguration.no_op())


def shutdown_tracing():
    """Shutdown tracing provider."""
    shutdown()

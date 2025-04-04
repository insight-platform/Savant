from savant_rs.telemetry import (
    ClientTlsConfig,
    ContextPropagationFormat,
    Identity,
    Protocol,
    TelemetryConfiguration,
    TracerConfiguration,
    init,
    init_from_file,
    shutdown,
)

from savant.config.schema import TracingParameters
from savant.utils.log import get_logger

logger = get_logger(__name__)

PROTOCOLS = {
    'grpc': Protocol.Grpc,
    'http_binary': Protocol.HttpBinary,
    'http_json': Protocol.HttpJson,
}


def init_tracing(module_name: str, tracing: TracingParameters):
    """Initialize tracing provider."""
    if tracing.provider == 'jaeger' or tracing.provider == 'opentelemetry':
        if tracing.provider == 'jaeger':
            # TODO: remove jaeger keyword support in Savant 0.6
            logger.warning(
                'Jaeger provider is deprecated. Use "opentelemetry" instead.'
            )
        else:
            logger.info('Using OpenTelemetry provider.')

        provider_params_config = tracing.provider_params_config
        provider_params = tracing.provider_params or {}
        if provider_params_config:
            logger.info(
                'Initializing tracing provider from JSON file %r.',
                provider_params_config,
            )
            if provider_params:
                logger.warning(
                    'Provider params from config attributes will be ignored because JSON file is specified.'
                )
            init_from_file(str(provider_params_config))
        else:
            logger.info('Initializing tracing provider from config attributes.')

            service_name = provider_params.get('service_name', module_name)

            endpoint = provider_params.get('endpoint')
            if not endpoint:
                raise ValueError(
                    'OpenTelemetry endpoint is not specified. Please specify it in the config file.'
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

                ca = tls.get('ca')
                if ca is None:
                    ca = tls.get(
                        'certificate'
                    )  # deprecated, for compatibility reasons.
                    # TODO: remove in Savant 0.6
                    if ca:
                        logger.warning(
                            'The "tls.certificate" key is deprecated. Use "tls.ca" instead.'
                        )

                tls_config = ClientTlsConfig(
                    ca=ca,
                    identity=tls_identity_config,
                )
            else:
                tls_config = None

            timeout = provider_params.get('timeout')
            if timeout is not None:
                timeout = int(timeout)

            logger.info(
                'Initializing OpenTelemetry tracer with service name %r and endpoint %r. Protocol: %s.',
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
                context_propagation_format=ContextPropagationFormat.W3C,
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

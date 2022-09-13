"""Avro helper functions."""
from io import BytesIO
from typing import Tuple

from fastavro import schemaless_reader, schemaless_writer
from .avro_schema_registry import (
    DECODING_REGISTRY,
    ENVELOPE_SCHEMA,
    SIGNATURE_TO_NAME,
    SignedSchema,
)


def __serialize(schema: dict, message: dict) -> bytes:
    byte_stream = BytesIO()
    schemaless_writer(byte_stream, schema, message)
    byte_stream.seek(0)
    return byte_stream.read()


def serialize(payload_schema: SignedSchema, message: dict) -> bytes:
    """Serialize a message according to schema."""
    payload_bin = __serialize(payload_schema.parsed, message)
    envelope = {'schema_sig': payload_schema.sig, 'payload': payload_bin}
    return __serialize(ENVELOPE_SCHEMA, envelope)


def __deserialize(schema: dict, data: bytes) -> dict:
    bytes_stream = BytesIO()
    bytes_stream.write(data)
    bytes_stream.seek(0)
    return schemaless_reader(bytes_stream, schema)


def deserialize(data: bytes) -> Tuple[str, dict]:
    """Deserialize message envelope."""
    envelope = __deserialize(ENVELOPE_SCHEMA, data)
    schema_sig = envelope['schema_sig']
    schema_name = SIGNATURE_TO_NAME[schema_sig]
    payload_schema = DECODING_REGISTRY[schema_sig]
    message = __deserialize(payload_schema, envelope['payload'])
    return schema_name, message

"""Avro schemas registry."""
from typing import Dict, NamedTuple
import pathlib
from fastavro.schema import load_schema, to_parsing_canonical_form, fingerprint


class SignedSchema(NamedTuple):
    """A named tuple for avro schema + its fingerprint."""

    parsed: dict
    """Parsed avro schema as returned by fastavro.schema.load_schema()"""

    # pylint:disable=line-too-long
    sig: bytes
    """Schema fingerprint generated based on canonical representation.

    .. note::

        See `Avro documentation <https://avro.apache.org/docs/current/spec.html#schema_fingerprints>`_
        for details.
    """


def get_signed_schema(schema_path: pathlib.Path) -> SignedSchema:
    """Load schema and generate schema fingerprint."""
    parsed_schema = load_schema(schema_path)
    canonical = to_parsing_canonical_form(parsed_schema)
    sig_hex = fingerprint(canonical, 'CRC-64-AVRO')
    return SignedSchema(parsed_schema, bytes.fromhex(sig_hex))


base_path = pathlib.Path(__file__).parent / 'avro-schemas'
payload_path = base_path / 'payload'
envelope_schema_path = base_path / 'vision.module.FrameEnvelope.avsc'

ENVELOPE_SCHEMA: dict = load_schema(envelope_schema_path)

ENCODING_REGISTRY: Dict[str, SignedSchema] = {}
"""{name: SignedSchema}"""

DECODING_REGISTRY: Dict[bytes, dict] = {}
"""{schema_fingerprint: parsed_schema}"""

SIGNATURE_TO_NAME: Dict[bytes, str] = {}
"""{schema_fingerprint: schema_name}"""


for payload in payload_path.iterdir():
    if payload.is_dir():
        schema_filename = f'vision.module.{payload.name}.avsc'
        signed_schema = get_signed_schema(payload / schema_filename)
        ENCODING_REGISTRY[payload.name] = signed_schema
        DECODING_REGISTRY[signed_schema.sig] = signed_schema.parsed
        SIGNATURE_TO_NAME[signed_schema.sig] = payload.name

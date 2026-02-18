# Testing Framework Notes

## Savant-rs API Discrepancies

### BlockingWriter.send_message

- **Observed**: The Python API requires 3 arguments: `send_message(topic, message, extra)` where `extra` is `bytes`.
- **PyI (zmq.pyi)**: Declares only 2 arguments: `send_message(topic, message)`.
- **Impact**: The stub is incomplete; callers must pass `b""` for frames with internal content.

### BlockingWriter.send_eos

- **Observed**: `send_eos(source_id, topic=None)` - topic defaults to source_id.
- **PyI**: Shows `send_eos(topic: str)` - parameter naming differs (source_id vs topic).

### MatchQuery.label

- **Observed**: Requires `StringExpression`, not a plain string. Use `MatchQuery.label(StringExpression.eq('person'))` not `MatchQuery.label('person')`.

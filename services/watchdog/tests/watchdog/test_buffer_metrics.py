from textwrap import dedent
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from aiohttp import ClientResponse
from watchdog.buffer_metrics import get_metrics, parse_metrics


@pytest.mark.asyncio
@mock.patch('aiohttp.ClientSession', new_callable=MagicMock)
async def test_get_metrics(session_mock: MagicMock):
    session = session_mock()
    response_mock = MagicMock(ClientResponse)
    response = response_mock()

    session_in_with = session.__aenter__.return_value
    session_in_with.get = response_mock
    response_in_with: AsyncMock = response.__aenter__.return_value
    response_in_with.text = AsyncMock(return_value='content')

    result = await get_metrics('localhost:8080')

    assert result == 'content'
    assert response_mock.call_count == 2
    assert response_mock.call_args_list[0] == call()  # initial call in test itself
    assert response_mock.call_args_list[1] == call('http://localhost:8080/metrics')


@pytest.mark.asyncio
async def test_get_metrics_session_exception():
    with mock.patch('aiohttp.ClientSession', side_effect=RuntimeError('error')):
        with pytest.raises(RuntimeError, match='error'):
            await get_metrics('localhost:8080')


@pytest.mark.asyncio
@mock.patch('aiohttp.ClientSession', new_callable=MagicMock)
async def test_get_metrics_response_exception(session_mock):
    with pytest.raises(RuntimeError, match='error'):
        session = session_mock()
        response_mock = MagicMock(ClientResponse, side_effect=RuntimeError('error'))

        session_in_with = session.__aenter__.return_value
        session_in_with.get = response_mock

        await get_metrics('localhost:8080')


@pytest.mark.asyncio
async def test_parse_metrics_openmetrics():
    """Counters use _total suffix; gauges are bare names."""
    content = dedent("""\
        # HELP received_messages Number of messages received by the adapter
        # TYPE received_messages counter
        received_messages_total{adapter="buffer"} 120.0
        # HELP pushed_messages Number of messages pushed to the buffer
        # TYPE pushed_messages counter
        pushed_messages_total{adapter="buffer"} 34.0
        # EOF
    """)
    result = await parse_metrics(content)
    assert result == {
        'received_messages_total': 120.0,
        'pushed_messages_total': 34.0,
    }


@pytest.mark.asyncio
async def test_parse_metrics_gauge():
    content = dedent("""\
        # HELP last_sent_message Timestamp of last sent message
        # TYPE last_sent_message gauge
        last_sent_message{reason="send_success"} 1720441634.544
        # EOF
    """)
    result = await parse_metrics(content)
    assert result == {'last_sent_message': 1720441634.544}


@pytest.mark.asyncio
async def test_parse_metrics_max_aggregation():
    """Multiple samples for the same metric name are aggregated with max()."""
    content = dedent("""\
        # HELP last_sent_message Timestamp of last sent message
        # TYPE last_sent_message gauge
        last_sent_message{reason="send_success"} 1720441600.0
        last_sent_message{reason="ack_success"} 1720441634.544
        # EOF
    """)
    result = await parse_metrics(content)
    assert result == {'last_sent_message': 1720441634.544}


@pytest.mark.asyncio
async def test_parse_metrics_max_aggregation_stale_label():
    """Stale label value should not shadow the fresh one."""
    content = dedent("""\
        # HELP last_sent_message Timestamp of last sent message
        # TYPE last_sent_message gauge
        last_sent_message{reason="ack_success"} 1720441634.544
        last_sent_message{reason="send_success"} 0.0
        # EOF
    """)
    result = await parse_metrics(content)
    # max() picks the fresh timestamp, not the stale 0.0
    assert result == {'last_sent_message': 1720441634.544}


@pytest.mark.asyncio
async def test_parse_metrics_label_filter():
    """When label_filters is set, only matching samples are considered."""
    content = dedent("""\
        # HELP last_sent_message Timestamp of last sent message
        # TYPE last_sent_message gauge
        last_sent_message{reason="send_success",adapter="buffer"} 100.0
        last_sent_message{reason="ack_success",adapter="buffer"} 200.0
        # EOF
    """)
    result = await parse_metrics(
        content,
        label_filters={'last_sent_message': {'reason': 'send_success'}},
    )
    assert result == {'last_sent_message': 100.0}


@pytest.mark.asyncio
async def test_parse_metrics_label_filter_no_match():
    """When filter matches nothing, the metric is absent from results."""
    content = dedent("""\
        # HELP last_sent_message Timestamp of last sent message
        # TYPE last_sent_message gauge
        last_sent_message{reason="send_success"} 100.0
        # EOF
    """)
    result = await parse_metrics(
        content,
        label_filters={'last_sent_message': {'reason': 'ack_success'}},
    )
    assert 'last_sent_message' not in result


@pytest.mark.asyncio
async def test_parse_metrics_label_filter_unfiltered_metrics_use_max():
    """Metrics not mentioned in label_filters still use max() aggregation."""
    content = dedent("""\
        # HELP buffer_size Number of messages in the buffer
        # TYPE buffer_size gauge
        buffer_size{adapter="a"} 10.0
        buffer_size{adapter="b"} 20.0
        # HELP last_sent_message Timestamp of last sent message
        # TYPE last_sent_message gauge
        last_sent_message{reason="send_success"} 100.0
        # EOF
    """)
    result = await parse_metrics(
        content,
        label_filters={'last_sent_message': {'reason': 'send_success'}},
    )
    # buffer_size is not in label_filters -> max()
    assert result['buffer_size'] == 20.0
    assert result['last_sent_message'] == 100.0


@pytest.mark.asyncio
async def test_parse_metrics_invalid_content_type():
    with pytest.raises(
        TypeError,
        match='initial_value must be str or None, not int',
    ):
        await parse_metrics(123)  # type: ignore

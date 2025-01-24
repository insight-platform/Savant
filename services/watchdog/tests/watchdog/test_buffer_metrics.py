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
@pytest.mark.parametrize(
    'content, expected',
    [
        (
            '''
            # HELP received_messages_total Number of messages received by the adapter
            # TYPE received_messages_total counter
            received_messages_total{adapter="buffer"} 120.0
            # HELP pushed_messages_total Number of messages pushed to the buffer
            # TYPE pushed_messages_total counter
            pushed_messages_total{adapter="buffer"} 34.0 1720441634544
            ''',
            {'received_messages_total': 120.0, 'pushed_messages_total': 34.0},
        ),
    ],
)
async def test_parse_metrics(content, expected):
    result = await parse_metrics(content)

    assert result == expected


@pytest.mark.asyncio
async def test_parse_metrics_invalid_content_type():
    with pytest.raises(
        TypeError,
        match='initial_value must be str or None, not int',
    ):
        await parse_metrics(123)  # type: ignore

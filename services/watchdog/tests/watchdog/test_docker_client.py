from unittest.mock import AsyncMock, Mock, call, patch

import pytest
import pytest_asyncio
from aiodocker import DockerError
from aiodocker.containers import DockerContainer, DockerContainers
from watchdog.run import DockerClient

DOCKER_ERROR = DockerError('status', {'message': 'error'})
RUNTIME_ERROR = RuntimeError('Test error')


@pytest_asyncio.fixture
def docker_mock():
    with patch('aiodocker.Docker', autospec=True) as mock:
        yield mock()


@pytest.mark.asyncio
async def test_get_containers(docker_mock):
    label1 = ['label1']
    label2 = ['label2', 'label3']
    containers_for_label1 = [Mock(), Mock()]
    containers_for_label2 = [Mock()]

    containers = Mock(
        DockerContainers,
        list=AsyncMock(side_effect=[containers_for_label1, containers_for_label2]),
    )
    docker_mock.containers = containers

    client = DockerClient()

    result = await client.get_containers([label1, label2])

    assert result == containers_for_label1 + containers_for_label2
    assert containers.list.call_count == 2
    assert containers.list.call_args_list[0] == call(
        all=True, filters={'label': label1}
    )
    assert containers.list.call_args_list[1] == call(
        all=True, filters={'label': label2}
    )


@pytest.mark.asyncio
async def test_get_containers_empty_labels(docker_mock):
    containers = Mock(DockerContainers, list=AsyncMock(side_effect=[]))
    docker_mock.containers = containers

    client = DockerClient()

    result = await client.get_containers([])

    assert result == []
    assert containers.list.call_count == 0


@pytest.mark.asyncio
async def test_get_containers_docker_error(docker_mock):
    containers = Mock(
        DockerContainers,
        list=AsyncMock(side_effect=DOCKER_ERROR),
    )
    docker_mock.containers = containers

    client = DockerClient()

    with pytest.raises(
        RuntimeError, match="Failed to list containers with labels \\['label1'\\]"
    ):
        await client.get_containers([['label1']])


@pytest.mark.asyncio
async def test_get_containers_exception(docker_mock):
    containers = Mock(DockerContainers, list=AsyncMock(side_effect=RUNTIME_ERROR))
    docker_mock.containers = containers

    client = DockerClient()

    with pytest.raises(RUNTIME_ERROR.__class__, match=str(RUNTIME_ERROR)):
        await client.get_containers([['label1']])


@pytest.mark.asyncio
async def test_close(docker_mock):
    client = DockerClient()

    await client.close()

    docker_mock.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_restart_container():
    container = AsyncMock(DockerContainer)

    await DockerClient.restart_container(container)

    container.restart.assert_awaited_once()


@pytest.mark.asyncio
async def test_restart_container_docker_error():
    container = Mock(
        DockerContainer,
        restart=AsyncMock(side_effect=DOCKER_ERROR),
    )

    await DockerClient.restart_container(container)

    container.restart.assert_awaited_once()


@pytest.mark.asyncio
async def test_restart_container_exception():
    container = Mock(DockerContainer, restart=AsyncMock(side_effect=RUNTIME_ERROR))

    with pytest.raises(RUNTIME_ERROR.__class__, match=str(RUNTIME_ERROR)):
        await DockerClient.restart_container(container)


@pytest.mark.asyncio
async def test_stop_container():
    container = AsyncMock(DockerContainer)

    await DockerClient.stop_container(container)

    container.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_container_docker_error():
    container = Mock(
        DockerContainer,
        stop=AsyncMock(side_effect=DOCKER_ERROR),
    )

    await DockerClient.stop_container(container)

    container.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_container_exception():
    container = Mock(DockerContainer, stop=AsyncMock(side_effect=RUNTIME_ERROR))

    with pytest.raises(RUNTIME_ERROR.__class__, match=str(RUNTIME_ERROR)):
        await DockerClient.stop_container(container)

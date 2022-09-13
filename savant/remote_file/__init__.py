"""Remote file management."""
from typing import Union
from pathlib import Path
import logging
from omegaconf import DictConfig, OmegaConf
from savant.remote_file.schema import RemoteFile
from savant.remote_file.base import RemoteFileManagerType, RemoteFileError
from savant.remote_file.http import HTTPFileHandler
from savant.remote_file.s3 import S3FileHandler
from savant.remote_file.utils import unpack_archive, read_file_checksum


__all__ = ['process_remote', 'RemoteFile']


RemoteFileManagerType.add_handler(S3FileHandler)
RemoteFileManagerType.add_handler(HTTPFileHandler)
logger = logging.getLogger(__name__)


def process_remote(
    remote: Union[DictConfig, RemoteFile], download_path: Path, dst_path: Path
):
    """Processes a remote archive file. Checks the relevance of local files
    (files are not relevant if the name, checksum or content has changed).
    Downloads, unpacks and moves the archive files to the specified location.

    :param remote: Remote file configuration
    :param download_path: Path to download remote files and cache them
        to check if they are up-to-date
    :param dst_path: Location of resulting (unpacked) files
    :return:
    """
    remote_config = OmegaConf.merge(RemoteFile(), remote)

    download_path.mkdir(parents=True, exist_ok=True)

    # to check if there are any differences between current and saved
    remote_config_file_path = download_path / 'remote.yaml'

    dst_path.mkdir(parents=True, exist_ok=True)

    handler = RemoteFileManagerType.find_handler(remote_config)
    if not handler:
        raise RemoteFileError(
            f'Remote file handler for url "{remote_config.url}" not found.'
        )

    def download(url: str) -> Path:
        try:
            logger.info('Downloading %s...', url)
            return handler.download(url, download_path)
        except Exception as exc:
            raise RemoteFileError(
                f'Error downloading remote file {url}. {exc}'
            ) from exc

    def update(checksum: bool = True):
        if checksum and remote_config.checksum_url:
            download(remote_config.checksum_url)

        file_path = download(remote_config.url)

        try:
            file_list = unpack_archive(file_path, dst_path)
        except Exception as exc:
            raise RemoteFileError(
                f'Error unpacking downloaded file {file_path}. {exc}'
            ) from exc

        OmegaConf.save(remote, remote_config_file_path)

        logger.info(
            'Remote file "%s" has been downloaded and unpacked. '
            'Files %s have been placed in "%s".',
            file_path,
            file_list,
            dst_path,
        )

    # destination folder is empty
    if not any(dst_path.iterdir()):
        logger.info(
            'Destination folder "%s" is empty, remote file will be downloaded.',
            dst_path,
        )
        return update()

    # download folder is empty
    if not any(download_path.iterdir()):
        logger.info(
            'Downloads folder "%s" is empty, there is no file to check, '
            'remote file will be downloaded.',
            download_path,
        )
        return update()

    # remote configuration was changed
    if remote_config_file_path.is_file():
        saved_remote_config = OmegaConf.load(remote_config_file_path)
        # cache and check the "user" remote config rather than the "processed" one
        # to prevent loading when the remote config schema changes
        if remote != saved_remote_config:
            logger.info(
                'Remote file configuration was changed, '
                'remote file will be downloaded.'
            )
            return update()

        if not remote_config.always_check:
            return

    # checksum validation
    if remote_config.checksum_url:
        logger.info('Verifying checksum of the remote file...')

        saved_checksum = None
        checksum_file_path = download_path / handler.get_file_name(
            remote_config.checksum_url
        )
        if checksum_file_path.is_file():
            saved_checksum = read_file_checksum(checksum_file_path)

        checksum_file_path = download(remote_config.checksum_url)
        checksum = read_file_checksum(checksum_file_path)
        if checksum != saved_checksum:
            logger.info(
                'Checksum of the remote file mismatch, remote file will be downloaded.'
            )
            return update(checksum=False)
        logger.info('Checksum of the remote file is valid. Remote file is up-to-date.')

    # content validation - must download, so update
    else:
        logger.info(
            'Remote file will be downloaded to check if the local files are up to date.'
        )
        update()

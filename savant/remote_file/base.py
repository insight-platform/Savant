"""Base classes for remote file managing."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import FrozenSet, Optional, Set, Type, Union
from urllib.parse import urlparse

from omegaconf import DictConfig

from savant.remote_file.schema import RemoteFile
from savant.utils.logging import get_logger

__all__ = ['RemoteFileManagerType', 'RemoteFileHandler', 'RemoteFileError']


class RemoteFileError(Exception):
    """General remote file exception class."""


class RemoteFileHandler(ABC):
    """Base remote file handler defines the minimum functionality required to
    download a remote file, specified using a URL and optional parameters."""

    supported_schemes: FrozenSet[str]

    def __init__(self, **params):
        self.logger = get_logger(self.__class__.__name__)
        self.params = params

    @staticmethod
    def get_file_name(url: str):
        """Get file name from url."""
        parsed_url = urlparse(url)
        return Path(parsed_url.path).name

    @abstractmethod
    def download(self, url: str, dst_path: Path) -> Path:
        """Downloads a file from a given remote url to a destination path. If
        file exists, will overwrite the existing file.

        :return: Path to the downloaded file
        """


class RemoteFileHandlerManager:
    """Manager registers handlers for remote files with different
    schemes/protocols and allows you to find a suitable handler for the
    specified remote configuration."""

    _handlers: Set[Type[RemoteFileHandler]] = set()

    @classmethod
    def add_handler(cls, handler: Type[RemoteFileHandler]):
        """Add a remote file handler."""
        cls._handlers.add(handler)

    @classmethod
    def remove_handler(cls, handler: Type[RemoteFileHandler]):
        """Remove a remote file handler."""
        cls._handlers.remove(handler)

    @classmethod
    def find_handler(
        cls, remote: Union[DictConfig, RemoteFile]
    ) -> Optional[RemoteFileHandler]:
        """Try to find a handler for remote."""
        parsed_url = urlparse(remote.url)
        if remote.checksum_url:
            parsed_checksum_url = urlparse(remote.checksum_url)
            if parsed_checksum_url.scheme != parsed_url.scheme:
                raise RemoteFileError(
                    'URL schemes of checksum file and main file are different.'
                )

        for handler in cls._handlers:
            if parsed_url.scheme in handler.supported_schemes:
                return handler(**remote.parameters)
        return None


RemoteFileManagerType = RemoteFileHandlerManager

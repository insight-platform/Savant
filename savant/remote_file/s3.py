"""S3 remote file handling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from savant.remote_file.base import RemoteFileError, RemoteFileHandler

__all__ = ['S3FileHandler']


class S3RemoteFileError(RemoteFileError):
    """S3 remote file exception class."""


@dataclass
class S3Config:
    """S3 storage configuration."""

    access_key: Optional[str] = None
    """S3 Access Key ID."""

    secret_key: Optional[str] = None
    """S3 Secret Access Key."""

    endpoint: Optional[str] = None
    """Endpoint URL to access S3."""

    region: Optional[str] = None
    """S3 remote region."""


class S3FileHandler(RemoteFileHandler):
    """File handler for S3/S3-like storages."""

    supported_schemes = frozenset(['s3'])

    def __init__(self, **params):
        super().__init__(**params)
        try:
            config = S3Config(**params)
            if config.access_key and config.secret_key:
                self.client = boto3.client(
                    's3',
                    aws_access_key_id=config.access_key,
                    aws_secret_access_key=config.secret_key,
                    endpoint_url=config.endpoint,
                    region_name=config.region,
                )
            else:
                self.client = boto3.client(
                    's3',
                    config=Config(signature_version=UNSIGNED),
                    endpoint_url=config.endpoint,
                    region_name=config.region,
                )
        except Exception as exc:
            raise S3RemoteFileError(
                f'Error initializing remote S3 file client. {exc}'
            ) from exc

    def download(self, url: str, dst_path: Path) -> Path:
        """Downloads a file from S3 storage."""
        bucket, key = url.split('://')[-1].split('/', 1)
        dst_file_path = dst_path / self.get_file_name(url)

        meta_data = self.client.head_object(Bucket=bucket, Key=key)
        file_size = int(meta_data.get('ContentLength'))
        with tqdm(total=file_size) as pbar:
            with open(dst_file_path, 'wb') as res:
                self.client.download_fileobj(
                    Bucket=bucket, Key=key, Fileobj=res, Callback=pbar.update
                )

        return dst_file_path

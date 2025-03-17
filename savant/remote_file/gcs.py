"""GCS remote file handling."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

from google.cloud import storage
from google.oauth2 import service_account
from tqdm import tqdm

from .base import RemoteFileError, RemoteFileHandler

__all__ = ['GCSFileHandler']


class GCSRemoteFileError(RemoteFileError):
    """GCS remote file exception class."""


@dataclass
class GCSConfig:
    """GCS storage configuration."""

    credentials_json_path: Optional[Path] = None
    """Path to service account JSON key file."""

    credentials_json: Optional[str] = None
    """Service account JSON key file content."""

    project_id: Optional[str] = None
    """Google Cloud project ID."""

    anonymous: bool = False
    """Whether to use anonymous/public access."""


class GCSFileHandler(RemoteFileHandler):
    """File handler for Google Cloud Storage."""

    supported_schemes = frozenset(['gs', 'gcs'])

    def __init__(self, **params):
        super().__init__(**params)
        try:
            config = GCSConfig(**params)

            if config.anonymous:
                self.client = storage.Client.create_anonymous_client()
            elif config.credentials_json_path:
                credentials = service_account.Credentials.from_service_account_file(
                    config.credentials_json_path
                )
                self.client = storage.Client(
                    credentials=credentials, project=config.project_id
                )
            elif config.credentials_json:
                credentials_info = json.loads(config.credentials_json)
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_info
                )
                self.client = storage.Client(
                    credentials=credentials, project=config.project_id
                )
            else:
                # Use default credentials from environment
                self.client = storage.Client(project=config.project_id)
        except Exception as exc:
            raise GCSRemoteFileError(
                f'Error initializing remote GCS file client. {exc}'
            ) from exc

    def download(self, url: str, dst_path: Path) -> Path:
        """Downloads a file from GCS storage."""
        # Parse the URL to extract bucket and key
        # Support both gs:// and gcs:// schemes
        url_parts = url.split('://')
        if len(url_parts) != 2:
            raise GCSRemoteFileError(f'Invalid GCS URL format: {url}')

        path_parts = url_parts[1].split('/', 1)
        if len(path_parts) != 2:
            raise GCSRemoteFileError(f'Invalid GCS URL format: {url}')

        bucket_name, blob_path = path_parts
        dst_file_path = dst_path / self.get_file_name(url)

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            blob.reload()
            file_size = blob.size

            with tqdm(
                total=file_size,
                unit='B',
                unit_scale=True,
            ) as pbar:
                with open(dst_file_path, 'wb') as file_obj:
                    blob.download_to_file(file_obj)
                    pbar.update(file_size)

            return dst_file_path
        except Exception as exc:
            raise GCSRemoteFileError(
                f'Error downloading file from GCS: {url}. {exc}'
            ) from exc

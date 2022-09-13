"""HTTP file handler."""
from typing import Optional
from pathlib import Path
import shutil
from tqdm import tqdm
import requests
from savant.remote_file.base import RemoteFileHandler

__all__ = ['HTTPFileHandler']


class HTTPFileHandler(RemoteFileHandler):
    """HTTP/HTTPS/FTP remote file handler."""

    supported_schemes = frozenset(('http', 'https', 'ftp'))

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        super().__init__(username=username, password=password)
        self.auth = None
        if 'username' in self.params and self.params['username']:
            self.auth = (
                self.params['username'],
                self.params['password'] if 'password' in self.params else None,
            )

    def download(self, url: str, dst_path: Path) -> Path:
        """Downloads a file using HTTP request.

        Username and password will be used to make a request using HTTP
        Basic auth.
        """
        dst_file_path = dst_path / self.get_file_name(url)

        with requests.get(url, stream=True, auth=self.auth) as req:
            req.raise_for_status()
            file_size = int(req.headers.get('Content-Length'))
            with tqdm.wrapattr(req.raw, 'read', total=file_size) as raw:
                with open(dst_file_path, 'wb') as res:
                    shutil.copyfileobj(raw, res)

        return dst_file_path

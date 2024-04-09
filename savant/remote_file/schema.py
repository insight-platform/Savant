"""Remote file schema definition."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import MISSING


@dataclass
class RemoteFile:
    """Remote file location specification."""

    url: str = MISSING
    """File URL.
    Format: ``scheme://[netloc/]path[?query]``, eg ``s3://models/yolov4.tar.gz``
    """

    checksum_url: Optional[str] = None
    """File checksum URL.
    If the URL is specified, it will be used to check the relevance of the file.
    The first line of the checksum file must contain the md5 hex digest.
    """

    always_check: bool = False
    """If True: always download and and check the remote file (or checksum).
    If False: download the remote file only if the content of the cached config 
    does not match the current config.
    """

    parameters: Dict[str, Any] = field(default_factory=dict)
    """File storage-specific parameters, eg user/password for http
    or endpoint/access_key/secret_key for s3.
    """

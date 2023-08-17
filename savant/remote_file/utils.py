"""Remote file utils."""
import hashlib
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional


def get_file_checksum(file_path: Path) -> str:
    """Generates md5 checksum for a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def read_file_checksum(checksum_file_path: Path) -> str:
    """Reads a checksum from a file.

    The first line of the file must contain md5 hex digest, eg
    "1cb835ff6a749a214f986bf9d0a3bbb7  file_name.ext\\n"
    """
    with open(checksum_file_path, 'r', encoding='utf8') as file:
        return file.readline().strip().split()[0]


def detect_archive_format(file_path: Path) -> Optional[str]:
    """Attempts to detect file format from the content by reading the first
    bytes. Used signatures from
    https://www.garykessler.net/library/file_sigs.html.

    :param file_path:  Archive file to detect
    :return: Archive format
    """

    # formats that shutil.unpack_archive() supports
    archive_format_signatures = {
        # ('gztar', ['.tar.gz', '.tgz'], "gzip'ed tar-file")
        # '\x1f\x8b' https://tools.ietf.org/html/rfc1952#page-6
        'gztar': b'\x1f\x8b\x08',
        # ('bztar', ['.tar.bz2', '.tbz2'], "bzip2'ed tar-file")
        'bztar': b'\x42\x5a\x68',
        # ('xztar', ['.tar.xz', '.txz'], "xz'ed tar-file")
        # https://tukaani.org/xz/xz-file-format.txt
        'xztar': b'\xfd\x37\x7a\x58\x5a\x00',
        # ('zip', ['.zip'], 'ZIP file')
        'zip': b'\x50\x4b\x03\x04',
    }

    with open(file_path, 'rb') as file:
        first_bytes = file.read(max(len(x) for x in archive_format_signatures.values()))

    for archive_format, signature in archive_format_signatures.items():
        if signature == first_bytes[: len(signature)]:
            return archive_format

    return None


def unpack_archive(file_path: Path, dst_path: Path) -> Optional[List[str]]:
    """Unpacks archive file to destination folder.

    Returns list of unpacked files.
    """
    # use `shutil.unpack_archive()`, but we need to get a list of files in the archive,
    # so detect format and use for this zip/tar directly
    archive_format = None
    for name, extensions, _ in shutil.get_unpack_formats():
        for extension in extensions:
            if str(file_path).endswith(extension):
                archive_format = name

    if not archive_format:
        # TODO: Use `file --mime-type`
        #  https://en.wikipedia.org/wiki/List_of_archive_formats
        # try to determine the archive format from the content
        archive_format = detect_archive_format(file_path)

    if not archive_format:
        raise ValueError(f'Unknown archive format "{file_path}".')

    if archive_format == 'zip':
        with zipfile.ZipFile(file_path) as zip_obj:
            file_list = zip_obj.namelist()

    else:
        with tarfile.open(file_path) as tar_obj:
            file_list = [tarinfo.name for tarinfo in tar_obj.getmembers()]

    if not file_list:
        raise FileNotFoundError(f'Empty archive "{file_path}".')

    shutil.unpack_archive(str(file_path), dst_path, archive_format)

    return file_list

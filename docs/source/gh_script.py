#!/usr/bin/env python3
"""Downloads and unpacks documentation for all published releases.
Usage: gh_script.py GH_OWNER/GH_REPO GH_TOKEN DST_PATH
"""
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pathlib import Path
import json
import shutil
import tarfile
import jinja2


ASSET_NAME = 'docs.tar'


class GHRequest:
    """GitHub API request helper."""

    def __init__(self, repository: str, token: str):
        self.repository = repository
        self.token = token

    def url(self, endpoint: str) -> str:
        return f'https://api.github.com/repos/{self.repository}/{endpoint}'

    def headers(self, content_type: str) -> dict:
        return {
            'Accept': content_type,
            'Authorization': f'Bearer {self.token}',
            'X-GitHub-Api-Version': '2022-11-28',
        }

    def request(self, endpoint: str, content_type: str) -> Request:
        return Request(url=self.url(endpoint), headers=self.headers(content_type))

    def request_json(self, endpoint: str):
        return self.request(endpoint, 'application/vnd.github+json')

    def request_data(self, endpoint: str):
        return self.request(endpoint, 'application/octet-stream')


def get_latest(req: GHRequest) -> str:
    """Returns latest release tag_name."""
    try:
        res = urlopen(req.request_json(endpoint='releases/latest'))
        release = json.loads(res.read())
        return release['tag_name']
    except HTTPError as exc:
        print(f'Request latest release. {exc}')


def get_versions(req: GHRequest) -> dict:
    """Returns releases dict(tag_name: asset_id)."""
    try:
        res = urlopen(req.request_json(endpoint='releases'))
        releases = json.loads(res.read())
    except HTTPError as exc:
        print(f'Request releases. {exc}')
        releases = []

    result = {}
    for release in releases:
        print(f'release {release["tag_name"]}:', end='')

        if release['draft'] or release['prerelease']:
            print('\tskip draft/pre-release')
            continue

        asset_id = None
        for asset in release['assets']:
            if asset['name'] == ASSET_NAME:
                print(f'\tfound asset id={asset["id"]} of size {asset["size"]}')
                asset_id = asset['id']
                break

        if asset_id is None:
            print('\tnot found asset')
            continue

        result[release['tag_name']] = asset_id

    return result


def download_asset(req: GHRequest, asset_id: int, download_path: Path) -> Path:
    """Downloads asset to path."""
    res = urlopen(req.request_data(f'releases/assets/{asset_id}'))
    asset_path = download_path / ASSET_NAME
    with open(asset_path, 'wb') as fp:
        fp.write(res.read())
    return asset_path


def untar(tar_path: Path):
    """Unpacks files from tar archive."""
    with tarfile.open(tar_path) as tar:
        tar.extractall(tar_path.parent)


def get_pages_url(req: GHRequest) -> str:
    """Gets information about a GitHub Pages site."""
    res = urlopen(req.request_json(endpoint='pages'))
    pages_info = json.loads(res.read())
    return pages_info['html_url']


def render_templates(variables: dict, dst: Path, src: Path = None):
    """Renders templates (*.html.tpl, *.js.tpl)."""
    src_path = Path(src) if src else Path(__file__).parent.resolve() / '_templates'
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(src_path))

    for tpl in src_path.glob('*.tpl'):
        print(f'render {tpl.name} => {str(dst / tpl.stem)}')
        with open(dst / tpl.stem, 'w') as fp:
            fp.write(env.get_template(tpl.name).render(variables))


def main(repository: str, token: str, dst_path: str):
    """Main.
    :param repository: GitHub repository: GH_OWNER/GH_REPO {{ github.repository }}
    :param token: GitHub API access token {{ github.token }}
    :param dst_path: Docs dir path
    """

    # prepare result dir
    result_path = Path(dst_path).absolute()
    result_path.mkdir(parents=True, exist_ok=True)

    # init request
    request = GHRequest(repository, token)

    # get latest version
    latest = get_latest(request) or 'develop'
    print('latest', latest)

    # get available versions
    versions = get_versions(request)

    # docs root url
    pages_url = get_pages_url(request).strip('/')

    # render html
    render_templates(
        dict(
            versions=list(versions) + ['develop'],
            latest=latest,
            pages_url=pages_url,
        ),
        result_path,
    )

    # download and unpack release docs
    for tag_name, assetid in versions.items():
        print(f'download release {tag_name} asset', end='')
        version_path = result_path / tag_name
        version_path.mkdir(parents=True, exist_ok=True)
        tar_file_path = download_asset(request, assetid, version_path)
        untar(tar_file_path)
        tar_file_path.unlink()
        print('\tdone')
        # add versions.js to the release
        shutil.copy(
            result_path / 'versions.js',
            version_path / '_static' / 'js' / 'versions.js',
        )


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('Usage: gh_script.py GH_OWNER/GH_REPO GH_TOKEN DST_PATH.')

    main(*sys.argv[1:])

#!/usr/bin/env python3
import json
import os
import sys
from typing import AnyStr, Optional
from urllib.request import Request, urlopen


def print_usage():
    """Show usage information."""
    print(f'Usage: {sys.argv[0]} <tag> <output_dir>')
    print(f'Example: {sys.argv[0]} 0.2.14 /tmp')
    print('Environment variables:')
    print('  GITHUB_TOKEN: GitHub personal access token')
    print('  GITHUB_REPOSITORY: GitHub repo name, default insight-platform/savant-rs')


def gh_request(
    repo: str, endpoint: str, content_type: str, token: Optional[str] = None
) -> AnyStr:
    """GitHub API request helper.

    :param repo: Repository name (owner/repo).
    :param endpoint: API endpoint.
    :param content_type: Content type.
    :param token: GitHub personal access token.
    """
    url = f'https://api.github.com/repos/{repo}/{endpoint}'
    headers = {
        'Accept': content_type,
        'X-GitHub-Api-Version': '2022-11-28',
    }
    if token is not None:
        headers['Authorization'] = f'Bearer {token}'
    req = Request(url, headers=headers)
    res = urlopen(req)
    return res.read()


def get_release_assets(tag: str, repo: str, token: Optional[str] = None) -> list:
    """Get the list of assets for a given release tag in a repository."""
    release = json.loads(
        gh_request(repo, f'releases/tags/{tag}', 'application/vnd.github+json', token)
    )
    return release.get('assets', [])


def download_asset(
    asset: dict, path: str, repo: str, token: Optional[str] = None
) -> str:
    """Download an asset and save it to the specified path."""
    data = gh_request(
        repo, f'releases/assets/{asset["id"]}', 'application/octet-stream', token
    )
    asset_path = os.path.join(os.path.abspath(path), asset['name'])
    os.makedirs(os.path.dirname(asset_path), exist_ok=True)
    with open(asset_path, 'wb') as fp:
        fp.write(data)
    return asset_path


def main():
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)

    gh_token = os.environ.get('GITHUB_TOKEN')
    gh_repo = os.environ.get('GITHUB_REPOSITORY', 'insight-platform/savant-rs')

    release_tag = sys.argv[1]
    download_path = sys.argv[2]

    python_short_version = f'cp{sys.version_info.major}{sys.version_info.minor}'

    arch = os.uname().machine

    assets = get_release_assets(release_tag, gh_repo, gh_token)
    if not assets:
        sys.exit(f'No assets found for tag {release_tag} in repository {gh_repo}.')

    asset_path = None
    for asset in assets:
        name = asset['name']
        if not name.startswith('savant_rs'):
            continue
        if not name.endswith('.whl'):
            continue
        if arch not in name:
            continue
        if python_short_version not in name:
            continue
        asset_path = download_asset(asset, download_path, gh_repo, gh_token)
        print(f'Downloaded {asset_path}.')
        break

    if asset_path is None:
        sys.exit(
            f'No savant_rs package found for tag {release_tag} in repository {gh_repo}.'
        )


if __name__ == '__main__':
    main()

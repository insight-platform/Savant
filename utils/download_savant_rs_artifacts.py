import requests
import os
import sys

python_short_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

# current architecture
arch = os.uname().machine

# help function
def print_usage():
    print(f"Usage: {sys.argv[0]} <tag> <output_dir>")
    print(f"Example: {sys.argv[0]} v0.1.0 /tmp")
    # environment variables
    print("Environment variables:")
    print("  GITHUB_TOKEN: GitHub personal access token")
    sys.exit(1)


if len(sys.argv) != 3:
    print_usage()

# Configuration
if 'GITHUB_TOKEN' in os.environ:
    TOKEN = os.environ['GITHUB_TOKEN']  # GitHub personal access token
else:
    TOKEN = None

REPO = 'insight-platform/savant-rs'  # Repository name
TAG = sys.argv[1]  # Release tag
OUTPUT_DIR = sys.argv[2] # Directory to save the downloaded artifacts

headers = {
    'Accept': 'application/vnd.github.v3+json',
}

if TOKEN:
    headers['Authorization'] = f'token {TOKEN}'

def download_file(url, path):
    """
    Download a file from a given URL and save it to the specified path.
    """
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded: {path}")

def get_release_assets(repo, tag):
    """
    Get the list of assets for a given release tag in a repository.
    """
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    assets = response.json().get('assets', [])
    return assets

def main():
    assets = get_release_assets(REPO, TAG)
    if not assets:
        print(f"No assets found for tag {TAG} in repository {REPO}.")
        return

    for asset in assets:
        name = asset['name']
        if arch not in name:
            continue
        if '.whl' in name and python_short_version not in name:
            continue
        download_url = asset['browser_download_url']
        output_path = os.path.join(OUTPUT_DIR, name)
        download_file(download_url, output_path)

if __name__ == "__main__":
    main()
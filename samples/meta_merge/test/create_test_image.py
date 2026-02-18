"""Create or download a test JPEG image with a person for YOLO detection."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print('Install Pillow: pip install Pillow', file=sys.stderr)
    sys.exit(1)

OUTPUT = Path(__file__).parent / 'test_image.jpeg'
# Sample image with person (Unsplash, free to use)
SAMPLE_URL = 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=640&q=80'


def download_sample() -> bool:
    """Download a sample image with a person."""
    try:
        import urllib.request

        urllib.request.urlretrieve(SAMPLE_URL, OUTPUT)
        return True
    except Exception as e:
        print(f'Download failed: {e}', file=sys.stderr)
        return False


def create_placeholder() -> None:
    """Create a minimal placeholder (no persons - for pipeline connectivity test)."""
    img = Image.new('RGB', (640, 480), color=(128, 128, 128))
    img.save(OUTPUT, 'JPEG', quality=85)
    print(f'Created placeholder {OUTPUT} (no persons - pipeline test only)')


def main() -> int:
    if OUTPUT.exists():
        print(f'{OUTPUT} already exists')
        return 0
    if not download_sample():
        create_placeholder()
    return 0


if __name__ == '__main__':
    sys.exit(main())

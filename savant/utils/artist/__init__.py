"""Artists package."""
from .artist import Artist
from .artist_opencv import ArtistGPUMat
from .position import Position

# BGR format
COLOR = {
    'red': (0, 0, 255),
    'green': (0, 128, 0),
    'blue': (255, 0, 0),
    'darkred': (0, 0, 139),
    'orangered': (0, 69, 255),
    'orange': (0, 165, 255),
    'yellow': (0, 255, 255),
    'lime': (0, 255, 0),
    'magenta': (255, 0, 255),
}

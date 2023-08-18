"""Animation module."""
from pathlib import Path

import cv2

from samples.peoplenet_detector.utils import load_sprite


class Animation:
    """Animation class. To render animation,
    call get_frame() for every source frame and draw returned image.

    :param sprites_dir: path to directory with sprites for the animation.
    :param fps: the speed of the animation, how many frames per seconds are played.
    :param sprite_height: the height of the animation in pixels.
        Sprites will be resized to this height, keeping aspect ratio.
    """

    def __init__(self, sprites_dir: str, fps: int, sprite_height: int) -> None:
        self.fps = fps
        self.frame_period_ns = 10**9 // self.fps
        self.sprites = [
            load_sprite(str(path), sprite_height)
            for path in sorted(Path(sprites_dir).glob('*.png'))
        ]
        self.current_sprite_idx = 0
        self.last_change_pts = 0

    @property
    def width(self) -> int:
        """Width of the animation's sprites after resize to target height."""
        return self.sprites[0].size()[0]

    def get_frame(self, pts: int) -> cv2.cuda.GpuMat:
        """Get animation's frame for the given source timestamp.

        :param pts: source frame presentation timestamp.
        :return: sprite data for the current frame.
        """
        sprite = self.sprites[self.current_sprite_idx]

        if pts - self.last_change_pts >= self.frame_period_ns:
            self.current_sprite_idx += 1
            if self.current_sprite_idx >= len(self.sprites):
                self.current_sprite_idx = 0
            self.last_change_pts = pts

        return sprite

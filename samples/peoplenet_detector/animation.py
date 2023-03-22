from pathlib import Path
import cv2
from samples.peoplenet_detector.utils import load_sprite


class Animation:
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
        return self.sprites[0].size()[0]

    def get_frame(self, pts: int) -> cv2.cuda.GpuMat:
        sprite = self.sprites[self.current_sprite_idx]

        if pts - self.last_change_pts >= self.frame_period_ns:
            self.current_sprite_idx += 1
            if self.current_sprite_idx >= len(self.sprites):
                self.current_sprite_idx = 0
            self.last_change_pts = pts

        return sprite

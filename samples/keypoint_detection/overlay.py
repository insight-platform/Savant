import numpy as np

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist

skeleton = [
    ([15, 13], (255, 0, 0, 255)),  # left leg
    ([13, 11], (255, 0, 0, 255)),  # left leg
    ([16, 14], (255, 0, 0, 255)),  # right leg
    ([14, 12], (255, 0, 0, 255)),  # right leg
    ([11, 12], (255, 0, 255, 255)),  # body
    ([5, 11], (255, 0, 255, 255)),  # body
    ([6, 12], (255, 0, 255, 255)),  # body
    ([5, 6], (255, 0, 255, 255)),  # body
    ([5, 7], (0, 255, 0, 255)),  # left arm
    ([7, 9], (0, 255, 0, 255)),  # left arm
    ([6, 8], (0, 255, 0, 255)),  # right arm
    ([8, 10], (0, 255, 0, 255)),  # right arm
    ([1, 2], (255, 255, 0, 255)),  # head
    ([0, 1], (255, 255, 0, 255)),  # head
    ([0, 2], (255, 255, 0, 255)),  # head
    ([1, 3], (255, 255, 0, 255)),  # head
    ([2, 4], (255, 255, 0, 255)),  # head
    ([3, 5], (255, 255, 0, 255)),  # head
    ([4, 6], (255, 255, 0, 255)),  # head
]


class Overlay(NvDsDrawFunc):
    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        for obj in frame_meta.objects:
            if obj.label == 'person':
                key_points = obj.get_attr_meta('yolov8npose', 'keypoint').value
                key_points = np.array(key_points).reshape(-1, 2)

                for pair, color in skeleton:
                    artist.add_line(
                        pt1=(int(key_points[pair[0]][0]), int(key_points[pair[0]][1])),
                        pt2=(int(key_points[pair[1]][0]), int(key_points[pair[1]][1])),
                        color=color,
                        thickness=2,
                    )
                for key_point in key_points:
                    if key_point[0] > 0 and key_point[1] > 0:
                        artist.add_circle(
                            center=(int(key_point[0]), int(key_point[1])),
                            radius=2,
                            color=(255, 0, 0, 255),
                            thickness=2,
                        )

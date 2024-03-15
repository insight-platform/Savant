from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.parameter_storage import param_storage
from savant.utils.artist import Artist

DETECTOR = param_storage()['detector']

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

# key points with the confidence below the threshold won't be displayed
KP_CONFIDENCE_THRESHOLD = 0.4

KP_LABELS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]


class Overlay(NvDsDrawFunc):
    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        # uncomment the following line to draw bounding boxes
        # super().draw_on_frame(frame_meta, artist)
        for obj in frame_meta.objects:
            if obj.label != 'person':
                continue

            kp_attr = obj.get_attr_meta(DETECTOR, 'keypoints')
            if kp_attr is None:
                continue

            key_points = kp_attr.value

            for pair, color in skeleton:
                if (
                    key_points[pair[0]][2] > KP_CONFIDENCE_THRESHOLD
                    and key_points[pair[1]][2] > KP_CONFIDENCE_THRESHOLD
                ):
                    artist.add_line(
                        pt1=(
                            int(key_points[pair[0]][0]),
                            int(key_points[pair[0]][1]),
                        ),
                        pt2=(
                            int(key_points[pair[1]][0]),
                            int(key_points[pair[1]][1]),
                        ),
                        color=color,
                        thickness=2,
                    )
            for i, (x, y, conf) in enumerate(key_points):
                if conf > KP_CONFIDENCE_THRESHOLD:
                    artist.add_circle(
                        center=(int(x), int(y)),
                        radius=2,
                        color=(255, 0, 0, 255),
                        thickness=2,
                    )
                    # show label
                    # artist.add_text(
                    #     text=KP_LABELS[i],
                    #     anchor=(int(key_point[0]), int(key_point[1])),
                    # )

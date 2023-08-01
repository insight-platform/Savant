from typing import Tuple
import numpy as np


def pack_person_id_img_n(person_id, img_n) -> int:
    person_id = np.uint64(person_id)
    img_n = np.uint64(img_n)
    packed = np.bitwise_or(np.left_shift(person_id, 32), img_n)
    return packed

def unpack_person_id_img_n(packed: int) -> Tuple[int, int]:
    person_id = np.right_shift(packed, 32)
    img_n = np.bitwise_and(packed, 0xFFFFFFFF)
    return person_id, img_n

from typing import Tuple
import numpy as np


BIT_SHIFT = np.uint64(32)
BIT_MASK = np.uint64(0xFFFFFFFF)


def pack_person_id_img_n(person_id, img_n) -> int:
    person_id = np.uint64(person_id)
    img_n = np.uint64(img_n)
    packed = np.bitwise_or(np.left_shift(person_id, BIT_SHIFT), img_n)
    return packed


def unpack_person_id_img_n(packed: int) -> Tuple[int, int]:
    person_id = np.right_shift(packed, BIT_SHIFT)
    img_n = np.bitwise_and(packed, BIT_MASK)
    return person_id, img_n

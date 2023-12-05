"""Non-maximum suppression (NMS) implementation."""
import cupy as cp
import numba as nb
import numpy as np

__all__ = ['nms_cpu', 'nms_gpu']


@nb.njit('u4[:](f4[:, :], f4[:], f4, u2)', nogil=True, cache=True)
def nms_cpu(
    bboxes: np.ndarray, confidences: np.ndarray, threshold: float, top_k: int = 300
) -> np.ndarray:
    """Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU). NumPy (CPU) version.

    :param bboxes: Boxes to perform NMS on.
        They are expected to be in (xc, yc, width, height) format.
    :param confidences: Scores for each one of the boxes.
    :param threshold: IoU threshold.
        Discards all overlapping boxes with IoU > threshold.
    :param top_k: Returns only K with max confidence/score.
    :return: Indices of the boxes that have been kept by NMS,
        sorted in decreasing order of scores.
    """
    x_left = bboxes[:, 0]
    y_top = bboxes[:, 1]
    x_right = bboxes[:, 0] + bboxes[:, 2]
    y_bottom = bboxes[:, 1] + bboxes[:, 3]

    areas = (x_right - x_left) * (y_bottom - y_top)
    order = confidences.argsort()[::-1].astype(np.uint32)

    mask = np.zeros((min(bboxes.shape[0], top_k),), dtype=np.uint32)
    mask_idx = 0
    while order.size > 0 and mask_idx < top_k:
        idx_self = order[0]
        idx_other = order[1:]

        mask[mask_idx] = idx_self
        mask_idx += 1

        xx1 = np.maximum(x_left[idx_self], x_left[idx_other])
        yy1 = np.maximum(y_top[idx_self], y_top[idx_other])
        xx2 = np.minimum(x_right[idx_self], x_right[idx_other])
        yy2 = np.minimum(y_bottom[idx_self], y_bottom[idx_other])

        width = np.maximum(0.0, xx2 - xx1)
        height = np.maximum(0.0, yy2 - yy1)
        inter = width * height

        over = inter / (areas[idx_self] + areas[idx_other] - inter)

        indices = np.where(over <= threshold)[0]
        order = order[indices + 1]

    return mask[:mask_idx]


def nms_gpu(
    bboxes: cp.ndarray, confidences: cp.ndarray, threshold: float, top_k: int = 300
) -> cp.ndarray:
    """Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU). CuPy (GPU) version.

    Kernel is borrowed from https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/non_maximum_suppression.py

    :param bboxes: Boxes to perform NMS on.
        They are expected to be in (xc, yc, width, height) format.
    :param confidences: Scores for each one of the boxes.
    :param threshold: IoU threshold.
        Discards all overlapping boxes with IoU > threshold.
    :param top_k: Returns only K with max confidence/score.
    :return: Indices of the boxes that have been kept by NMS,
        sorted in decreasing order of scores.
    """
    _bboxes = cp.zeros_like(bboxes)
    _bboxes[:, 0] += bboxes[:, 0]
    _bboxes[:, 1] += bboxes[:, 1]
    _bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    _bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    order = confidences.argsort()[::-1].astype(np.uint32)
    sorted_bboxes = _bboxes[order, :]
    mask = _call_nms_kernel(sorted_bboxes, threshold)
    mask = order[mask]
    return mask[:top_k]


_nms_gpu_code = '''
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__
inline float devIoU(float const *const bbox_a, float const *const bbox_b) {
  float top = max(bbox_a[0], bbox_b[0]);
  float bottom = min(bbox_a[2], bbox_b[2]);
  float left = max(bbox_a[1], bbox_b[1]);
  float right = min(bbox_a[3], bbox_b[3]);
  float height = max(bottom - top, 0.f);
  float width = max(right - left, 0.f);
  float area_i = height * width;
  float area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]);
  float area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]);
  return area_i / (area_a + area_b - area_i);
}

extern "C"
__global__
void nms_kernel(const int n_bbox, const float thresh,
                const float *dev_bbox,
                unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
        min(n_bbox - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_bbox - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_bbox[threadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_bbox[threadIdx.x * 4 + 0] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_bbox[threadIdx.x * 4 + 1] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_bbox[threadIdx.x * 4 + 2] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_bbox[threadIdx.x * 4 + 3] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_bbox + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_bbox + i * 4) >= thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_bbox, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}
'''


def _call_nms_kernel(bboxes: cp.ndarray, threshold: float) -> cp.ndarray:
    n_bbox = bboxes.shape[0]
    threads_per_block = 64

    col_blocks = np.ceil(n_bbox / threads_per_block).astype(np.int32)
    blocks = (col_blocks, col_blocks, 1)
    threads = (threads_per_block, 1, 1)

    mask = cp.zeros((n_bbox * col_blocks,), dtype=cp.uint64)

    c_bboxes = cp.ascontiguousarray(bboxes, dtype=cp.float32)

    kernel = cp.RawKernel(_nms_gpu_code, 'nms_kernel')
    kernel(
        blocks,
        threads,
        args=(cp.int32(n_bbox), cp.float32(threshold), c_bboxes, mask),
    )

    return _nms_gpu_post(mask.get(), n_bbox, threads_per_block, col_blocks)


@nb.njit('u4[:](u8[:], u2, u2, u2)', nogil=True, cache=True)
def _nms_gpu_post(
    mask: np.ndarray, n_bbox: int, threads_per_block: int, col_blocks: int
) -> np.ndarray:
    n_selection = 0
    one_ull = np.uint64(1)
    selection = np.zeros((n_bbox,), dtype=np.uint32)
    remv = np.zeros((col_blocks,), dtype=np.uint64)

    for i in range(n_bbox):
        n_block = i // threads_per_block
        in_block = np.uint64(i % threads_per_block)

        if not (remv[n_block] & one_ull << in_block):
            selection[n_selection] = i
            n_selection += 1

            index = i * col_blocks
            for j in range(n_block, col_blocks):
                remv[j] |= mask[index + j]

    return selection[:n_selection]

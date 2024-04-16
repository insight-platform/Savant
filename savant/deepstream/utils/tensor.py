"""DeepStream tensor utils."""

import ctypes
from typing import List, Optional

import cupy as cp
import numpy as np
import pyds

__all__ = [
    'nvds_infer_tensor_meta_to_outputs',
    'nvds_infer_tensor_meta_to_outputs_cupy',
]

DATA_TYPE_MAP = {
    pyds.NvDsInferDataType.FLOAT: ctypes.c_float,
    # pyds.NvDsInferDataType.HALF: # TODO: ctypes doesn't support half precision
    pyds.NvDsInferDataType.INT8: ctypes.c_int8,
    pyds.NvDsInferDataType.INT32: ctypes.c_int32,
}


def nvds_infer_tensor_meta_to_outputs(
    tensor_meta: pyds.NvDsInferTensorMeta, layer_names: List[str]
) -> List[Optional[np.ndarray]]:
    """Fetches output of specified layers from pyds.NvDsInferTensorMeta.
    Returns tensors on host as ``numpy.ndarray``.

    :param tensor_meta: NvDsInferTensorMeta structure.
    :param layer_names: Names of layers to return, order is important.
    :return: List of specified layer data.
    """
    layers = [None] * len(layer_names)
    for i in range(tensor_meta.num_output_layers):
        layer_info = pyds.get_nvds_LayerInfo(tensor_meta, i)
        if layer_info.isInput or layer_info.layerName not in layer_names:
            continue
        if layer_info.dataType not in DATA_TYPE_MAP:
            raise ValueError(
                f'Unsupported layer "{layer_info.layerName}" '
                f'data type {layer_info.dataType}.'
            )
        layers[layer_names.index(layer_info.layerName)] = np.ctypeslib.as_array(
            ctypes.cast(
                pyds.get_ptr(layer_info.buffer),
                ctypes.POINTER(DATA_TYPE_MAP[layer_info.dataType]),
            ),
            shape=layer_info.dims.d[: layer_info.dims.numDims],
        )
    return layers


def nvds_infer_tensor_meta_to_outputs_cupy(
    tensor_meta: pyds.NvDsInferTensorMeta,
    layer_names: List[str],
) -> List[Optional[cp.ndarray]]:
    """Fetches output of specified layers from pyds.NvDsInferTensorMeta.
    Returns tensors on device as ``cupy.ndarray`` (zero-copy).

    :param tensor_meta: NvDsInferTensorMeta structure.
    :param layer_names: Names of layers to return, order is important.
    :return: List of specified layer data.
    """
    layers = [None] * len(layer_names)
    out_buf_ptrs_dev = ctypes.cast(
        pyds.get_ptr(tensor_meta.out_buf_ptrs_dev),
        ctypes.POINTER(ctypes.c_void_p),
    )
    out_buf_ptrs_dev_addr = ctypes.addressof(out_buf_ptrs_dev.contents)
    for i in range(tensor_meta.num_output_layers):
        layer_info = pyds.get_nvds_LayerInfo(tensor_meta, i)
        if layer_info.isInput or layer_info.layerName not in layer_names:
            continue
        if layer_info.dataType not in DATA_TYPE_MAP:
            raise ValueError(
                f'Unsupported layer "{layer_info.layerName}" '
                f'data type {layer_info.dataType}.'
            )
        casted_dev = ctypes.cast(
            ctypes.cast(
                out_buf_ptrs_dev_addr, ctypes.POINTER(ctypes.c_void_p)
            ).contents,
            ctypes.POINTER(DATA_TYPE_MAP[layer_info.dataType]),
        )
        casted_dev_ptr = ctypes.addressof(casted_dev.contents)
        out_buf_ptrs_dev_addr += ctypes.sizeof(ctypes.c_void_p)
        unowned_mem = cp.cuda.UnownedMemory(
            casted_dev_ptr,
            layer_info.dims.numElements
            * ctypes.sizeof(DATA_TYPE_MAP[layer_info.dataType]),
            None,
        )
        mem_ptr = cp.cuda.MemoryPointer(unowned_mem, 0)
        layers[layer_names.index(layer_info.layerName)] = cp.ndarray(
            shape=layer_info.dims.d[: layer_info.dims.numDims],
            dtype=DATA_TYPE_MAP[layer_info.dataType],
            memptr=mem_ptr,
            order='C',
        )
    return layers

"""DeepStream tensor utils."""
import ctypes
from typing import List, Optional, Union

import cupy as cp
import numpy as np
import pyds


def nvds_infer_tensor_meta_to_outputs(
    tensor_meta: pyds.NvDsInferTensorMeta,
    layer_names: List[str],
    gpu: bool = False,
) -> List[Optional[Union[np.ndarray, cp.ndarray]]]:
    """Fetches output of specified layers from pyds.NvDsInferTensorMeta.

    :param tensor_meta: NvDsInferTensorMeta structure.
    :param layer_names: Names of layers to return, order is important.
    :param gpu: Set to True to get the output device buffers (cupy.ndarray)
        instead of host buffers (numpy.ndarray).
    :return: List of specified layer data.
    """
    data_type_map = {
        pyds.NvDsInferDataType.FLOAT: ctypes.c_float,
        # pyds.NvDsInferDataType.HALF: # TODO: ctypes doesn't support half precision
        pyds.NvDsInferDataType.INT8: ctypes.c_int8,
        pyds.NvDsInferDataType.INT32: ctypes.c_int32,
    }
    layers = [None] * len(layer_names)

    out_buf_ptrs_dev_addr = None
    if gpu:
        out_buf_ptrs_dev = ctypes.cast(
            pyds.get_ptr(tensor_meta.out_buf_ptrs_dev),
            ctypes.POINTER(ctypes.c_void_p),
        )
        out_buf_ptrs_dev_addr = ctypes.addressof(out_buf_ptrs_dev.contents)

    for i in range(tensor_meta.num_output_layers):
        layer_info = pyds.get_nvds_LayerInfo(tensor_meta, i)

        if layer_info.isInput or layer_info.layerName not in layer_names:
            continue

        if layer_info.dataType not in data_type_map:
            raise ValueError(
                f'Unsupported layer "{layer_info.layerName}" '
                f'data type {layer_info.dataType}.'
            )

        if gpu:
            casted_dev = ctypes.cast(
                ctypes.cast(
                    out_buf_ptrs_dev_addr, ctypes.POINTER(ctypes.c_void_p)
                ).contents,
                ctypes.POINTER(data_type_map[layer_info.dataType]),
            )
            casted_dev_ptr = ctypes.addressof(casted_dev.contents)
            out_buf_ptrs_dev_addr += ctypes.sizeof(ctypes.c_void_p)

            unowned_mem = cp.cuda.UnownedMemory(
                casted_dev_ptr,
                layer_info.dims.numElements
                * ctypes.sizeof(data_type_map[layer_info.dataType]),
                None,
            )
            mem_ptr = cp.cuda.MemoryPointer(unowned_mem, 0)
            layers[layer_names.index(layer_info.layerName)] = cp.ndarray(
                shape=layer_info.dims.d[: layer_info.dims.numDims],
                dtype=data_type_map[layer_info.dataType],
                memptr=mem_ptr,
                order='C',
            )

        else:
            layers[layer_names.index(layer_info.layerName)] = np.ascontiguousarray(
                np.ctypeslib.as_array(
                    ctypes.cast(
                        pyds.get_ptr(layer_info.buffer),
                        ctypes.POINTER(data_type_map[layer_info.dataType]),
                    ),
                    shape=layer_info.dims.d[: layer_info.dims.numDims],
                )
            )

    return layers

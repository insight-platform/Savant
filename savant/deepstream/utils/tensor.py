"""DeepStream tensor utils."""
import ctypes
from typing import List, Optional

import numpy as np
import pyds


def nvds_infer_tensor_meta_to_outputs(
    tensor_meta: pyds.NvDsInferTensorMeta, layer_names: List[str]
) -> List[Optional[np.ndarray]]:
    """Fetches output of specified layers from pyds.NvDsInferTensorMeta.

    :param tensor_meta: NvDsInferTensorMeta structure.
    :param layer_names: Names of layers to return, order is important.
    :return: List of specified layer data.
    """
    data_type_map = {
        pyds.NvDsInferDataType.FLOAT: ctypes.c_float,
        # pyds.NvDsInferDataType.HALF: # TODO: ctypes doesn't support half precision
        pyds.NvDsInferDataType.INT8: ctypes.c_int8,
        pyds.NvDsInferDataType.INT32: ctypes.c_int32,
    }
    layers = [None] * len(layer_names)
    for i in range(tensor_meta.num_output_layers):
        layer_info = pyds.get_nvds_LayerInfo(tensor_meta, i)
        if not layer_info.isInput and layer_info.layerName in layer_names:
            if layer_info.dataType not in data_type_map:
                raise ValueError(
                    f'Unsupported layer "{layer_info.layerName}" '
                    f'data type {layer_info.dataType}.'
                )
            layers[layer_names.index(layer_info.layerName)] = np.copy(
                np.ctypeslib.as_array(
                    ctypes.cast(
                        pyds.get_ptr(layer_info.buffer),
                        ctypes.POINTER(data_type_map[layer_info.dataType]),
                    ),
                    shape=layer_info.dims.d[: layer_info.dims.numDims],
                )
            )
    return layers

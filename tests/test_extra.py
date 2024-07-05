"""Test extra packages."""


def test_extra():
    """Try to import extra packages and check the version."""
    import torch

    print('PyTorch', torch.__version__)

    import torchaudio

    print('Torchaudio', torchaudio.__version__)

    import torchvision

    print('Torchvision', torchvision.__version__)

    import tensorrt

    print('TensorRT', tensorrt.__version__)

    import torch2trt

    print('Torch2TRT OK')

    import onnx

    print('ONNX', onnx.__version__)

    import onnxruntime

    print('ONNX Runtime', onnxruntime.__version__)

    import pycuda

    print('PyCUDA', pycuda.VERSION_TEXT)

    import pandas

    print('Pandas', pandas.__version__)

    import polars

    print('Polars', polars.__version__)

    import sklearn

    print('Scikit-Learn', sklearn.__version__)

    import jupyterlab

    print('JupyterLab', jupyterlab.__version__)


if __name__ == '__main__':
    test_extra()

"""Test onnxruntime."""

import numpy as np
import onnxruntime as ort
import torch
from torchvision.models import ResNet18_Weights, resnet


def test_onnxruntime():
    model = resnet.resnet18(weights=ResNet18_Weights.DEFAULT).eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(model, dummy_input, 'resnet18.onnx')

    ort_session = ort.InferenceSession(
        'resnet18.onnx', providers=['CUDAExecutionProvider']
    )

    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    torch_out = model(dummy_input).detach().numpy()

    assert np.allclose(ort_outs[0], torch_out, atol=1e-2)


if __name__ == '__main__':
    test_onnxruntime()

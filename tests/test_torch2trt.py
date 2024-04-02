"""Test torch2trt."""
import numpy as np
import torch
from torch2trt import torch2trt
from torchvision.models import ResNet18_Weights, resnet


def test_torch2trt():
    """Test torch2trt."""
    model = resnet.resnet18(weights=ResNet18_Weights.DEFAULT).eval().cuda()

    dummy_input = torch.randn(1, 3, 224, 224).cuda()

    model_trt = torch2trt(model, [dummy_input])

    torch_out = model(dummy_input).detach().cpu().numpy()

    trt_out = model_trt(dummy_input).detach().cpu().numpy()

    assert np.allclose(trt_out, torch_out, atol=1e-02)

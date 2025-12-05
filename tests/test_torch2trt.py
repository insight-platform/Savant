"""Test torch2trt."""

import numpy as np
import torch
from torch2trt import torch2trt
from torchvision.models import resnet


def test_torch2trt():
    """Test torch2trt."""
    model = resnet.resnet18().eval().cuda()

    dummy_input = torch.full((1, 3, 224, 224), 0.5, dtype=torch.float32).cuda()

    model_trt = torch2trt(model, [dummy_input])

    torch_out = model(dummy_input).detach().cpu().numpy()

    trt_out = model_trt(dummy_input).detach().cpu().numpy()

    assert np.allclose(trt_out, torch_out, atol=1e-02)

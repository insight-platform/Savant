import onnx
import torch
from torch import nn

model = nn.Identity()

dummy_input = torch.zeros(1, 3, 100, 100)
onnx_model_path = 'identity.onnx'

torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}},
)

onnx_model = onnx.load(onnx_model_path)

print(onnx.helper.printable_graph(onnx_model.graph))

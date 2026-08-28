"""RT-DETR PyTorch to ONNX export for Savant.

Derived from DeepStream-Yolo/utils/export_rtdetr_pytorch.py,
with the output head replaced by one that matches the layout expected
by Savant's YOLO output converter (savant.converter.yolo.TensorToBBoxConverter),
so that the model needs no custom nvinfer parsing library.

The script is expected to be copied into the RT-DETR/rtdetr_pytorch directory
of the RT-DETR repo, see the sample README for the exact versions used.
"""

import os

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import YAMLConfig


class SavantOutput(nn.Module):
    """Converts RT-DETR decoder output to a single (num_classes + 4) x num_queries
    tensor: cxcywh boxes in network input pixels followed by per-class scores.

    That is the layout the Savant YOLO converter recognizes by
    ``shape[0] == num_detected_classes + 4``, the same one YOLOv8/v11 export to.
    """

    def __init__(self, img_size, use_focal_loss):
        super().__init__()
        self.img_size = img_size
        self.use_focal_loss = use_focal_loss

    def forward(self, x):
        # decoder boxes are cxcywh normalized to [0, 1]
        boxes = x["pred_boxes"]
        scale = (
            torch.as_tensor([[*self.img_size]], dtype=boxes.dtype, device=boxes.device)
            .flip(1)
            .tile([1, 2])
            .unsqueeze(1)
        )
        boxes = boxes * scale
        # the converter picks the class itself, all class scores are kept
        scores = (
            F.sigmoid(x["pred_logits"])
            if self.use_focal_loss
            else F.softmax(x["pred_logits"], dim=-1)[:, :, :-1]
        )
        return torch.cat([boxes, scores], dim=-1).transpose(1, 2)


def rtdetr_pytorch_export(weights, cfg_file, device):
    cfg = YAMLConfig(cfg_file, resume=weights)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
    cfg.model.load_state_dict(state)
    return cfg.model.deploy().to(device), cfg.postprocessor.use_focal_loss


def suppress_warnings():
    import warnings

    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)


def main(args):
    suppress_warnings()

    print(f"\nStarting: {args.weights}")

    print("Opening RT-DETR PyTorch model")

    device = torch.device("cpu")
    model, use_focal_loss = rtdetr_pytorch_export(args.weights, args.config, device)

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    model = nn.Sequential(model, SavantOutput(img_size, use_focal_loss))

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = args.weights.rsplit(".", 1)[0] + ".onnx"

    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    print("Exporting the model to ONNX")
    torch.onnx.export(
        model,
        onnx_input_im,
        onnx_output_file,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes if args.dynamic else None,
    )

    if args.simplify:
        print("Simplifying the ONNX model")
        import onnxslim

        model_onnx = onnx.load(onnx_output_file)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f"Done: {onnx_output_file}\n")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Savant RT-DETR PyTorch conversion")
    parser.add_argument(
        "-w",
        "--weights",
        required=True,
        type=str,
        help="Input weights (.pth) file path (required)",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Input YAML (.yml) file path (required)",
    )
    parser.add_argument(
        "-s",
        "--size",
        nargs="+",
        type=int,
        default=[640],
        help="Inference size [H,W] (default [640])",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplify model")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch-size")
    parser.add_argument("--batch", type=int, default=1, help="Static batch-size")
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise RuntimeError("Invalid weights file")
    if not os.path.isfile(args.config):
        raise RuntimeError("Invalid config file")
    if args.dynamic and args.batch > 1:
        raise RuntimeError(
            "Cannot set dynamic batch-size and static batch-size at same time"
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

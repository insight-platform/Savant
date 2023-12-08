"""Custom PyFunc implementation inference PyTorch model."""
from savant_rs.primitives.geometry import BBox

from savant.meta.object import ObjectMeta
from savant.utils.memory_repr_pytorch import pytorch_tensor_as_opencv_gpu_mat, opencv_gpu_mat_as_pytorch_tensor


import cv2
import torch
import torchvision
import torchvision.transforms as transforms

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import alpha_comp, nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.utils.artist import Artist


class PyTorchInfer(NvDsPyFuncPlugin):
    """Custom frame processor."""

    def __init__(self, conf_threshold, iou_threshold, road_mask_color, line_mask_color, **kwargs):
        super().__init__(**kwargs)
        import sys

        print(sys.path)
        self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
        self.model.cuda()
        self.model.eval()
        self.road_mask_color = torch.tensor(
            [road_mask_color], dtype=torch.uint8, device='cuda'
        )
        self.line_mask_color = torch.tensor(
            [line_mask_color], dtype=torch.uint8, device='cuda'
        )
        self.bg_color = torch.tensor([[0, 0, 0, 0]], dtype=torch.uint8, device='cuda')
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        pass

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame.

        :param buffer: GStreamer buffer.
        :param frame_meta: Processed frame metadata.
        """
        stream = self.get_cuda_stream(frame_meta)
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            with Artist(frame_mat, stream) as artist:
                with torch.no_grad():
                    w, h = frame_mat.size()
                    input_image = cv2.cuda.GpuMat()
                    input_image = cv2.cuda.resize(
                        frame_mat, (640, 480), stream=artist.stream
                    )

                    input_tensor = opencv_gpu_mat_as_pytorch_tensor(input_image).permute(2, 0, 1)
                    input_tensor = input_tensor[:3, :, :].float() / 255
                    input_tensor = self.normalize(input_tensor).unsqueeze(0)
                    det_out, da_seg_out, ll_seg_out = self.model(input_tensor)
                    da_seg_out = da_seg_out.detach()
                    ll_seg_out = ll_seg_out.detach()

                    self.postprocess_bbox(det_out, frame_meta, input_tensor, h, w)

                    da_seg_mask = torch.nn.functional.interpolate(
                        da_seg_out, size=(h, w), mode='bilinear'
                    )
                    ll_seg_mask = torch.nn.functional.interpolate(
                        ll_seg_out, size=(h, w), mode='bilinear'
                    )
                    da_seg_mask = torch.max(da_seg_mask, 1)[1].squeeze(0)
                    ll_seg_mask = torch.max(ll_seg_mask, 1)[1].squeeze(0)
                    mask_seg = torch.where(
                        da_seg_mask.bool()[..., None],
                        self.road_mask_color,
                        self.bg_color,
                    )
                    ll_mask = torch.where(
                        ll_seg_mask.bool()[..., None], self.line_mask_color, self.bg_color
                    )

                    alpha_comp(
                        frame_mat,
                        overlay=pytorch_tensor_as_opencv_gpu_mat(mask_seg),
                        start=(0, 0),
                        stream=stream,
                    )
                    alpha_comp(
                        frame_mat,
                        overlay=pytorch_tensor_as_opencv_gpu_mat(ll_mask),
                        start=(0, 0),
                        stream=stream,
                    )

    def postprocess_bbox(self, det_out, frame_meta, input_tensor_my, h, w):
        inf_out = det_out[0].squeeze(0)
        x = inf_out[inf_out[:, 4] > self.conf_threshold, :]
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = xywh2xyxy(x[:, :4])
        i = torchvision.ops.nms(box, x[:, 4], self.iou_threshold)
        output = x[i]
        output[:, :4] = scale_coords(
            input_tensor_my.shape[2:], output[:, :4], (h, w)
        ).round()
        for obj_meta_tensor in output:
            bbox = BBox(
                float(obj_meta_tensor[0]),
                float(obj_meta_tensor[1]),
                float(obj_meta_tensor[2]),
                float(obj_meta_tensor[3]),
            )
            obj_meta = ObjectMeta(
                element_name='yolop',
                label='car',
                bbox=bbox,
                confidence=float(obj_meta_tensor[4]),
            )
            frame_meta.add_obj_meta(obj_meta)

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        self.logger.debug('Got GST_NVEVENT_STREAM_EOS for source %s.', source_id)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x_center, y_center, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape):

    coords[:, :4] /= torch.tensor(
        [
            img1_shape[1] / img0_shape[1],
            img1_shape[0] / img0_shape[0],
            img1_shape[1] / img0_shape[1],
            img1_shape[0] / img0_shape[0],
        ],
        device='cuda',
    )

    clip_coords(coords, img0_shape)
    return coords




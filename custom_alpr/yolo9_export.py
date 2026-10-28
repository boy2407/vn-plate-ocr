"""Single-class YOLOv9 ONNX wrapper with the FastALPR Nx7 output contract."""

import torch
from torch import nn
from torchvision.ops import nms


class OrtNms(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, boxes, scores, max_output, iou_threshold, score_threshold):
        selected = torch.where(scores[0, 0] >= float(score_threshold.item()))[0]
        keep = nms(boxes[0, selected], scores[0, 0, selected], float(iou_threshold.item()))
        indices = selected[keep[: int(max_output.item())]]
        zeros = torch.zeros_like(indices)
        return torch.stack((zeros, zeros, indices), dim=1)

    @staticmethod
    def symbolic(graph, boxes, scores, max_output, iou_threshold, score_threshold):
        return graph.op(
            "NonMaxSuppression", boxes, scores, max_output, iou_threshold, score_threshold
        )


class FastPlateExport(nn.Module):
    """Keep model parameters intact; only decode boxes and apply single-class NMS."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("max_output", torch.tensor([100], dtype=torch.int64))
        self.register_buffer("iou_threshold", torch.tensor([0.45]))
        self.register_buffer("score_threshold", torch.tensor([0.001]))

    def forward(self, images):
        raw = self.model(images)
        coordinates = raw[:, :4].transpose(1, 2)
        center, half_size = coordinates[..., :2], coordinates[..., 2:] / 2
        boxes = torch.cat((center - half_size, center + half_size), dim=2)
        scores = raw[:, 4:5]
        selected = OrtNms.apply(
            boxes, scores, self.max_output, self.iou_threshold, self.score_threshold
        )
        batch, anchors = selected[:, 0], selected[:, 2]
        return torch.cat(
            (
                selected[:, 0:1].float(),
                boxes[batch, anchors],
                selected[:, 1:2].float(),
                scores[batch, 0, anchors].unsqueeze(1),
            ),
            dim=1,
        )

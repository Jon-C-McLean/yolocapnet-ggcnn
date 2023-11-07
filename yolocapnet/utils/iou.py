import torch

EPSILON = 1e-6

def iou_width_height(boxes_a, boxes_b):
    intersection = torch.min(boxes_a[..., 0], boxes_b[..., 0]) * torch.min(boxes_a[..., 1], boxes_b[..., 1])
    union = (boxes_a[..., 0] * boxes_a[..., 1]) + (boxes_b[..., 0] * boxes_b[..., 1]) - intersection

    return intersection / union

def iou(preds, labels, format='midpoint'):
    assert format in ['corners', 'midpoint'], 'Format must be either "corners" or "midpoint"'

    if format == "midpoint":
        box1 = [
            preds[..., 0:1] - preds[..., 2:3] / 2,
            preds[..., 1:2] - preds[..., 3:4] / 2,
            preds[..., 0:1] + preds[..., 2:3] / 2,
            preds[..., 1:2] + preds[..., 3:4] / 2
        ]

        box2 = [
            labels[..., 0:1] - labels[..., 2:3] / 2,
            labels[..., 1:2] - labels[..., 3:4] / 2,
            labels[..., 0:1] + labels[..., 2:3] / 2,
            labels[..., 1:2] + labels[..., 3:4] / 2
        ]
    elif format == "corners":
        box1 = [
            preds[..., 0:1],
            preds[..., 1:2],
            preds[..., 2:3],
            preds[..., 3:4]
        ]

        box2 = [
            labels[..., 0:1],
            labels[..., 1:2],
            labels[..., 2:3],
            labels[..., 3:4]
        ]

    coords = [torch.max(box1[0], box2[0]), torch.max(box1[1], box2[1]), torch.min(box1[2], box2[2]), torch.min(box1[3], box2[3])] # x1 y1 x2 y2

    intersection = (coords[3] - coords[0]).clamp(0) * (coords[2] - coords[1]).clamp(0)
    box1_area = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    box2_area = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))

    return intersection / (box1_area + box2_area - intersection + EPSILON)
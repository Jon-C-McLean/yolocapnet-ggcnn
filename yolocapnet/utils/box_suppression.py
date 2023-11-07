import torch
import numpy as np

from .iou import iou

def nms(bounding, iou_threshold, threshold, format='corners'):
    '''
    Non-maximum suppression algorithm for removing overlapping bounding boxes.

    Parameters:
        bounding (list): list of lists containing bounding boxes [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold for removing overlapping bounding boxes
        threshold (float): threshold for removing bounding boxes with low confidence
        format (str): format of bounding boxes, either "corners" or "midpoint"
    Returns:
        resulting_boxes (list): list of lists containing bounding boxes after NMS given a specific IOU threshold [class_pred, prob_score, x1, y1, x2, y2]
    '''
    assert format in ['corners', 'midpoint'], 'Format must be either "corners" or "midpoint"'

    bboxes = [box for box in bounding if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    resulting_boxes = []

    while bboxes:
        sel_box = bboxes.pop(0)

        bboxes = [
            box for box in bboxes
            if box[0] != sel_box[0] or iou(torch.tensor(sel_box[2:]), torch.tensor(box[2:]), format=format) < iou_threshold
        ]

        resulting_boxes.append(sel_box)
    
    return resulting_boxes
# modules/yolo_postprocess.py

import torch
import torchvision

def box_iou(box1, box2):
    """
    Returns the IoU of two sets of boxes
    box1: (N, 4), box2: (M, 4)
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]).clamp(0) * (box[:, 3] - box[:, 1]).clamp(0)

    area1 = box_area(box1)
    area2 = box_area(box2)

    inter = (
        torch.min(box1[:, None, 2:], box2[:, 2:])
        - torch.max(box1[:, None, :2], box2[:, :2])
    ).clamp(0).prod(2)

    return inter / (area1[:, None] + area2 - inter + 1e-6)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Performs Non-Maximum Suppression (NMS) on inference results.
    Expects input format: (N, 85) where 85 = [x1, y1, x2, y2, object_conf, class_conf...]
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence threshold

        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections per class
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if not x.shape[0]:
            continue

        # NMS
        boxes, scores = x[:, :4], x[:, 4]
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[keep][:max_det]

    return output

def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

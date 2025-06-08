import numpy as np

# bbox 중심 좌표 계산
def get_center(bbox):
    n = len(bbox)
    if n in (4, 5):                        # 4 또는 5 길이 모두 허용
        x1, y1, x2, y2 = bbox[:4]          # conf(5번째)는 무시
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    elif n == 2:                           # 이미 (cx,cy)
        return bbox
    else:
        raise ValueError(f"[get_center] len must be 2, 4 or 5, got {n}")

# 두 bbox간 중심 좌표 거리 계산
def center_distance(bbox1, bbox2):
    c1 = get_center(bbox1)
    c2 = get_center(bbox2)
    return np.linalg.norm(np.subtract(c1, c2))


# # 두 bbox간 IOU 계산 함수
# def iou(bbox1, bbox2):
#     x1, y1, x2, y2 = bbox1
#     x3, y3, x4, y4 = bbox2
#     xi1 = max(x1, x3)
#     yi1 = max(y1, y3)
#     xi2 = min(x2, x4)
#     yi2 = min(y2, y4)
#     inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
#     union_area = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter_area
#     return inter_area / union_area if union_area > 0 else 0

import numpy as np

# bbox 중심 좌표 계산
def get_center(bbox):
    n = len(bbox)
    if n in (4, 5):                        # 좌표 이외의 값이 있어도 허용
        x1, y1, x2, y2 = bbox[:4]          # conf(5번째)는 무시
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    elif n == 2:                           # 이미 (cx,cy) 형태
        return bbox
    else:
        raise ValueError(f"[get_center] len must be 2, 4 or 5, got {n}")

# 두 bbox간 중심 좌표 거리 계산
def center_distance(bbox1, bbox2):
    c1 = get_center(bbox1)
    c2 = get_center(bbox2)
    return np.linalg.norm(np.subtract(c1, c2))

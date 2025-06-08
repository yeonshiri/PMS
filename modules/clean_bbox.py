# modules/utils.py (또는 별도 파일)
import numpy as np
from modules.utils import center_distance

def rm_duplicate(bboxes, distance_thresh=20, keep="max_conf"):
    """
    중심 좌표 거리가 `distance_thresh` 미만인 bbox 들을 하나로 병합.
    
    Args:
        bboxes (list): [[x1,y1,x2,y2] 또는 [x1,y1,x2,y2,conf], ...]
        distance_thresh (int): 병합 거리 임계값 (pixel)
        keep (str): "max_conf" | "avg"
            - "max_conf": conf 가장 큰 박스를 대표값으로 사용
            - "avg":      좌표·conf 모두 평균

    Returns:
        list: 병합된 [[x1,y1,x2,y2,conf], ...]  ← 항상 5-요소
    """

    filtered, used = [], [False] * len(bboxes)

    for i, base in enumerate(bboxes):
        if used[i]:
            continue
        group = [base]
        used[i] = True

        for j in range(i + 1, len(bboxes)):
            if used[j]:
                continue
            if center_distance(base, bboxes[j]) < distance_thresh:
                group.append(bboxes[j])
                used[j] = True

        if keep == "avg":
            arr = np.vstack([g[:5] for g in group])
            merged = arr.mean(axis=0)
            merged_box = merged[:4].astype(int).tolist() + [float(merged[4])]
        else:  # max_conf
            best = max(group, key=lambda g: g[4])
            merged_box = best[:4] + [best[4]]

        filtered.append(merged_box)

    return filtered

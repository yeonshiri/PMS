# modules/sort_tracker.py

from .reid_embedder import ReIDEmbedder
from .sort import Sort
import numpy as np

embedder = ReIDEmbedder("reid.onnx", use_gpu=False)   # 필요 시 True

# 개체별 트래커
sort_trackers = {
    "person":  Sort(max_age=5),
    "mat":     Sort(max_age=10),
    "bottle":  Sort(max_age=30)
}

MIN_HITS = {"person": 3, "mat": 3, "bottle": 3}
MIN_AGE  = {"person": 3, "mat": 3, "bottle": 3}

def track_with_sort(detections, obj_type, frame = None):
    if not detections:
        return []

    tracker = sort_trackers[obj_type]
    min_hits = MIN_HITS[obj_type]
    min_age = MIN_AGE[obj_type]
    dets = np.asarray([d[:5] for d in detections], dtype=np.float32)
        # ───────── Re-ID 테스트 ─────────
    feats = None
    if frame is not None and len(dets):
        crops = []
        for x1,y1,x2,y2,*_ in dets:
            x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
            h, w, _ = frame.shape
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w,x2), min(h,y2)
            if x2 > x1 and y2 > y1:
                crops.append(frame[y1:y2, x1:x2])
        feats = embedder(crops) if crops else None             # (N,512)
    tracker.update(dets, feats)

    results = []

    for trk in tracker.trackers:
        if (
            trk.hits >= min_hits and
            trk.time_since_update == 0 and
            trk.age >= min_age
        ):
            try:
                x1, y1, x2, y2 = np.ravel(trk.get_state())
                track_id = int(trk.id)
                results.append([int(x1), int(y1), int(x2), int(y2), track_id])
            except Exception as e:
                print(f"[WARN] unpack error: {e}")

    return results


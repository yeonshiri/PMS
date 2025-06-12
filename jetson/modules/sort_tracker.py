# modules/sort_tracker.py

from modules.sort_jet import Sort
import numpy as np

# 개체별 트래커
sort_trackers = {
    "person":  Sort(max_age=5),
    "mat":     Sort(max_age=10),
    "bottle":  Sort(max_age=30)
}

MIN_HITS = {"person": 3, "mat": 3, "bottle": 3}
MIN_AGE  = {"person": 3, "mat": 3, "bottle": 3}


def track_with_sort(detections, obj_type):
    if not detections:
        return []

    tracker = sort_trackers[obj_type]
    min_hits = MIN_HITS[obj_type]
    min_age = MIN_AGE[obj_type]
    dets = np.array(detections)
    tracker.update(dets)

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


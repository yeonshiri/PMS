from .utils import get_center, center_distance

# 튜닝 파라미터 그대로 유지
PICNIC_DIST      = 100
NEW_MAT_THRESH   = 30
MAT_CONFIRM_SEC  = 3
MAT_GONE_SEC     = 1
PICNIC_SEC       = 3
LEAVE_SEC        = 3
INSIDE_MARGIN    = 50

def point_in_rect(pt, rect, margin=0):
    x, y = pt
    x1, y1, x2, y2 = rect
    return (x1 - margin <= x <= x2 + margin) and (y1 - margin <= y <= x2 + margin)

def update_sessions(states, mat_bboxes, frame, fps):
    sessions       = states.setdefault("sessions", {})
    mat_candidates = states.setdefault("mat_candidates", {})

    confirm_frames = int(MAT_CONFIRM_SEC * fps)
    leave_frames   = int(MAT_GONE_SEC * fps)

    current_mats = {mid: mbox for mid, mbox in mat_bboxes}

    for mid, mbox in current_mats.items():
        cnt, _ = mat_candidates.get(mid, (0, mbox))
        mat_candidates[mid] = (cnt + 1, mbox)

    for mid in list(mat_candidates):
        if mid not in current_mats:
            cnt, _ = mat_candidates[mid]
            if cnt < confirm_frames:
                del mat_candidates[mid]

    for mid, (cnt, mbox) in list(mat_candidates.items()):
        if cnt < confirm_frames:
            continue

        m_center = get_center(mbox)

        if mid not in sessions:
            sessions[mid] = {
                "anchor": m_center,
                "bbox":   mbox,
                "users":  set(),
                "bottles": set(),
                "active": True,
                "last_seen": frame
            }
        else:
            sessions[mid].update(
                anchor=m_center,
                bbox=mbox,
                active=True,
                last_seen=frame
            )
            for pid in sessions[mid]["users"]:
                ps = states["person_states"].get(pid)
                if ps and ps["state"] == "finish":
                    ps.update(state="away", count_time=0, absent_time=0)

            for bid in sessions[mid]["bottles"]:
                bs = states["bottle_states"].get(bid)
                if bs and bs["state"] == ("pre", "trash"):
                    bs.update(state="away", count_time=0, absent_time=0)

        del mat_candidates[mid]

    for mid, mbox in current_mats.items():
        if mid in sessions:
            sessions[mid].update(
                anchor=get_center(mbox),
                bbox=mbox,
                active=True,
                last_seen=frame
            )

    for mid, sess in sessions.items():
        if sess["active"] and frame - sess["last_seen"] >= leave_frames:
            sess["active"] = False

    # inactive 세션에 연결된 사람/병 상태 전이
    for mid, sess in sessions.items():
        if not sess["active"]:
            for pid in sess["users"]:
                ps = states["person_states"].get(pid)
                if ps and ps["state"] != "finish":
                    ps.update(state="finish", count_time=0, absent_time=0)

            for bid in sess["bottles"]:
                bs = states["bottle_states"].get(bid)
                if bs and bs["state"] != "pre":
                    bs.update(state="pre", count_time=0, absent_time=0)

def update_person_states(states, person_bboxes, frame, fps):
    person_states     = states.setdefault("person_states", {})
    person_to_session = states.setdefault("person_to_session", {})
    sessions          = states.get("sessions", {})

    picnic_frames = int(PICNIC_SEC * fps)
    leave_frames  = int(LEAVE_SEC * fps)

    for pid, pbox in person_bboxes:
        ps = person_states.setdefault(pid, {"state": "no picnic", "count_time": 0, "absent_time": 0})

        if ps["state"] == "finish":
            continue

        if not sessions:
            ps.update(state="no picnic", count_time=0, absent_time=0)
            person_to_session.pop(pid, None)
            continue

        p_center = get_center(pbox)
        sid, _ = min(((mid, center_distance(p_center, s["anchor"])) for mid, s in sessions.items()), key=lambda t: t[1])
        sess = sessions[sid]
        inside = point_in_rect(p_center, sess["bbox"], INSIDE_MARGIN)

        if inside:
            ps["count_time"] += 1
            ps["absent_time"] = 0

            if ps["state"] in ("no picnic", "away") and ps["count_time"] >= picnic_frames:
                ps.update(state="picnic", count_time=0, absent_time=0)
                person_to_session[pid] = sid
                sess["users"].add(pid)

        else:
            ps["count_time"] = 0
            ps["absent_time"] += 1

            if ps["state"] == "picnic" and ps["absent_time"] >= leave_frames:
                ps.update(state="away", count_time=0)

        ps["last_seen"] = frame

def update_bottle_states(states, bottle_bboxes, frame, fps):
    bottle_states     = states.setdefault("bottle_states", {})
    bottle_to_session = states.setdefault("bottle_to_session", {})
    sessions          = states.get("sessions", {})

    picnic_frames = int(PICNIC_SEC * fps)
    leave_frames  = int(LEAVE_SEC * fps)

    for bid, bbox in bottle_bboxes:
        bs = bottle_states.setdefault(bid, {"state": "no picnic", "count_time": 0, "absent_time": 0})

        if bs["state"] == "trash":
            continue

        if not sessions:
            bs.update(state="no picnic", count_time=0, absent_time=0)
            bottle_to_session.pop(bid, None)
            continue

        b_center = get_center(bbox)
        closest_sid, _ = min(((mid, center_distance(b_center, s["anchor"])) for mid, s in sessions.items()), key=lambda t: t[1])
        sess = sessions[closest_sid]
        inside = point_in_rect(b_center, sess["bbox"], INSIDE_MARGIN)

        if inside:
            bs["count_time"] += 1
            bs["absent_time"] = 0

            if bs["state"] in ("no picnic", "away") and bs["count_time"] >= picnic_frames:
                bs.update(state="picnic", count_time=0, absent_time=0)
                bottle_to_session[bid] = closest_sid
                sess["bottles"].add(bid)
        else:
            bs["count_time"] = 0
            bs["absent_time"] += 1

            if bs["state"] == "picnic" and bs["absent_time"] >= leave_frames:
                bs.update(state="away", count_time=0)

        bs["last_seen"] = frame

def resolve_pre_bottles(states, person_bboxes):
    bottle_states     = states.get("bottle_states", {})
    person_states     = states.get("person_states", {})
    bottle_to_session = states.get("bottle_to_session", {})
    sessions          = states.get("sessions", {})

    person_bbox_dict  = dict(person_bboxes)

    for bid, bs in bottle_states.items():
        if bs["state"] != "pre":
            continue

        sid = bottle_to_session.get(bid)
        sess = sessions.get(sid)
        if not sess:
            continue

        has_finish_with_bbox = any(
            pid in person_bbox_dict
            for pid in sess["users"]
            if person_states.get(pid, {}).get("state") == "finish"
        )

        if not has_finish_with_bbox:
            bs.update(state="trash", count_time=0, absent_time=0)



def update_states(states, person_bboxes, mat_bboxes, bottle_bboxes, fps):
    frame = states.setdefault("frame_count", 0)
    states["frame_count"] += 1

    update_sessions(states, mat_bboxes, frame, fps)
    update_person_states(states, person_bboxes, frame, fps)
    update_bottle_states(states, bottle_bboxes, frame, fps)
    resolve_pre_bottles(states, person_bboxes)

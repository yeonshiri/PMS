from .utils import get_center, center_distance

MAT_CONFIRM_SEC  = 3    # 돗자리 생겼다고 판정하는 기준 시간
MAT_GONE_SEC     = 2    # 돗자리 사라졌다고 판정하는 기준 시간
PICNIC_SEC       = 3    # picnic 시작이라고 판정하는 기준 시간
LEAVE_SEC        = 3    # 돗자리를 벗어났다고 판정하는 기준 시간
INSIDE_MARGIN    = 50   # 돗자리 내부로 간주할 때 경계 margin

# no picnic : 처음 등장했을 때부터 돗자리에 등록되기 전까지의 상태
# picnic : 돗자리에서 3초 이상 존재해서 되는 상태
# away : 돗자리는 그대로 있고 잠시 자리비운 상태
# finish : 돗자리가 사라져서 picnic이 끝난 상태
# warning : 돗자리는 없는데 사람은 있을 수 있는 유예 상태

# 특정 point가 사각형 내부에 있는지 판단하는 함수
def point_in_rect(pt, rect, margin=0):
    x, y = pt
    x1, y1, x2, y2 = rect
    return (x1 - margin <= x <= x2 + margin) and (y1 - margin <= y <= x2 + margin)

# 돗자리 fsm
def update_sessions(states, mat_bboxes, frame, fps):            # 돗자리 세션 관리
    sessions       = states.setdefault("sessions", {})          # 확정된 돗자리는 session으로
    mat_candidates = states.setdefault("mat_candidates", {})    # 후보 돗자리는 여기로

    confirm_frames = int(MAT_CONFIRM_SEC * fps) # 3초
    leave_frames   = int(MAT_GONE_SEC * fps)    # 2초

    current_mats = {mid: mbox for mid, mbox in mat_bboxes}  # 현재 화면에 있는 돗자리들

    for mid, mbox in current_mats.items():                  # 모든 돗자리를 일단 후보로
        cnt, _ = mat_candidates.get(mid, (0, mbox))
        mat_candidates[mid] = (cnt + 1, mbox)

    for mid in list(mat_candidates):                        # 돗자리가 3초 안에 사라지면 후보 삭제
        if mid not in current_mats:
            cnt, _ = mat_candidates[mid]
            if cnt < confirm_frames:
                del mat_candidates[mid]

    for mid, (cnt, mbox) in list(mat_candidates.items()):
        if cnt < confirm_frames:
            continue

        # 돗자리가 3초 이상 유지될 때 
        m_center = get_center(mbox)

        if mid not in sessions:     # session에 없으면 추가
            sessions[mid] = {
                "anchor": m_center,
                "bbox":   mbox,
                "users":  set(),
                "bottles": set(),
                "active": True,
                "last_seen": frame
            }
        else:
            sessions[mid].update(   # 이미 있던 돗자리면 좌표 최신화
                anchor=m_center,
                bbox=mbox,
                active=True,
                last_seen=frame
            )

            # 잘못된 판정 되돌리는 코드
            for pid in sessions[mid]["users"]:                              # 해당 돗자리 사용자에 대해
                ps = states["person_states"].get(pid)                       # id 가져와서
                if ps and ps["state"] == "finish":                          # id가 있고 finish 상태면
                    ps.update(state="away", count_time=0, absent_time=0)    # away 상태로 변경

            for bid in sessions[mid]["bottles"]:                            # 해당 돗자리 물병에 대해
                bs = states["bottle_states"].get(bid)                       # id 가져와서
                if bs and bs["state"] in ("warning", "trash"):              # id가 있고 warning나 finish 상태면
                    bs.update(state="away", count_time=0, absent_time=0)    # away 상태로 변경

        del mat_candidates[mid]     # 실제 돗자리니까 후보에서는 삭제

    for mid, mbox in current_mats.items():  # 매 프레임마다 세션 위치 최신화
        if mid in sessions:
            sessions[mid].update(
                anchor=get_center(mbox),
                bbox=mbox,
                active=True,
                last_seen=frame
            ) 

    for mid, sess in sessions.items():
        if sess["active"] and frame - sess["last_seen"] >= leave_frames:    # 돗자리가 active 상태고 사라진지 2초 이상이면
            sess["active"] = False                                          # 돗자리 inactive

    # inactive 세션에 연결된 사람/물병 상태 전이
    for mid, sess in sessions.items():                                      # 모든 돗자리에 대해
        if not sess["active"]:                                              # 돗자리가 inactive면
            for pid in sess["users"]:
                ps = states["person_states"].get(pid)                       # 해당 session에 속한 사람을
                if ps and ps["state"] != "finish":                          # (이미 finish인 사람은 건너뛰기)
                    ps.update(state="finish", count_time=0, absent_time=0)  # 모두 finish 상태로 변경

            for bid in sess["bottles"]:   
                bs = states["bottle_states"].get(bid)                           # 해당 session에 속한 물병도
                if bs and bs["state"] != "warning":                             # (이미 warning인 물병 건너뛰기)
                    bs.update(state="warning", count_time=0, absent_time=0)     # 모두 warning상태로 변경

# 사람 fsm
def update_person_states(states, person_bboxes, frame, fps):
    person_states     = states.setdefault("person_states", {})
    person_to_session = states.setdefault("person_to_session", {})
    sessions          = states.get("sessions", {})

    picnic_frames = int(PICNIC_SEC * fps)   # 3초 있으면 picnic 상태
    leave_frames  = int(LEAVE_SEC * fps)    # 3초 있으면 away 상태

    for pid, pbox in person_bboxes:     # 화면에 있는 사람마다
        ps = person_states.setdefault(pid, {"state": "no picnic", "count_time": 0, "absent_time": 0, "last_seen": frame}) ###

        if ps["state"] == "finish":     # finish인 사람은 update 안함
            continue

        if not sessions:                # 화면에 돗자리가 없으면 기본 상태 유지하면서 초기화 
            ps.update(state="no picnic", count_time=0, absent_time=0)
            person_to_session.pop(pid, None)
            continue

        # 화면에 돗자리가 있고 finish가 아닐 때
        p_center = get_center(pbox)   
        # 중심좌표끼리 가장 가까운 돗자리 session 선택 
        sid, _ = min(((mid, center_distance(p_center, s["anchor"])) for mid, s in sessions.items()), key=lambda t: t[1])
        sess = sessions[sid]
        inside = point_in_rect(p_center, sess["bbox"], INSIDE_MARGIN)   # 가장 가까운 돗자리 bbox에 margin 이내로 가까워지면 inside

        if inside:                  # 돗자리 안에 있을 때
            ps["count_time"] += 1   # picnic count 시작
            ps["absent_time"] = 0   # away count 초기화

            if ps["state"] in ("no picnic", "away") and ps["count_time"] >= picnic_frames:  # no picnic이나 away상태에서 3초 이상 유지되면
                ps.update(state="picnic", count_time=0, absent_time=0)                      # picnic 상태로 변경
                person_to_session[pid] = sid                                                # PID와 SID link
                sess["users"].add(pid)                                                      # 해당 돗자리 session에 추가

        else:                       # 돗자리 밖에 있을 때
            ps["count_time"] = 0    # picnic count 초기화
            ps["absent_time"] += 1  # away count 시작

            if ps["state"] == "picnic" and ps["absent_time"] >= leave_frames:               # picnic이던 사람이고 3초 이상 나가있으면
                ps.update(state="away", count_time=0)                                       # away 상태로 변경

        ps["last_seen"] = frame     # frame update

# 물병 fsm
def update_bottle_states(states, bottle_bboxes, frame, fps):
    bottle_states     = states.setdefault("bottle_states", {})
    bottle_to_session = states.setdefault("bottle_to_session", {})
    sessions          = states.get("sessions", {})

    picnic_frames = int(PICNIC_SEC * fps)   # 3초 있으면 picnic 상태
    leave_frames  = int(LEAVE_SEC * fps)    # 3초 있으면 away 상태

    for bid, bbox in bottle_bboxes:
        bs = bottle_states.setdefault(bid, {"state": "no picnic", "count_time": 0, "absent_time": 0, "last_seen": frame}) ###

        if bs["state"] == "trash":      # trash 상태인 물병은 update 안함
            continue

        if not sessions:                # 화면에 돗자리가 없으면 기본 상태 유지하면서 초기화
            bs.update(state="no picnic", count_time=0, absent_time=0)
            bottle_to_session.pop(bid, None)
            continue

        # trash 상태가 아니고 돗자리가 있을 때
        b_center = get_center(bbox)
        # 중심좌표끼리 가장 가까운 돗자리 session 선택
        closest_sid, _ = min(((mid, center_distance(b_center, s["anchor"])) for mid, s in sessions.items()), key=lambda t: t[1])
        sess = sessions[closest_sid]
        inside = point_in_rect(b_center, sess["bbox"], INSIDE_MARGIN)   # 가장 가까운 돗자리 bbox에 margin 이내로 가까워지면 inside

        if inside:                  # 돗자리 안에 있을 때
            bs["count_time"] += 1   # picnic count 시작
            bs["absent_time"] = 0   # away count 초기화

            if bs["state"] in ("no picnic", "away") and bs["count_time"] >= picnic_frames:  # no picnic이나 away상태에서 3초 이상 유지되면
                bs.update(state="picnic", count_time=0, absent_time=0)                      # picnic 상태로 변경
                bottle_to_session[bid] = closest_sid                                        # BID와 SID link
                sess["bottles"].add(bid)                                                    # 해당 돗자리 session에 추가

        else:                       # 돗자리 밖에 있을 때
            bs["count_time"] = 0    # picnic count 초기화
            bs["absent_time"] += 1  # away count 시작

            if bs["state"] == "picnic" and bs["absent_time"] >= leave_frames:               # picnic 상태 물병이고 3초 이상 나가있으면
                bs.update(state="away", count_time=0)                                       # away 상태로 변경

        bs["last_seen"] = frame     # frame updatae

def warning_to_trash(states, person_bboxes):
    bottle_states     = states.get("bottle_states", {})
    person_states     = states.get("person_states", {})
    bottle_to_session = states.get("bottle_to_session", {})
    sessions          = states.get("sessions", {})

    person_bbox_dict  = dict(person_bboxes) # {pid: bbox, ...} 형태의 id table

    for bid, bs in bottle_states.items():   # 화면 속 물병들에 대해 
        if bs["state"] != "warning":        # warning 상태가 아니면 넘어감
            continue

        # warning 상태인 물병은
        sid = bottle_to_session.get(bid)    # ID 업로드
        sess = sessions.get(sid)            # session에 속해있었는지 확인
        if not sess:
            continue

        has_finish_with_bbox = any(
            pid in person_bbox_dict                                 
            for pid in sess["users"]                                
            if person_states.get(pid, {}).get("state") == "finish"  
        )
        # 물병이 속한 session 사용자에 대해 그 사람이 finish 상태이면서 화면에서 사라지면 False

        if not has_finish_with_bbox:
            bs.update(state="trash", count_time=0, absent_time=0)   # 조건이 맞으면 Trash 상태로 변경

# 각 state 매 프레임마다 update
def update_states(states, person_bboxes, mat_bboxes, bottle_bboxes, fps):
    frame = states.setdefault("frame_count", 0)
    states["frame_count"] += 1

    update_sessions(states, mat_bboxes, frame, fps)
    update_person_states(states, person_bboxes, frame, fps)
    update_bottle_states(states, bottle_bboxes, frame, fps)
    warning_to_trash(states, person_bboxes)

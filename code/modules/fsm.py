from .utils import get_center, center_distance

picnic_dist     = 100   # 사람-돗자리 중심 거리 임계값. 100픽셀
new_mat_thresh  = 30    # 새로운 돗자리 판단 기준 (거리)
mat_gone_sec    = 5     # 돗자리 유지 시간 (sec)
picnic_sec      = 5

# person_bboxes  = [(pid, person_bbox),  ...]
# mat_bboxes     = [(mid, mat_bbox),     ...]
# bottle_bboxes  = [(bid, bottle_bbox),  ...]

def update_states(states, person_bboxes, mat_bboxes, bottle_bboxes, fps):
    frame = states["frame_count"]
    sessions           = states.setdefault("sessions", {})
    person_states      = states.setdefault("person_states", {})
    person_to_session  = states.setdefault("person_to_session", {})
    bottle_to_session  = states.setdefault("bottle_to_session", {})

    # 1-1) 돗자리 관리 : 기존 돗자리의 중심 좌표와 threshold 이상으로 멀면 새 돗자리 session 생성
    for mid, mbox in mat_bboxes:
        m_center = get_center(mbox)
        if all(center_distance(m_center, s["anchor"]) > new_mat_thresh for s in sessions.values()):
            sessions[mid] = {
                "anchor":   m_center,
                "users":    set(),
                "bottles":  set(),
                "active":   False,
                "last_seen": frame,
            }

        # 기존에 존재하는 돗자리는 session에서 anchor 보정 & frame 갱신
        if mid in sessions:
            sessions[mid]["anchor"]    = m_center
            sessions[mid]["last_seen"] = frame

    # 1-2) 5초 이상 안 보이는 돗자리 → Session에서 비활성화
    for mid, sess in list(sessions.items()):
        if frame - sess["last_seen"] > mat_gone_sec * fps:
            sess["active"] = False

    # ───────────────── 2. 사람 ↔ Session 링크 ───────────────────
    # (거리 < PICNIC_DIST 이어서 n프레임 유지되면 picnic)
    picnic_frame = int(picnic_sec * fps)

    for pid, pbox in person_bboxes:
        # ─ person_states 초기화 ─
        ps = person_states.setdefault(
            pid, {"state": "no picnic",
                  "last_seen": frame,
                  "count_time": 0})
        
        # 세션이 하나도 없다면 바로 no-picnic 처리
        if not sessions:
            ps.update({"state":"no picnic", "last_seen":frame, "count_time":0})
            person_to_session.pop(pid, None)
            continue

        # 최소 거리인 mat 찾기
        p_center = get_center(pbox)
        sid, best_dist = min(
            ((mid, center_distance(p_center, sess["anchor"]))
                for mid, sess in sessions.items()),
            key=lambda t: t[1])
        
        # 링크 조건 만족 여부
        if best_dist < picnic_dist:
            ps["count_time"] += 1
            if ps["count_time"] >= picnic_frame:
                # picnic 확정
                ps["state"] = "picnic"
                person_to_session[pid] = sid
                sessions[sid]["users"].add(pid)
                sessions[sid]["active"] = True
        else:
            # 조건 깼으니 카운터 리셋, 상태도 no picnic
            ps["count_time"] = 0
            ps["state"]      = "no picnic"
            person_to_session.pop(pid, None)
        
        ps["last_seen"] = frame

    # ───────────────── 3. 물병 ↔ Session 링크 ──────────────────
    for bid, bbox in bottle_bboxes:
        if bid in bottle_to_session:
            # 이미 등록된 물병 → 위치 갱신만
            continue

        b_center = get_center(bbox)
        for mid, sess in sessions.items():
            if center_distance(b_center, sess["anchor"]) < picnic_dist:
                bottle_to_session[bid] = mid
                sess["bottles"].add(bid)
                break   # 하나의 세션에만 연결
                
    # ───────────────── 4. 물병 무단투기 ──────────────────       
    for mid, sess in sessions.items():
        # ① 아직 피크닉 중
        if sess["active"]:
            continue 

        # ② leave 상태 + bottles 잔존 + users 없음 → 무단투기
        still_bottles = any(bid in dict(bottle_bboxes) for bid in sess["bottles"])
        still_users   = any(pid in dict(person_bboxes) for pid in sess["users"])
        if still_bottles and (not still_users):
            states["trash_detected"] = True
            break
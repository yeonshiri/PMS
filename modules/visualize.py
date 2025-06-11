import cv2
from .utils import get_center

# 색 팔레트 (BGR)
COLOR_PERSON  = (  0,200,255)   # 주황
COLOR_MAT     = (  0,255,  0)   # 초록
COLOR_BOTTLE  = (255,  0,  0)   # 파랑
COLOR_TEXT_BG = ( 50, 50, 50)

def drawing(frame, persons, mats, bottles, states):
    # 현재 프레임에 감지된 ID 수집
    current_person_ids = set(pid for pid, _ in persons)
    current_bottle_ids = set(bid for bid, _ in bottles)

    # 1) 돗자리 (세션) 그리기 ──────────────────────────
    for mid, mb in mats:
        x1,y1,x2,y2 = mb
        cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_MAT, 2)
        cv2.putText(frame, f"Mat {mid}", (x1, y2+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAT, 1)

    # 2) 사람 ─────────────────────────────────────────
    for pid, pb in persons:
        x1,y1,x2,y2 = pb
        p_state = states["person_states"].get(pid, {}).get("state", "no")
        cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_PERSON, 2)
        cv2.putText(frame, f"ID {pid} : {p_state}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_PERSON, 2)

        # anchor 라인: 사람 → 돗자리 중심
        sid = states["person_to_session"].get(pid)
        if sid:
            anchor = states["sessions"][sid]["anchor"]
            pc     = get_center(pb)
            cv2.line(frame, pc, anchor, (180,180,180), 1)

    # 3) 물병 ──────────────────────────────────────────
    bottle_states = states.get("bottle_states", {})
    for bid, bb in bottles:
        x1, y1, x2, y2 = bb
        b_state = bottle_states.get(bid, {}).get("state", "no")

        # 상태에 따라 색상 설정
        if b_state == "pre":
            color = (0, 165, 255)  # 주황 (Orange)
        elif b_state == "trash":
            color = (0, 0, 255)    # 빨강 (Red)
        else:
            color = COLOR_BOTTLE  # 기본 파랑

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Btl {bid} : {b_state}", (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 4) 무단투기 알림 ────────────────────────────────
    if states.get("trash_detected", False):
        text = "Trash detected!"
        (tw,th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cv2.rectangle(frame, (20,20-th), (20+tw+10, 20+10), COLOR_TEXT_BG, -1)
        cv2.putText(frame, text, (25,20+th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

    # 5) 디버그 정보 오버레이 ─────────────────────────
    y = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    sessions = states.get("sessions", {})
    person_states = states.get("person_states", {})
    person_to_session = states.get("person_to_session", {})
    bottle_states = states.get("bottle_states", {})
    bottle_to_session = states.get("bottle_to_session", {})

    # 세션 상태
    for mid, sess in sessions.items():
        user_infos = [
            f"{pid}({person_states.get(pid, {}).get('state', '?')})"
            for pid in sess['users']
        ]
        bottle_infos = [
            f"{bid}({bottle_states.get(bid, {}).get('state', '?')})"
            for bid in sess['bottles']
        ]
        text = f"[S{mid}] active={sess['active']}\n  users=[{', '.join(user_infos)}]\n  bottles=[{', '.join(bottle_infos)}]"
        for line in text.split('\n'):
            cv2.putText(frame, line, (10, y), font, 0.5, (0,255,0), 1)
            y += 20

    y += 10  # 간격

    # 사람 상태 따로 출력 (현재 프레임에 감지된 사람만)
    for pid, ps in person_states.items():
        if pid not in current_person_ids:
            continue
        sid = person_to_session.get(pid, "None")
        text = f"[P{pid}] {ps['state']:<9} c={ps['count_time']:>2} a={ps['absent_time']:>2} s={sid}"
        color = (255,255,255)
        if ps['state'] == "finish":
            color = (0,165,255)
        elif ps['state'] == "picnic":
            color = (0,255,255)
        cv2.putText(frame, text, (10, y), font, 0.5, color, 1)
        y += 20

    # 병 상태 따로 출력 (현재 프레임에 감지된 병만)
    for bid, bs in bottle_states.items():
        if bid not in current_bottle_ids:
            continue
        sid = bottle_to_session.get(bid, "None")
        text = f"[B{bid}] {bs['state']:<9} c={bs['count_time']:>2} a={bs['absent_time']:>2} s={sid}"
        
        # 상태에 따라 로그 색상 설정
        if bs['state'] == "trash":
            color = (0, 0, 255)        # 빨강
        elif bs['state'] == "pre":
            color = (0, 165, 255)      # 주황
        elif bs['state'] == "picnic":
            color = (255, 255, 0)      # 노랑
        else:
            color = (255, 255, 255)    # 기본 흰색

        cv2.putText(frame, text, (10, y), font, 0.5, color, 1)
        y += 20
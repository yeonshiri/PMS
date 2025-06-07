import cv2
from .utils import get_center

# 색 팔레트 (BGR)
COLOR_PERSON  = (  0,200,255)   # 주황
COLOR_MAT     = (  0,255,  0)   # 초록
COLOR_BOTTLE  = (255,  0,  0)   # 파랑
COLOR_TEXT_BG = ( 50, 50, 50)

def drawing(frame, persons, mats, bottles, states):
    # """시각화 함수
    # persons, mats, bottles : [(id, (x1,y1,x2,y2)), ...]
    # states                 : FSM 상태 딕셔너리
    # """
    # 1) 돗자리 (세션) 먼저 그리기  ──────────────────────────
    for mid, mb in mats:
        x1,y1,x2,y2 = mb
        cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_MAT, 2)
        cv2.putText(frame, f"Mat {mid}", (x1, y2+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAT, 1)

    # 2) 사람  ────────────────────────────────────────────
    for pid, pb in persons:
        x1,y1,x2,y2 = pb
        p_state = states["person_states"].get(pid, {}).get("state", "no")
        cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_PERSON, 2)
        cv2.putText(frame, f"ID {pid} : {p_state}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_PERSON, 2)

        # anchor 라인(선택) : 사람 → 해당 돗자리 중심
        sid = states["person_to_session"].get(pid)
        if sid:
            anchor = states["sessions"][sid]["anchor"]
            pc     = get_center(pb)
            cv2.line(frame, pc, anchor, (180,180,180), 1)

    # 3) 물병  ─────────────────────────────────────────────
    for bid, bb in bottles:
        x1,y1,x2,y2 = bb
        cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_BOTTLE, 2)
        cv2.putText(frame, f"Btl {bid}", (x1, y2+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BOTTLE, 1)

    # 4) 무단투기 알림  ───────────────────────────────────
    if states.get("trash_detected", False):
        text = "Trash detected!"
        (tw,th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cv2.rectangle(frame, (20,20-th), (20+tw+10, 20+10), COLOR_TEXT_BG, -1)
        cv2.putText(frame, text, (25,20+th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

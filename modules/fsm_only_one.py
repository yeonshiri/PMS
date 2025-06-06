import numpy as np
from .utils import get_center, center_distance, iou

# state update logic + 무단투기 판단 logic
def update_states(states, person_bboxes, mat_bboxes, bottle_bboxes, fps):
    frame = states["frame_count"]                                   # 현재 frame 값 저장

    if mat_bboxes:                                                  # 돗자리가 화면에 감지 될 때
        latest_mat = mat_bboxes[-1]                                 # 가장 최근에 추가된 돗자리 지정

        if states["mat_visible_since"] is None:                     # 돗자리가 처음 나타난거면
            states["mat_visible_since"] = frame                     # 해당 frame값으로 시간 저장

        elif (frame - states["mat_visible_since"]) > fps * 3:       # 돗자리 bbox가 3초 이상 유지되면
            states["picnic_active"] = True                          # picnic signal 활성화
            states["mat_anchors"].append(get_center(latest_mat))    # 최신 돗자리의 중심 좌표 저장
            states["registered_bottle_anchors"] += [              
                get_center(b) for b in bottle_bboxes
                if center_distance(b, latest_mat) < 100             # 최신 돗자리 중심 좌표에서 가까운 물병들은 물병 리스트에 등록
            ]
        states["mat_disappear_since"] = None                        # 돗자리 bbox가 나타났으니 돗자리 사라짐 signal 비활성화

    else:                                                                           # 감지되는 돗자리가 없으면
        states["mat_visible_since"] = None                                          # 나타남 signal 비활성화
        if states["picnic_active"] and states["mat_disappear_since"] is None:       # picnic signal이 활성화 되어있고 돗자리 사라짐 signal 비활성화면
            states["mat_disappear_since"] = frame                                   # 피크닉 종료 시점으로 판단 --> 해당 frame값으로 시간 저장

        elif states["picnic_active"] and states["mat_disappear_since"] is not None: # picnic signal이 활성화 되어있고 돗자리 사라짐 signal 활성화면
            if (frame - states["mat_disappear_since"]) > fps * 3:                   # 3초 이상 사라졌을 때
                states["picnic_active"] = False                                     # picnic signal 비활성화
                states["mat_anchor"] = None                                         # 돗자리 중심좌표 초기화
                states["mat_disappear_since"] = None                                # 돗자리 사라짐 signal 비활성화

    # 무단투기 상태 판단
    states["trash_detected"] = False
    if not states["picnic_active"] and states["registered_bottle_anchors"]:     # picnic signal 비활성화 + 물병 리스트에 등록된 값이 있으면
        for b in bottle_bboxes:                                                 # 각 물병에 대해
            b_center = get_center(b)
            for saved in states["registered_bottle_anchors"]:                   
                if np.linalg.norm(np.array(b_center) - np.array(saved)) < 30:   # 리스트에 등록된 중심좌표와 현재 물병 중심좌표 거리가 가까우면
                    states["trash_detected"] = True                             # 무단 투기 상태 signal 활성화

# 개별 person state logic (IOU + 거리 조건)
def get_person_state(person_bboxes, mat_bboxes, picnic_active):
    for i in enumerate(person_bboxes):
        if not picnic_active:       # picnic signal 비활성화일 때
            return "no picnic"      # no picnic 상태
        
        for mat in mat_bboxes:      # picnic signal 활성화일 때 각 돗자리에 대해
            if center_distance(person_bboxes, mat) < 100 or iou(person_bboxes, mat) > 0.1:
                return "picnic"     # 좌표 중심 거리 or IOU 조건이 맞으면 picnic 상태
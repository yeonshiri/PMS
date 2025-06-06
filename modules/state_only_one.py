# 초기 signal 및 변수 초기화
def initialize_states():
    return {
        "frame_count": 0,                   # 현재 프레임 번호(시간 추적용)
        "mat_visible_since": None,       # 돗자리가 처음 감지될 때 signal 리스트
        "mat_disappear_since": None,     # 돗자리가 사라졌을 때 signal 리스트
        "picnic_active": None,           # picnic 상태 활성화 여부 signal 리스트
        "mat_anchors": [],                  # 돗자리의 중심 좌표 리스트
        "registered_bottle_anchors": [],    # 물병 중심 좌표를 등록할 리스트
        "trash_detected": False,            # 무단투기 발생 여부 signal
    }
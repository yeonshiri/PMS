# 초기 signal 및 변수 초기화
def initialize_states():
    return {
        "frame_count": 0,                 # 현재 프레임 번호(시간 추적용)
        "sessions": {},             # 돗자리 session 관리
        "person_states": {},        # 사람별 ID 관리 딕셔너리. {id: {"state":"no picnic", "last_seen":frame}} 형태
        "person_to_session": {},    # 사람 ID와 돗자리 ID의 link를 관리하는 table
        "bottle_to_session": {},    # 물병 ID와 돗자리 ID와 link를 관리하는 table
        "trash_detected": False,    # 무단투기 발생 여부 signal
    }

# {
#   "sessions": {          # mat_id ▶ Picnic Session
#       17: {
#           "anchor": (cx,cy),
#           "users":   {3, 8},
#           "bottles": {21, 25},
#           "active":  True,
#           "last_seen": frame
#       },
#       ...
#   },
#   "person_to_session": {},   # person_id ▶ mat_id
#   "bottle_to_session": {},   # bottle_id ▶ mat_id
# }
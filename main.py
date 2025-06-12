# main.py ──────────────────────────────────────────────────────────────
from modules.state      import initialize_states        # 초기 상태 초기화
from modules.detection  import yolo_to_deepsort         # YOLO 결과 → [x1,y1,x2,y2,conf,label] 리스트
from modules.fsm        import update_states            # FSM 갱신
from modules.visualize  import drawing                  # 화면 그리기
from modules.sort_tracker import track_with_sort        # 클래스별 SORT 추적기

import cv2
import time
from ultralytics import YOLO

MODEL_PATH = "final.pt"     # 학습된 YOLO 가중치
VIDEO_PATH = "2.mp4"        # 비디오 파일 (실시간이면 0)

# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # 0) 모델 및 상태 초기화
    model  = YOLO(MODEL_PATH)
    states = initialize_states()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open video source: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # 간혹 0.0이 나오는 카메라도 있어서 기본 30FPS
    frame_interval = 1.0 / fps
    prev_time = time.time()

    # 1) 메인 루프
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1-1) YOLO 추론
        results = model(frame)[0]

        # ① YOLO box에서 클래스 ID 포함해 새 리스트 만들기
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf  = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf, cls_id])

        # ② 클래스 매핑
        LABELS = {0: "person", 1: "mat", 2: "bottle"}

        # ③ 필터링
        det_person = [d[:5] for d in detections if LABELS[d[5]] == "person"]
        det_mat    = [d[:5] for d in detections if LABELS[d[5]] == "mat"]
        det_bottle = [d[:5] for d in detections if LABELS[d[5]] == "bottle"]

        # 1-3) SORT 추적기로 ID 부여
        trk_person, _  = track_with_sort(det_person,  obj_type="person")
        trk_mat,    _  = track_with_sort(det_mat,     obj_type="mat")
        trk_bottle, _  = track_with_sort(det_bottle,  obj_type="bottle")

        # 1-4) (id, bbox) 튜플 리스트로 변환
        person_bboxes = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_person]
        mat_bboxes    = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_mat]
        bottle_bboxes = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_bottle]

        # 1-5) FSM 업데이트 + 시각화
        update_states(states, person_bboxes, mat_bboxes, bottle_bboxes, fps)
        drawing(frame, person_bboxes, mat_bboxes, bottle_bboxes, states)

        # 1-6) 화면 출력
        cv2.imshow("Picnic Trash Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # 1-7) 원본 FPS 맞추기
        elapsed = time.time() - prev_time
        if elapsed < frame_interval:                 # 실제 처리 속도가 더 빠를 때만 sleep
            time.sleep(frame_interval - elapsed)
        prev_time = time.time()

        states["frame_count"] += 1                   # (옵션) 통계용

    # 2) 종료 처리
    cap.release()
    cv2.destroyAllWindows()

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

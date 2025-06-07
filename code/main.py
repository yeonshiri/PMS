from modules.state      import initialize_states                     # 초기 상태 초기화
from modules.detection  import yolo_to_deepsort     # 사람, 돗자리, 물병 bbox 리스트 추출 + deepsort용 리스트 추출
from modules.fsm        import update_states
from modules.visualize  import drawing

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model_path = "mat_tune_1.pt"    # pt 경로
video_path = "video1.mp4"       # 비디오 경로 ... (나중에 실시간 처리로 변경)

def main():
    model = YOLO(model_path)
    tracker = DeepSort(embedder="disabled", max_age=30) # 젯슨나노 실행을 위해 REID 기능 비활성화

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    states = initialize_states()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = yolo_to_deepsort(results)
        # person_bboxes, mat_bboxes, bottle_bboxes = extract_bboxes(results)
        tracks = tracker.update_tracks(detections, frame=frame)

        # 3) track → 클래스별 (id, bbox) 리스트로 분리
        person_bboxes, mat_bboxes, bottle_bboxes = [], [], []
        for trk in tracks:
            if not trk.is_confirmed():  # 1프레임짜리는 skip
                continue
            tid = trk.track_id     # 고유 ID
            cls = trk.det_class    # "person" | "mat" | "bottle"
            x1, y1, x2, y2 = map(int, trk.to_ltrb())
            bbox = (x1, y1, x2, y2)

            if cls == "person":
                person_bboxes.append((tid, bbox))
            elif cls == "mat":
                mat_bboxes.append((tid, bbox))
            elif cls == "bottle":
                bottle_bboxes.append((tid, bbox))

        # 4) FSM 업데이트 (ID 포함 버전)
        update_states(states, person_bboxes, mat_bboxes, bottle_bboxes, fps)

        # 5) 시각화 (ID 전달)
        drawing(frame, person_bboxes, mat_bboxes, bottle_bboxes, states)

        cv2.imshow("Picnic Trash Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        states["frame_count"] += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

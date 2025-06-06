from modules.state_only_one import initialize_states             # 초기 상태 초기화
from modules.detection_only_one import extract_bboxes            # 사람, 돗자리, 물병 bbox 리스트 추출
from modules.fsm_only_one import update_states
from modules.visualize_only_one import drawing
import cv2
import numpy as np
from ultralytics import YOLO

model_path = "mat_tune_3.pt"    # pt 경로
video_path = "1.mp4"       # 비디오 경로 ... (나중에 실시간 처리로 변경)

def main():
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    states = initialize_states()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 0)  # 수직 뒤집기 (영상이 상하반전일 경우)

        results = model(frame)[0]
        person_bboxes, mat_bboxes, bottle_bboxes = extract_bboxes(results)
        update_states(states, person_bboxes, mat_bboxes, bottle_bboxes, fps)
        drawing(frame, person_bboxes, mat_bboxes, bottle_bboxes, states)

        cv2.imshow("Picnic Trash Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        states["frame_count"] += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

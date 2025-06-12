from modules.state import initialize_states
from modules.detection import yolo_to_sort
from modules.fsm import update_states
from modules.visualize import drawing
from modules.sort_tracker import track_with_sort
from modules.clean_bbox import rm_duplicate

import cv2
import time
import threading
import queue
import torch
import pathlib

MODEL_PATH = "best_yes_demo.pt"
VIDEO_PATH = "video9.mp4"  # 웹캠 사용 시 0
BUFFER_SIZE = 1

def yolo_worker(frame_q: "queue.Queue[tuple[cv2.Mat,float]]", result_dict: dict):
    pathlib.PosixPath = pathlib.WindowsPath
    model = torch.hub.load('ultralytics/yolov5', 'custom',
            path=MODEL_PATH,        # final.pt 경로
            force_reload=False)     # 이미 다운로드된 경우 재다운로드 방지
    states = initialize_states()

    while True:
        item = frame_q.get()
        if item is None:
            break
        frame, orig_fps = item

        # 1) YOLO 추론
        dets = model(frame, size=frame.shape[0])      # Detections 객체 반환
        pred = dets.pred[0]                           # (N×6) 텐서: [x1,y1,x2,y2,conf,cls]
        results = []
        for *xyxy, conf, cls in pred.cpu().numpy():
            cls = int(cls)
            name = model.names[cls]                   # 클래스 이름 매핑
            results.append([*xyxy, float(conf), name])

        # 3) DeepSORT 변환
        detections = yolo_to_sort(results)

        # 2) 클래스별 필터링 + 중복 제거
        det_person = rm_duplicate([d[:5] for d in detections if d[5] == "person"], 20, "max_conf")
        det_mat    = rm_duplicate([d[:5] for d in detections if d[5] == "mat"],    20, "max_conf")
        det_bottle = rm_duplicate([d[:5] for d in detections if d[5] == "bottle"], 20, "max_conf")

        # 3) SORT 추적
        trk_person = track_with_sort(det_person, "person",  frame)
        trk_mat    = track_with_sort(det_mat,    "mat",     frame)
        trk_bottle = track_with_sort(det_bottle, "bottle",  frame)

        person_bb  = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_person]
        mat_bb     = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_mat]
        bottle_bb  = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_bottle]

        # 4) FSM 상태 업데이트
        update_states(states, person_bb, mat_bb, bottle_bb, orig_fps)

        # 5) 최신 결과 저장
        result_dict["bboxes"] = (person_bb, mat_bb, bottle_bb)
        result_dict["states"] = states.copy()

def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_PATH}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = 1.0 / orig_fps

    frame_q: "queue.Queue[tuple[cv2.Mat,float]]" = queue.Queue(maxsize=BUFFER_SIZE)
    result_dict: dict = {"bboxes": ([], [], []), "states": {}}

    worker = threading.Thread(target=yolo_worker, args=(frame_q, result_dict), daemon=True)
    worker.start()

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 최신 프레임만 유지
        if not frame_q.empty():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put((frame.copy(), orig_fps))

        # 추론 결과 시각화
        person_bb, mat_bb, bottle_bb = result_dict.get("bboxes", ([], [], []))
        states                       = result_dict.get("states",  {})
        drawing(frame, person_bb, mat_bb, bottle_bb, states)

        # 화면 출력
        cv2.imshow("Picnic Trash Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # FPS 고정
        elapsed = time.time() - prev_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        prev_time = time.time()

    # 종료 처리
    frame_q.put(None)
    worker.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

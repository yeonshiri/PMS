import cv2
import time
import threading
import queue
import numpy as np
from modules.state import initialize_states
from modules.detection import yolo_to_deepsort
from modules.fsm import update_states
from modules.visualize import drawing
from modules.sort_tracker import track_with_sort
from modules.clean_bbox import rm_duplicate

MODEL_PATH = "picnic_5n.engine"
VIDEO_PATH = "3.mp4"
BUFFER_SIZE = 1


def yolo_worker(frame_q: queue.Queue, result_dict: dict):
    from modules.detect import TRTInfer  # ⚠️ worker 안에서 import (pycuda.autoinit 분리 보장)
    model = TRTInfer(MODEL_PATH)        # ✅ context 포함 객체 생성은 스레드 안에서!

    states = initialize_states()

    while True:
        item = frame_q.get()
        if item is None:
            break

        frame, orig_fps = item

        raw_detections = model.infer(frame)
        detections = yolo_to_deepsort(raw_detections)

        det_person = rm_duplicate([d[:5] for d in detections if d[5] == "person"], 20, "max_conf")
        det_mat    = rm_duplicate([d[:5] for d in detections if d[5] == "mat"],    20, "max_conf")
        det_bottle = rm_duplicate([d[:5] for d in detections if d[5] == "bottle"], 20, "max_conf")

        trk_person = track_with_sort(det_person, "person")
        trk_mat    = track_with_sort(det_mat,    "mat")
        trk_bottle = track_with_sort(det_bottle, "bottle")

        person_bb = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_person]
        mat_bb    = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_mat]
        bottle_bb = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_bottle]

        update_states(states, person_bb, mat_bb, bottle_bb, orig_fps)

        result_dict["bboxes"] = (person_bb, mat_bb, bottle_bb)
        result_dict["states"] = states.copy()

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open video source: {VIDEO_PATH}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps == 0 or np.isnan(orig_fps):
        orig_fps = 30
    frame_interval = 1.0 / orig_fps

    frame_q = queue.Queue(maxsize=BUFFER_SIZE)
    result_dict = {"bboxes": ([], [], []), "states": {}}

    worker = threading.Thread(target=yolo_worker, args=(frame_q, result_dict), daemon=True)
    worker.start()

    prev_time = time.time()
    total_frames = 0
    total_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Webcam read failed.")
            break

        start_time = time.time()

        if not frame_q.empty():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put((frame.copy(), orig_fps))

        person_bb, mat_bb, bottle_bb = result_dict.get("bboxes", ([], [], []))
        states = result_dict.get("states", {})
        drawing(frame, person_bb, mat_bb, bottle_bb, states)

        # FPS 계산 및 누적
        elapsed = time.time() - start_time
        total_frames += 1
        total_time += elapsed
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("🧺 Picnic Trash Detection – Realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        elapsed_total = time.time() - prev_time
        if elapsed_total < frame_interval:
            time.sleep(frame_interval - elapsed_total)
        prev_time = time.time()

    # 종료 후 평균 FPS 출력
    avg_fps = total_frames / total_time if total_time > 0 else 0
    print(f"✅ 평균 FPS: {avg_fps:.2f} over {total_frames} frames.")

    frame_q.put(None)
    worker.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

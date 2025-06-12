import cv2
import time
import threading
import queue
import torch
import pathlib
import statistics

from modules.state import initialize_states
from modules.detection import yolo_to_deepsort
from modules.fsm import update_states
from modules.visualize import drawing
from modules.sort_tracker import track_with_sort
from modules.clean_bbox import rm_duplicate

MODEL_PATH = "picnic_5n.pt"
VIDEO_PATH = "3.mp4"
BUFFER_SIZE = 1

def yolo_worker(frame_q: "queue.Queue[tuple[cv2.Mat,float]]", result_dict: dict):
    pathlib.PosixPath = pathlib.WindowsPath  # for Windows compatibility
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=MODEL_PATH, force_reload=False)
    states = initialize_states()

    while True:
        item = frame_q.get()
        if item is None:
            break
        frame, orig_fps = item

        start_infer = time.time()
        dets = model(frame, size=frame.shape[0])
        pred = dets.pred[0]
        results = []
        for *xyxy, conf, cls in pred.cpu().numpy():
            cls = int(cls)
            name = model.names[cls]
            results.append([*xyxy, float(conf), name])
        infer_time = time.time() - start_infer
        result_dict["infer_time"] = infer_time

        detections = yolo_to_deepsort(results)

        det_person = rm_duplicate([d[:5] for d in detections if d[5] == "person"], 20, "max_conf")
        det_mat    = rm_duplicate([d[:5] for d in detections if d[5] == "mat"],    20, "max_conf")
        det_bottle = rm_duplicate([d[:5] for d in detections if d[5] == "bottle"], 20, "max_conf")

        trk_person = track_with_sort(det_person, "person")
        trk_mat    = track_with_sort(det_mat,    "mat")
        trk_bottle = track_with_sort(det_bottle, "bottle")

        person_bb  = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_person]
        mat_bb     = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_mat]
        bottle_bb  = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_bottle]

        update_states(states, person_bb, mat_bb, bottle_bb, orig_fps)

        result_dict["bboxes"] = (person_bb, mat_bb, bottle_bb)
        result_dict["states"] = states.copy()

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open video source: {VIDEO_PATH}")

    frame_q = queue.Queue(maxsize=BUFFER_SIZE)
    result_dict = {"bboxes": ([], [], []), "states": {}, "infer_time": 0.0}
    inf_fps_list = []

    worker = threading.Thread(target=yolo_worker, args=(frame_q, result_dict), daemon=True)
    worker.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 최신 프레임으로 큐 갱신
        if not frame_q.empty():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put((frame.copy(), 30))  # dummy fps

        # 추론 결과 및 상태 가져오기
        person_bb, mat_bb, bottle_bb = result_dict.get("bboxes", ([], [], []))
        states = result_dict.get("states", {})
        infer_time = result_dict.get("infer_time", 0.0)

        # INF FPS 기록
        if infer_time > 0:
            inf_fps_list.append(1.0 / infer_time)

        # 시각화
        drawing(frame, person_bb, mat_bb, bottle_bb, states)
        cv2.putText(frame, f"Infer: {infer_time * 1000:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 화면 출력
        cv2.imshow("Picnic Trash Detection (PT)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 종료 처리
    frame_q.put(None)
    worker.join()
    cap.release()
    cv2.destroyAllWindows()

    # 평균 추론 FPS 출력
    if inf_fps_list:
        print("\n📊 PyTorch (.pt) 모델 추론 성능 요약")
        print(f"✅ 평균 INF FPS: {statistics.mean(inf_fps_list):.2f}")
        print(f"📈 최대 INF FPS: {max(inf_fps_list):.2f}")
        print(f"📉 최소 INF FPS: {min(inf_fps_list):.2f}")

if __name__ == "__main__":
    main()

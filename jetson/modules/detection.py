# modules/detection.py

def yolo_to_deepsort(results, classes_of_interest=("person", "bottle", "mat")):
    """
    TensorRT 결과 → [[x1, y1, x2, y2, conf, label_str], ...]
    classes_of_interest: 추적할 클래스 문자열 목록
    """
    detections = []
    for det in results:
        if len(det) != 6:
            continue
        x1, y1, x2, y2, conf, label = det
        if label not in classes_of_interest:
            continue
        detections.append([x1, y1, x2, y2, conf, label])
    return detections

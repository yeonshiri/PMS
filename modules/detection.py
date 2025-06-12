def yolo_to_deepsort(results, classes_of_interest=("person", "bottle", "mat")):
    """
    TensorRT 추론 결과 → [[x1, y1, x2, y2, conf, label_str], ...]
    - results: List of [x1, y1, x2, y2, conf, label_str]
    - classes_of_interest: 추적할 클래스 이름 튜플
    """
    detections = []

    for det in results:
        x1, y1, x2, y2, conf, label = det
        if label not in classes_of_interest:
            continue

        detections.append([
            int(x1), int(y1), int(x2), int(y2),
            float(conf), label
        ])

    return detections

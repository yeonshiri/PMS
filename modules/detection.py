# modules/detection.py
def yolo_to_deepsort(results, classes_of_interest=(0, 2, 1)):
    """
    Ultralytics YOLO 결과 → [[x1, y1, x2, y2, conf, label_str], ...]
    - classes_of_interest: 추적할 클래스 ID 튜플
      COCO 기준 0=person, 39=bottle. 80은 사용자 정의 id로 'mat'로 가정.
    """
    detections = []

    for box, conf, cls in zip(results.boxes.xyxy,
                              results.boxes.conf,
                              results.boxes.cls):
        cls_id = int(cls.item())
        if cls_id not in classes_of_interest:
            continue                        # 관심 없는 클래스 skip

        # bbox 좌표
        x1, y1, x2, y2 = map(int, box.cpu().numpy())

        # 신뢰도
        score = float(conf.item())

        # 클래스 → 문자열
        label = (
            "person"  if cls_id == 0  else
            "mat"     if cls_id == 2 else
            "bottle"  if cls_id == 1 else
            str(cls_id)
        )

        # track_with_sort() 에서 앞 5개만 사용, main.py 에서 6번째(label)로 필터
        detections.append([x1, y1, x2, y2, score, label])

    return detections

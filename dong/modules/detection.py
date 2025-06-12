# modules/detection.py
def yolo_to_sort(results,
                     classes_of_interest=(0, 1, 2),
                     model_names=None):
    """
    YOLO 추론 결과 → [[x1, y1, x2, y2, conf, label_str], ...]  로 변환
    - results: ① YOLOv5 Detections 객체 ② list형 [(x1,y1,x2,y2,conf,label/cls), ...]
    - classes_of_interest: 추적할 클래스 인덱스 튜플
      (기본 0=person, 1=bottle, 2=mat)
    - model_names: 클래스 인덱스→이름 매핑(dict 또는 list), 없으면 label_str 사용
    """
    detections = []

    # ─────────────────────────────────────────────
    # 1) YOLOv5 · YOLOv8 Detections 객체 형태 처리
    # ─────────────────────────────────────────────
    if hasattr(results, "boxes"):
        # 각 속성: 텐서
        for box, conf, cls in zip(results.boxes.xyxy,
                                  results.boxes.conf,
                                  results.boxes.cls):
            cls_id = int(cls.item())
            if cls_id not in classes_of_interest:
                continue

            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            score = float(conf.item())
            # 이름 매핑
            label = (
                model_names[cls_id] if model_names else
                ("person" if cls_id == 0 else
                 "bottle" if cls_id == 1 else
                 "mat"    if cls_id == 2 else
                 str(cls_id))
            )

            detections.append([x1, y1, x2, y2, score, label])
        return detections

    # ─────────────────────────────────────────────
    # 2) 이미 리스트 형태로 (x1,y1,x2,y2,conf,label/cls) 들어온 경우
    # ─────────────────────────────────────────────
    if isinstance(results, list):
        for item in results:
            # item = [x1,y1,x2,y2,conf,label]  또는 label 자리에 cls(int)
            x1, y1, x2, y2, score, lab = item

            # lab이 정수면 cls, 문자열이면 이미 라벨
            if isinstance(lab, int):
                cls_id = lab
                if cls_id not in classes_of_interest:
                    continue
                label = (
                    model_names[cls_id] if model_names else
                    ("person" if cls_id == 0 else
                     "bottle" if cls_id == 1 else
                     "mat"    if cls_id == 2 else
                     str(cls_id))
                )
            else:
                # 문자열일 땐 그대로 라벨 사용
                label = lab
            detections.append([int(x1), int(y1), int(x2), int(y2),
                               float(score), label])
        return detections

    # ─────────────────────────────────────────────
    # 그 외 형식 → 빈 리스트 반환
    # ─────────────────────────────────────────────
    return detections

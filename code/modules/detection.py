# YOLOv8n 모델에서 객체별 bbox 추출 + tracking ID 부여를 위한 단계

# DeepSORT 입력용 리스트 반환
def yolo_to_deepsort(results, classes_of_interest=(0, 80, 39)):

    detections = []

    for box, conf, cls in zip(results.boxes.xyxy,
                              results.boxes.conf,
                              results.boxes.cls):
        cls_id = int(cls.cpu().numpy())
        if cls_id not in classes_of_interest:
            continue

        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        w, h = x2 - x1, y2 - y1
        class_name = (
            "person" if cls_id == 0 else
            "mat"    if cls_id == 80 else
            "bottle" if cls_id == 39 else
            str(cls_id)
        )
        detections.append(([x1, y1, w, h], conf.cpu().item(), class_name))

    return detections


# def extract_bboxes(results):
#     person_bboxes = []
#     mat_bboxes = []
#     bottle_bboxes = []

#     for box, cls in zip(results.boxes.xyxy, results.boxes.cls): # 각 객체의 bbox 좌표, class값
#         x1, y1, x2, y2 = map(int, box.cpu().numpy())
#         cls_id = int(cls.cpu().numpy())
#         if   cls_id == 0:   # 사람
#             person_bboxes.append((x1, y1, x2, y2))
#         elif cls_id == 80:  # 돗자리
#             mat_bboxes.append((x1, y1, x2, y2))
#         elif cls_id == 39:  # 물병
#             bottle_bboxes.append((x1, y1, x2, y2))

#     return person_bboxes, mat_bboxes, bottle_bboxes

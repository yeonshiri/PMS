# YOLOv8n 모델에서 객체별 bbox 추출
def extract_bboxes(results):
    person_bboxes = []
    mat_bboxes = []
    bottle_bboxes = []

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls): # 각 객체의 bbox 좌표, class값
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        cls_id = int(cls.cpu().numpy())
        if   cls_id == 0:   # 사람
            person_bboxes.append((x1, y1, x2, y2))
        elif cls_id == 80:  # 돗자리
            mat_bboxes.append((x1, y1, x2, y2))
        elif cls_id == 39:  # 물병
            bottle_bboxes.append((x1, y1, x2, y2))

    return person_bboxes, mat_bboxes, bottle_bboxes

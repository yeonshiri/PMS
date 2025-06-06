import cv2
from .fsm_only_one import get_person_state

def drawing(frame, person_bboxes, mat_bboxes, bottle_bboxes, states):
    for i, person in enumerate(person_bboxes):
        state = get_person_state(person, mat_bboxes, states["picnic_active"])
        x1, y1, x2, y2 = person
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f"Person {i+1} - {state}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    for x1, y1, x2, y2 in mat_bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "mat", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for x1, y1, x2, y2 in bottle_bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "bottle", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if states["trash_detected"]:
        cv2.putText(frame, "⚠ 무단투기 발생", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

import cv2

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
]


def draw_detections(image, detections, score_threshold=0.5):

    boxes = detections[0]["boxes"].cpu().numpy()
    labels = detections[0]["labels"].cpu().numpy()
    scores = detections[0]["scores"].cpu().numpy()

    img = image.copy()

    for box, label, score in zip(boxes, labels, scores):

        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)

        label_name = (
            COCO_CLASSES[label]
            if label < len(COCO_CLASSES)
            else f"class_{label}"
        )

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

        text = f"{label_name} {score:.2f}"

        cv2.putText(
            img,
            text,
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

    return img
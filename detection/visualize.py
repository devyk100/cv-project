import cv2
import torch
from torchvision.ops import nms

COCO_CLASSES = [
"__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
"train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
"parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
"tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
"baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
"donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
"toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
"vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def draw_detections(image, detections, score_threshold=0.10): # change this for better results

    boxes = detections[0]["boxes"].detach().cpu().numpy()
    labels = detections[0]["labels"].detach().cpu().numpy()
    scores = detections[0]["scores"].detach().cpu().numpy()

    # ---------- ADD THIS BLOCK HERE ----------
    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)

    keep = nms(boxes_tensor, scores_tensor, 0.5)

    boxes = boxes_tensor[keep].numpy()
    scores = scores_tensor[keep].numpy()
    labels = labels[keep]
    # -----------------------------------------
    # ---------- ADD THIS NEXT ----------
    top_k = 5

    indices = scores.argsort()[::-1][:top_k]

    boxes = boxes[indices]
    labels = labels[indices]
    scores = scores[indices]
    # ----------------------------------

    img = image.copy()

    h, w = img.shape[:2]

    # adaptive scaling for small images
    thickness = max(1, int(min(h, w) / 200))
    font_scale = min(h, w) / 500

    for box, label, score in zip(boxes, labels, scores):

        if score < score_threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)


        scale_x = image.shape[1] / 800
        scale_y = image.shape[0] / 800

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        label_name = (
            COCO_CLASSES[label]
            if label < len(COCO_CLASSES)
            else f"class_{label}"
        )

        color = (0, 255, 0)

        # draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        text = f"{label_name} {score:.2f}"

        (text_w, text_h), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        # draw text background
        cv2.rectangle(
            img,
            (x1, y1 - text_h - 6),
            (x1 + text_w + 4, y1),
            color,
            -1
        )

        # draw text
        cv2.putText(
            img,
            text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    return img
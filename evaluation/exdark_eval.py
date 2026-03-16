import os
import numpy as np
from collections import defaultdict

# ------------------------------------------------------------
# Class mapping (ExDark dataset)
# ------------------------------------------------------------

CLASS_MAP = {
    "Bicycle":0,
    "Boat":1,
    "Bottle":2,
    "Bus":3,
    "Car":4,
    "Cat":5,
    "Chair":6,
    "Cup":7,
    "Dog":8,
    "Motorbike":9,
    "People":10,
    "Table":11
}

NUM_CLASSES = len(CLASS_MAP)

# ------------------------------------------------------------
# Load ExDark bbGt annotations
# ------------------------------------------------------------

def load_exdark_annotation(path):

    boxes = []
    labels = []
    areas = []

    with open(path) as f:
        lines = f.readlines()

    for line in lines:

        if line.startswith("%"):
            continue

        parts = line.strip().split()

        cls = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        boxes.append([x, y, x+w, y+h])
        labels.append(CLASS_MAP[cls])
        areas.append(w*h)

    return np.array(boxes), np.array(labels), np.array(areas)


# ------------------------------------------------------------
# IoU
# ------------------------------------------------------------

def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB-xA) * max(0, yB-yA)

    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    union = areaA + areaB - inter

    return inter / union if union > 0 else 0


# ------------------------------------------------------------
# Match predictions to ground truth
# ------------------------------------------------------------

def match_detections(pred_boxes, pred_labels, pred_scores,
                     gt_boxes, gt_labels,
                     iou_threshold):

    order = np.argsort(-pred_scores)

    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]
    pred_scores = pred_scores[order]

    matched = set()

    TP = np.zeros(len(pred_boxes))
    FP = np.zeros(len(pred_boxes))

    for i in range(len(pred_boxes)):

        box = pred_boxes[i]
        label = pred_labels[i]

        best_iou = 0
        best_gt = -1

        for j in range(len(gt_boxes)):

            if gt_labels[j] != label:
                continue

            iou = compute_iou(box, gt_boxes[j])

            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_threshold and best_gt not in matched:

            TP[i] = 1
            matched.add(best_gt)

        else:
            FP[i] = 1

    return TP, FP, len(gt_boxes)


# ------------------------------------------------------------
# Compute AP
# ------------------------------------------------------------

def compute_ap(tp, fp, total_gt):

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recalls = tp / (total_gt + 1e-6)
    precisions = tp / (tp + fp + 1e-6)

    ap = 0

    for t in np.linspace(0,1,101):

        p = precisions[recalls >= t]

        if len(p) == 0:
            p = 0
        else:
            p = max(p)

        ap += p / 101

    return ap


# ------------------------------------------------------------
# Object size split
# ------------------------------------------------------------

def object_size_category(area):

    if area < 32*32:
        return "small"
    elif area < 96*96:
        return "medium"
    else:
        return "large"


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------

def evaluate_dataset(dataset_images,
                     annotation_root,
                     predictions):

    """
    dataset_images : list of image filenames
    annotation_root : dataset/annotations
    predictions : dict
        image_name -> list of detections
        detection = {bbox:[x1,y1,x2,y2], score, class}
    """

    iou_thresholds = np.arange(0.5, 0.96, 0.05)

    ap_list = []
    ap50 = []
    ap75 = []

    aps = []
    apm = []
    apl = []

    recalls = []

    for iou_thr in iou_thresholds:

        all_tp = []
        all_fp = []
        total_gt = 0

        for img in dataset_images:

            ann_file = find_annotation_file(annotation_root, img)

            gt_boxes, gt_labels, areas = load_exdark_annotation(ann_file)

            preds = predictions.get(img, [])

            if len(preds) == 0:
                continue

            pred_boxes = np.array([p["bbox"] for p in preds])
            pred_labels = np.array([p["class"] for p in preds])
            pred_scores = np.array([p["score"] for p in preds])

            tp, fp, n_gt = match_detections(
                pred_boxes,
                pred_labels,
                pred_scores,
                gt_boxes,
                gt_labels,
                iou_thr
            )

            total_gt += n_gt

            all_tp.extend(tp)
            all_fp.extend(fp)

        ap = compute_ap(np.array(all_tp), np.array(all_fp), total_gt)

        ap_list.append(ap)

        if abs(iou_thr - 0.5) < 1e-6:
            ap50 = ap

        if abs(iou_thr - 0.75) < 1e-6:
            ap75 = ap

    AP = np.mean(ap_list)

    results = {
        "AP": AP,
        "AP50": ap50,
        "AP75": ap75
    }

    return results


# ------------------------------------------------------------
# Find annotation file
# ------------------------------------------------------------

def find_annotation_file(annotation_root, image_name):

    for root, dirs, files in os.walk(annotation_root):

        file = image_name + ".txt"

        if file in files:
            return os.path.join(root, file)

    raise Exception("Annotation not found for " + image_name)
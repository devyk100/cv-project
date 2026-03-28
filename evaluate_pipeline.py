import os
import cv2
import torch
import numpy as np

from detection.edge_detection import EdgeDetector
from enhancement.cloud_enhancement import CloudEnhancer
from preprocessing.single_preprocess import preprocess_single_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# PATHS
# -----------------------------

INPUT_DIR  = "enhancement/dataset/input"   # preprocessed Y_*.png files
IMAGE_ROOT = "dataset/1"                   # original ExDark images
ANN_ROOT   = "dataset/annotations"         # .png.txt annotation files

MAX_IMAGES = 300
SCORE_THRESHOLD = 0.3
IOU_THRESHOLD   = 0.5

# -----------------------------
# BUILD FILE INDEXES
# -----------------------------

def build_index(root_dir, exts):
    """Walk root_dir and map filename -> full path for given extensions."""
    file_map = {}
    for root, _, files in os.walk(root_dir):
        for f in files:
            if any(f.endswith(ext) for ext in exts):  # exts must be a list
                file_map[f] = os.path.join(root, f)
    return file_map

image_map      = build_index(IMAGE_ROOT, [".jpg", ".png"])
annotation_map = build_index(ANN_ROOT,   [".txt"])          # was a string before — bug fix

print("Indexed images:", len(image_map))
print("Indexed annotations:", len(annotation_map))

# -----------------------------
# LOAD MODELS
# -----------------------------

models = [
    "enhancement/model_clahe.pth",
    "enhancement/model_hist.pth",
    "enhancement/model_sharpen.pth",
    "enhancement/model_bilateral.pth",
    "enhancement/model_gamma.pth"
]

cloud = CloudEnhancer(models)
edge  = EdgeDetector()

# -----------------------------
# IoU
# -----------------------------

def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter

    return inter / union if union > 0 else 0.0

# -----------------------------
# PARSE ANNOTATION
# annotation format: class x y w h ...  (one box per line, skip % lines)
# -----------------------------

def parse_annotation(txt_path):

    boxes = []

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line or line.startswith("%"):
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        x  = int(parts[1])
        y  = int(parts[2])
        w  = int(parts[3])
        h  = int(parts[4])

        boxes.append([x, y, x + w, y + h])

    return boxes

# -----------------------------
# MATCH PREDICTIONS TO GT
# returns tp, fp, fn
# -----------------------------

def evaluate(pred_boxes, gt_boxes, iou_thresh=IOU_THRESHOLD):

    tp = 0
    fp = 0
    matched = set()

    for pb in pred_boxes:

        found = False

        for i, gb in enumerate(gt_boxes):

            if i in matched:
                continue

            if compute_iou(pb, gb) >= iou_thresh:
                tp += 1
                matched.add(i)
                found = True
                break

        if not found:
            fp += 1

    fn = len(gt_boxes) - len(matched)

    return tp, fp, fn

# -----------------------------
# SCALE BOXES from 800x800 back to original image size
# -----------------------------

def scale_boxes(boxes, orig_h, orig_w):

    scale_x = orig_w / 800
    scale_y = orig_h / 800

    scaled = []

    for x1, y1, x2, y2 in boxes:
        scaled.append([
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        ])

    return scaled

# -----------------------------
# MAIN LOOP
# -----------------------------

base_tp, base_fp, base_fn = 0, 0, 0
enh_tp,  enh_fp,  enh_fn  = 0, 0, 0

processed = 0

for img_name in sorted(os.listdir(INPUT_DIR)):

    if processed >= MAX_IMAGES:
        break

    # img_name is like Y_2015_00001.png
    # strip Y_ and extension to get base: 2015_00001.png
    base_name = img_name.replace("Y_", "", 1)
    txt_name  = base_name + ".txt"

    if base_name not in image_map or txt_name not in annotation_map:
        continue

    image_path = image_map[base_name]
    ann_path   = annotation_map[txt_name]

    # load original image for preprocessing (not the preprocessed Y)
    image = cv2.imread(image_path)
    if image is None:
        continue

    orig_h, orig_w = image.shape[:2]
    gt_boxes = parse_annotation(ann_path)

    if not gt_boxes:
        continue

    # ----------------------------------------
    # BASELINE: plain Faster R-CNN on original
    # ----------------------------------------

    baseline_input = cv2.resize(image, (800, 800))
    tensor = (
        torch.tensor(baseline_input)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float() / 255
    ).to(device)

    with torch.no_grad():
        baseline_det = edge.model(tensor)

    base_boxes  = baseline_det[0]["boxes"].detach().cpu().numpy()
    base_scores = baseline_det[0]["scores"].detach().cpu().numpy()

    baseline_pred = scale_boxes(
        [[int(x1), int(y1), int(x2), int(y2)]
         for (x1, y1, x2, y2), s in zip(base_boxes, base_scores)
         if s >= SCORE_THRESHOLD],
        orig_h, orig_w
    )

    tp, fp, fn = evaluate(baseline_pred, gt_boxes)
    base_tp += tp
    base_fp += fp
    base_fn += fn

    # ----------------------------------------
    # ENHANCED PIPELINE
    # use original image for preprocess so Cr/Cb are correct
    # ----------------------------------------

    rgb_resized, Y, Cr, Cb = preprocess_single_image(image_path)
    Y_tensor = torch.tensor(Y)

    enhanced_images, weights = cloud.enhance_tensor(Y_tensor)

    rgb_images   = edge.reconstruct_rgb(enhanced_images, Cr, Cb)
    features     = edge.extract_features(rgb_images)
    orig_feat    = edge.extract_features([rgb_resized])[0]
    all_weighted = edge.compute_weighted_features(orig_feat, features, weights)
    fused_feat   = edge.fuse_feature_maps(all_weighted)

    detector_input = cv2.resize(rgb_resized, (800, 800))
    detector_tensor = (
        torch.tensor(detector_input)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float() / 255
    ).to(device)

    detections = edge.detect(orig_feat, fused_feat, detector_tensor)

    enh_boxes  = detections[0]["boxes"].detach().cpu().numpy()
    enh_scores = detections[0]["scores"].detach().cpu().numpy()

    enhanced_pred = scale_boxes(
        [[int(x1), int(y1), int(x2), int(y2)]
         for (x1, y1, x2, y2), s in zip(enh_boxes, enh_scores)
         if s >= SCORE_THRESHOLD],
        orig_h, orig_w
    )

    tp, fp, fn = evaluate(enhanced_pred, gt_boxes)
    enh_tp += tp
    enh_fp += fp
    enh_fn += fn

    processed += 1

    if processed % 50 == 0:
        print(f"Processed: {processed}/{MAX_IMAGES}")

    if processed <= 5:
        print(f"\nImage: {base_name}")
        print(f"  GT:       {gt_boxes}")
        print(f"  Baseline: {baseline_pred}")
        print(f"  Enhanced: {enhanced_pred}")

# -----------------------------
# FINAL METRICS
# -----------------------------

def metrics(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

base_p, base_r, base_f1 = metrics(base_tp, base_fp, base_fn)
enh_p,  enh_r,  enh_f1  = metrics(enh_tp,  enh_fp,  enh_fn)

print("\n========== RESULTS ==========")
print(f"Processed images : {processed}")
print(f"IoU threshold    : {IOU_THRESHOLD}")
print(f"Score threshold  : {SCORE_THRESHOLD}")

print("\n--- Baseline (Faster R-CNN, no enhancement) ---")
print(f"  TP: {base_tp}  FP: {base_fp}  FN: {base_fn}")
print(f"  Precision : {base_p:.4f}")
print(f"  Recall    : {base_r:.4f}")
print(f"  F1        : {base_f1:.4f}")

print("\n--- Enhanced Pipeline ---")
print(f"  TP: {enh_tp}  FP: {enh_fp}  FN: {enh_fn}")
print(f"  Precision : {enh_p:.4f}")
print(f"  Recall    : {enh_r:.4f}")
print(f"  F1        : {enh_f1:.4f}")

print("\n--- Delta (Enhanced - Baseline) ---")
print(f"  Precision : {enh_p  - base_p:+.4f}")
print(f"  Recall    : {enh_r  - base_r:+.4f}")
print(f"  F1        : {enh_f1 - base_f1:+.4f}")
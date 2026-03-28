"""
evaluate_pipeline.py

Evaluates baseline Faster R-CNN vs the enhanced pipeline on the full ExDark dataset.

NOTE: The enhancement models were trained on all 7363 ExDark images (no split filtering),
so these results include images the enhancement subnetworks have seen during training.
For a clean evaluation, filter using imageclasslist.txt (split=3 for test-only images).
Results should be interpreted as an upper bound on enhancement quality.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from detection.edge_detection import EdgeDetector
from enhancement.cloud_enhancement import CloudEnhancer
from preprocessing.single_preprocess import preprocess_single_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# PATHS
# -----------------------------

INPUT_DIR  = "enhancement/dataset/input"
IMAGE_ROOT = "dataset/1"
ANN_ROOT   = "dataset/annotations"

SCORE_THRESHOLD = 0.05
IOU_THRESHOLD   = 0.5
NUM_WORKERS     = 8

# -----------------------------
# BUILD FILE INDEXES
# -----------------------------

def build_index(root_dir, exts):
    file_map = {}
    for root, _, files in os.walk(root_dir):
        for f in files:
            if any(f.endswith(ext) for ext in exts):
                file_map[f] = os.path.join(root, f)
    return file_map

image_map      = build_index(IMAGE_ROOT, [".jpg", ".png"])
annotation_map = build_index(ANN_ROOT,   [".txt"])

print(f"Indexed images: {len(image_map)}")
print(f"Indexed annotations: {len(annotation_map)}")

# -----------------------------
# BUILD EVALUATION LIST
# -----------------------------

eval_items = []

for img_name in sorted(os.listdir(INPUT_DIR)):

    if not any(img_name.endswith(ext) for ext in [".jpg", ".png"]):
        continue

    base_name = img_name.replace("Y_", "", 1)
    txt_name  = base_name + ".txt"

    if base_name not in image_map or txt_name not in annotation_map:
        continue

    eval_items.append({
        "img_name":   img_name,
        "base_name":  base_name,
        "image_path": image_map[base_name],
        "ann_path":   annotation_map[txt_name],
        "input_path": os.path.join(INPUT_DIR, img_name),
    })

print(f"Evaluation items: {len(eval_items)}")

# -----------------------------
# DATASET for parallel image loading
# -----------------------------

class EvalDataset(Dataset):

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        item = self.items[idx]

        image = cv2.imread(item["image_path"])
        if image is None:
            return None

        orig_h, orig_w = image.shape[:2]

        gt_boxes = parse_annotation(item["ann_path"])
        if not gt_boxes:
            return None

        return {
            "image":      image,
            "orig_h":     orig_h,
            "orig_w":     orig_w,
            "gt_boxes":   gt_boxes,
            "image_path": item["image_path"],
            "base_name":  item["base_name"],
        }

# -----------------------------
# PARSE ANNOTATION
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

        x = int(parts[1])
        y = int(parts[2])
        w = int(parts[3])
        h = int(parts[4])

        boxes.append([x, y, x + w, y + h])

    return boxes

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
# EVALUATE predictions vs GT
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
# SCALE BOXES from 800x800 to original size
# -----------------------------

def scale_boxes(boxes, orig_h, orig_w):

    scale_x = orig_w / 800
    scale_y = orig_h / 800

    return [
        [int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)]
        for x1, y1, x2, y2 in boxes
    ]

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
# BATCHED BACKBONE INFERENCE
# Process all 6 images (1 orig + 5 enhanced) in a single backbone call
# instead of 6 separate forward passes — much faster on L40S
# -----------------------------

def extract_features_batched(images_list):
    """
    images_list: list of H×W×3 numpy arrays
    Returns: list of feature dicts, one per image
    """
    tensors = []

    for img in images_list:
        img = cv2.resize(img, (800, 800))
        t   = torch.tensor(img).permute(2, 0, 1).float() / 255
        tensors.append(t)

    batch = torch.stack(tensors).to(device)   # (N, 3, 800, 800)

    with torch.no_grad():
        batch_feats = edge.backbone(batch)     # returns dict of FPN levels

    # split batch dim back into per-image dicts
    per_image = []
    for i in range(len(tensors)):
        feat = {k: batch_feats[k][i:i+1] for k in batch_feats}
        per_image.append(feat)

    return per_image

# -----------------------------
# MAIN EVALUATION LOOP
# -----------------------------

base_tp, base_fp, base_fn = 0, 0, 0
enh_tp,  enh_fp,  enh_fn  = 0, 0, 0
processed = 0

for item in tqdm(eval_items, desc="Evaluating"):

    image      = cv2.imread(item["image_path"])
    if image is None:
        continue

    orig_h, orig_w = image.shape[:2]
    gt_boxes       = parse_annotation(item["ann_path"])

    if not gt_boxes:
        continue

    # ----------------------------------------
    # BASELINE
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
    # ----------------------------------------

    rgb_resized, Y, Cr, Cb = preprocess_single_image(item["image_path"])
    Y_tensor = torch.tensor(Y)

    enhanced_images, weights = cloud.enhance_tensor(Y_tensor)
    rgb_enhanced = edge.reconstruct_rgb(enhanced_images, Cr, Cb)

    # batch all 6 images through backbone in one shot
    all_rgb     = [rgb_resized] + rgb_enhanced
    all_feats   = extract_features_batched(all_rgb)

    orig_feat        = all_feats[0]
    enhanced_feats   = all_feats[1:]

    all_weighted = edge.compute_weighted_features(orig_feat, enhanced_feats, weights)
    fused_feat   = edge.fuse_feature_maps(all_weighted)

    detector_input  = cv2.resize(rgb_resized, (800, 800))
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
print()
print("NOTE: Enhancement models were trained on all 7363 ExDark images.")
print("No train/test split was applied. Results are an upper bound.")
print("For clean metrics, filter with imageclasslist.txt (split=3).")

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
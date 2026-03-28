import os
import cv2
import torch
import xml.etree.ElementTree as ET

from detection.edge_detection import EdgeDetector, WeightedFasterRCNN
from enhancement.cloud_enhancement import CloudEnhancer
from preprocessing.single_preprocess import preprocess_single_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# PATHS
# -----------------------------

INPUT_DIR = "enhancement/dataset/input"
IMAGE_ROOT = "dataset/1"
ANN_ROOT = "dataset/annotations"

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

image_map = build_index(IMAGE_ROOT, [".jpg", ".png"])
annotation_map = build_index(ANN_ROOT, ".txt")

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
edge = EdgeDetector()
detector = WeightedFasterRCNN()

# -----------------------------
# IOU
# -----------------------------

def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB-xA) * max(0, yB-yA)

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    union = areaA + areaB - inter

    return inter / union if union > 0 else 0

# -----------------------------
# PARSE XML
# -----------------------------

def parse_annotation(txt_path):

    boxes = []

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("%"):
            continue

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        cls = parts[0]
        x = int(parts[1])
        y = int(parts[2])
        w = int(parts[3])
        h = int(parts[4])

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        boxes.append([x1, y1, x2, y2])

    return boxes
# -----------------------------
# EVALUATION FUNCTION
# -----------------------------

def evaluate(pred_boxes, gt_boxes, iou_thresh=0.5):

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
# MAIN LOOP
# -----------------------------

total_tp = 0
total_fp = 0
total_fn = 0

processed = 0

for img_name in os.listdir(INPUT_DIR):

    base_name = img_name.replace("Y_", "")

    txt_name = base_name + ".txt"

    if base_name not in image_map or txt_name not in annotation_map:
        continue

    image_path = image_map[base_name]
    xml_path = annotation_map[txt_name]

    image = cv2.imread(image_path)
    if image is None:
        continue

    # -----------------------------
    # BASELINE
    # -----------------------------

    baseline_input = cv2.resize(image, (800,800))

    tensor = torch.tensor(baseline_input).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        baseline_det = edge.model(tensor)

    base_boxes = baseline_det[0]["boxes"].detach().cpu().numpy()
    base_scores = baseline_det[0]["scores"].detach().cpu().numpy()

    baseline_pred = []

    for b, s in zip(base_boxes, base_scores):
        if s < 0.03:
            continue

        x1,y1,x2,y2 = b

        scale_x = image.shape[1] / 800
        scale_y = image.shape[0] / 800

        baseline_pred.append([
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        ])

    # -----------------------------
    # ENHANCED PIPELINE
    # -----------------------------

    input_path = os.path.join(INPUT_DIR, img_name)

    rgb_resized, Y, Cr, Cb = preprocess_single_image(input_path)

    Y_tensor = torch.tensor(Y)

    enhanced_images, weights = cloud.enhance_tensor(Y_tensor)

    rgb_images = edge.reconstruct_rgb(enhanced_images, Cr, Cb)

    features = edge.extract_features(rgb_images)

    orig_features = edge.extract_features([rgb_resized])[0]

    weighted_feats = edge.apply_weights(features, weights)

    fused = edge.combine_features(orig_features, weighted_feats)

    detector_input = cv2.resize(rgb_resized, (800,800))

    tensor = torch.tensor(detector_input).permute(2,0,1).unsqueeze(0).float().to(device)

    detections = detector.detect_with_features(tensor, fused)

    enh_boxes = detections[0]["boxes"].detach().cpu().numpy()
    enh_scores = detections[0]["scores"].detach().cpu().numpy()

    enhanced_pred = []

    for b, s in zip(enh_boxes, enh_scores):
        if s < 0.3:
            continue

        x1,y1,x2,y2 = b

        scale_x = rgb_resized.shape[1] / 800
        scale_y = rgb_resized.shape[0] / 800

        enhanced_pred.append([
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        ])

    # -----------------------------
    # GROUND TRUTH
    # -----------------------------

    gt_boxes = parse_annotation(xml_path)

    # -----------------------------
    # EVALUATE (ENHANCED ONLY)
    # -----------------------------

    tp, fp, fn = evaluate(enhanced_pred, gt_boxes)

    total_tp += tp
    total_fp += fp
    total_fn += fn

    processed += 1

    if processed % 100 == 0:
        print("Processed:", processed)
        if(processed == 300):
            break


    if processed < 5:
        print("Image:", base_name)
        print("GT:", gt_boxes)
        print("Pred:", enhanced_pred)

# -----------------------------
# FINAL METRICS
# -----------------------------

precision = total_tp / (total_tp + total_fp + 1e-6)
recall = total_tp / (total_tp + total_fn + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print("\n===== RESULTS =====")
print("Processed images:", processed)

print("TP:", total_tp)
print("FP:", total_fp)
print("FN:", total_fn)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
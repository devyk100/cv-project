import os
import cv2
import torch

from enhancement.cloud_enhancement import CloudEnhancer
from detection.edge_detection import EdgeDetector, WeightedFasterRCNN
from preprocessing.single_preprocess import preprocess_single_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "enhancement/dataset/input"
ORIGINAL_DIR = "dataset/1"   # ORIGINAL images
ANN_DIR = "dataset/annotations"

models = [
    "enhancement/model_clahe.pth",
    "enhancement/model_hist.pth",
    "enhancement/model_sharpen.pth",
    "enhancement/model_bilateral.pth",
    "enhancement/model_gamma.pth"
]

print("Reached here")

cloud = CloudEnhancer(models)
edge = EdgeDetector()
detector = WeightedFasterRCNN()

print("Reached here")

# -----------------------------
# IoU
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
# Parse ExDark
# -----------------------------
def parse_exdark(img_name):

    base = img_name.replace("Y_", "")       # Y_2015_00001.png → 2015_00001.png
    txt_name = base + ".txt"                # → 2015_00001.png.txt

    for root, dirs, files in os.walk(ANN_DIR):

        if txt_name in files:

            txt_path = os.path.join(root, txt_name)
            print("✅ Found:", txt_path)

            boxes = []

            with open(txt_path, "r") as f:
                for line in f:

                    if line.startswith("%"):
                        continue

                    parts = line.strip().split()

                    if len(parts) < 5:
                        continue

                    x, y, w, h = map(float, parts[1:5])

                    xmin = x
                    ymin = y
                    xmax = x + w
                    ymax = y + h

                    boxes.append([xmin, ymin, xmax, ymax])

            return boxes

    print("❌ Missing:", txt_name)
    return []


# -----------------------------
# MAIN
# -----------------------------

all_preds = []
all_gt = {}

print(os.listdir(IMAGE_DIR))

for img_name in os.listdir(IMAGE_DIR):
    print("For image", img_name)
    base = img_name.replace("Y_", "")
    print("For base image", base)
    orig_path = os.path.join(ORIGINAL_DIR, base)

    if not os.path.exists(orig_path):
        continue

    original = cv2.imread(orig_path)
    H, W = original.shape[:2]

    gt = parse_exdark(img_name)
    if len(gt) == 0:
        continue

    all_gt[img_name] = gt

    # -----------------------------
    # ENHANCEMENT (224 space)
    # -----------------------------
    rgb_resized, Y, Cr, Cb = preprocess_single_image(orig_path)

    Y_tensor = torch.tensor(Y)

    enhanced, weights = cloud.enhance_tensor(Y_tensor)

    rgb_images = edge.reconstruct_rgb(enhanced, Cr, Cb)

    features = edge.extract_features(rgb_images)
    orig_features = edge.extract_features([rgb_resized])[0]

    weighted = edge.apply_weights(features, weights)
    fused = edge.combine_features(orig_features, weighted)

    # -----------------------------
    # DETECTOR (ORIGINAL SCALE)
    # -----------------------------
    detector_input = cv2.resize(original, (800,800))

    tensor = torch.tensor(detector_input).permute(2,0,1).unsqueeze(0).float().to(device)

    detections = detector.detect_with_features(tensor, fused)

    boxes = detections[0]["boxes"].detach().cpu().numpy()
    scores = detections[0]["scores"].detach().cpu().numpy()

    # scale BACK to original
    sx = W / 800
    sy = H / 800

    for b, s in zip(boxes, scores):

        if s < 0.3:
            continue

        x1,y1,x2,y2 = b

        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy

        all_preds.append({
            "image_id": img_name,
            "score": float(s),
            "box": [x1,y1,x2,y2]
        })


# -----------------------------
# SORT
# -----------------------------
all_preds.sort(key=lambda x: x["score"], reverse=True)

tp, fp = [], []
used = {}

total_gt = sum(len(v) for v in all_gt.values())

for pred in all_preds:

    img = pred["image_id"]
    box = pred["box"]

    best_iou = 0
    best_idx = -1

    for i, gt in enumerate(all_gt[img]):

        iou = compute_iou(box, gt)

        if iou > best_iou:
            best_iou = iou
            best_idx = i

    if best_iou >= 0.5 and (img,best_idx) not in used:
        tp.append(1)
        fp.append(0)
        used[(img,best_idx)] = True
    else:
        tp.append(0)
        fp.append(1)


# -----------------------------
# AP
# -----------------------------
tp_c, fp_c = [], []
t,f = 0,0

for i,j in zip(tp,fp):
    t+=i; f+=j
    tp_c.append(t)
    fp_c.append(f)

prec, rec = [], []

for t,f in zip(tp_c, fp_c):
    prec.append(t/(t+f+1e-6))
    rec.append(t/total_gt)

ap = 0
for i in range(1,len(prec)):
    ap += (rec[i]-rec[i-1])*prec[i]

print("\n===== RESULT =====")
print("GT:", total_gt)
print("Pred:", len(all_preds))
print("AP50:", ap)
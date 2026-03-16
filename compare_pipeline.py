import os
import cv2
import torch

from detection.edge_detection import EdgeDetector, WeightedFasterRCNN
from enhancement.cloud_enhancement import CloudEnhancer
from preprocessing.single_preprocess import preprocess_single_image
from detection.visualize import draw_detections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Models
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
# Paths
# -----------------------------

image_dir = "enhancement/dataset/input"
output_dir = "comparison_results"

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Loop through dataset
# -----------------------------

for img_name in os.listdir(image_dir):

    image_path = os.path.join(image_dir, img_name)

    # -----------------------------
    # Convert Y_ filename → original filename
    # -----------------------------

    base_name = img_name.replace("Y_", "")

    # -----------------------------
    # Load original RGB image
    # -----------------------------

    original_path = os.path.join("dataset/1", base_name)

    if not os.path.exists(original_path):
        original_path = image_path

    image = cv2.imread(original_path)

    if image is None:
        continue

    # -----------------------------
    # BASELINE DETECTOR
    # -----------------------------

    baseline_input = cv2.resize(image, (800,800))

    tensor = torch.tensor(baseline_input).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        baseline_det = edge.model(tensor)

    baseline_vis = draw_detections(baseline_input, baseline_det, score_threshold=0.3)

    baseline_vis = cv2.resize(baseline_vis, (image.shape[1], image.shape[0]))

    # -----------------------------
    # ENHANCED PIPELINE
    # -----------------------------

    rgb_resized, Y, Cr, Cb = preprocess_single_image(image_path)

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

    enhanced_vis = draw_detections(detector_input, detections, score_threshold=0.3)

    enhanced_vis = cv2.resize(enhanced_vis, (image.shape[1], image.shape[0]))

    # -----------------------------
    # SIDE-BY-SIDE COMPARISON
    # -----------------------------

    comparison = cv2.hconcat([baseline_vis, enhanced_vis])

    output_path = os.path.join(output_dir, base_name)

    cv2.imwrite(output_path, comparison)

    print("Saved:", output_path)
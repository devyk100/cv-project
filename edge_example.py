import torch
import cv2
import os
from enhancement.cloud_enhancement import CloudEnhancer
from detection.edge_detection import EdgeDetector
from preprocessing.single_preprocess import preprocess_single_image
from detection.visualize import draw_detections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load enhancement models
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

# -----------------------------
# Input image
# -----------------------------

image_path = "dataset/1/Bicycle/2015_00020.jpg"

rgb_resized, Y, Cr, Cb = preprocess_single_image(image_path)

Y_tensor = torch.tensor(Y)

# -----------------------------
# Cloud stage
# Subnetworks run on Y, produce N enhanced Y'_i and weights v_i (eq. 6)
# -----------------------------

enhanced_images, weights = cloud.enhance_tensor(Y_tensor)

print("Dynamic weights:", weights)

# -----------------------------
# Edge stage
# -----------------------------

# Step 1: reconstruct RGB from each enhanced Y'_i + original Cr, Cb
rgb_images = edge.reconstruct_rgb(enhanced_images, Cr, Cb)

# Step 2: extract backbone features from each enhanced RGB and original RGB
enhanced_features = edge.extract_features(rgb_images)
orig_feat = edge.extract_features([rgb_resized])[0]

# Step 3: weight each enhanced feature map by v_i, orig stays at weight=1
all_weighted = edge.compute_weighted_features(orig_feat, enhanced_features, weights)

# Step 4: sum all weighted feature maps for ROI heads
fused_feat = edge.fuse_feature_maps(all_weighted)

# Step 5: detect
#   RPN  → original image features only  (paper Section III-C)
#   ROI  → fused weighted feature maps   (eq. 7)
detector_input = cv2.resize(rgb_resized, (800, 800))

detector_tensor = (
    torch.tensor(detector_input)
    .permute(2, 0, 1)
    .unsqueeze(0)
    .float() / 255
).to(device)

detections = edge.detect(orig_feat, fused_feat, detector_tensor)

print("Detections:", detections)

# -----------------------------
# Visualize and save
# -----------------------------

vis = draw_detections(detector_input, detections)

output_dir = "detection_result"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, os.path.basename(image_path))
cv2.imwrite(output_path, vis)

print(f"Saved: {output_path}")
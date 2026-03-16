import torch
import cv2
import os
from enhancement.cloud_enhancement import CloudEnhancer
from detection.edge_detection import EdgeDetector, WeightedFasterRCNN
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

image_path = "example-dataset/cat1/WIN_20260316_09_13_30_Pro.jpg"

rgb_resized, Y, Cr, Cb = preprocess_single_image(image_path)

Y_tensor = torch.tensor(Y)


# -----------------------------
# Cloud stage
# -----------------------------

enhanced_images, weights = cloud.enhance_tensor(Y_tensor)

print("Dynamic weights:", weights)


# -----------------------------
# Edge stage
# -----------------------------

# Step 1: reconstruct RGB
rgb_images = edge.reconstruct_rgb(enhanced_images, Cr, Cb)

# Step 2: extract features
features = edge.extract_features(rgb_images)

# original image features
orig_features = edge.extract_features([rgb_resized])[0]

# Step 3: apply weights
weighted_feats = edge.apply_weights(features, weights)

# Step 4: combine features
combined_feats = edge.combine_features(orig_features, weighted_feats)

# Step 5: run detector
# detections = edge.detect(rgb_resized) # first it was run on the original RGB device

detector = WeightedFasterRCNN()

detector_input = cv2.resize(rgb_resized, (800, 800))

detector_tensor = torch.tensor(detector_input).permute(2,0,1).unsqueeze(0).float().to(device)

detections = detector.detect_with_features(
    detector_tensor,
    combined_feats
)

print("Detections:", detections)

detector_input = cv2.resize(rgb_resized, (800, 800))
vis = draw_detections(detector_input, detections)

output_dir = "detection_result"
os.makedirs(output_dir, exist_ok=True)

image_name = os.path.basename(image_path)

output_path = os.path.join(output_dir, image_name)

cv2.imwrite(output_path, vis)

print(f"Saved: {output_path}")



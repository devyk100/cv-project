import os
import cv2
import torch

from detection.edge_detection import EdgeDetector, WeightedFasterRCNN
from enhancement.cloud_enhancement import CloudEnhancer
from preprocessing.single_preprocess import preprocess_single_image
from detection.visualize import draw_detections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# models
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

image_dir = "example-dataset/cat1"
output_dir = "comparison_results"

os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(image_dir):

    image_path = os.path.join(image_dir, img_name)

    image = cv2.imread(image_path)

    # -----------------
    # BASELINE
    # -----------------

    tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        baseline_det = edge.model(tensor)

    baseline_vis = draw_detections(image, baseline_det)

    # -----------------
    # ENHANCED
    # -----------------

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

    # detections = edge.detect_with_features(tensor, fused)
    detections = detector.detect_with_features(tensor, fused)

    enhanced_vis = draw_detections(detector_input, detections)

    enhanced_vis = cv2.resize(enhanced_vis, (image.shape[1], image.shape[0]))

    # -----------------
    # SIDE BY SIDE
    # -----------------

    comparison = cv2.hconcat([baseline_vis, enhanced_vis])

    output_path = os.path.join(output_dir, img_name)

    cv2.imwrite(output_path, comparison)

    print("Saved:", output_path)
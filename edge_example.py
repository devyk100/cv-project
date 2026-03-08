import torch
import cv2
import numpy as np

from enhancement.cloud_enhancement import CloudEnhancer
from detection.edge_detection import EdgeDetector
from preprocessing.single_preprocess import preprocess_single_image

# Load models

import os

BASE = "enhancement"

models = [
    os.path.join(BASE, "model_clahe.pth"),
    os.path.join(BASE, "model_hist.pth"),
    os.path.join(BASE, "model_sharpen.pth"),
    os.path.join(BASE, "model_bilateral.pth"),
    os.path.join(BASE, "model_gamma.pth"),
]

targets = [
    "dataset/target_clahe",
    "dataset/target_hist",
    "dataset/target_sharpen",
    "dataset/target_bilateral",
    "dataset/target_gamma"
]


cloud = CloudEnhancer(models)
edge = EdgeDetector()


# Input image

image_path = "./example-dataset/cat1/2015_00009.jpg"

rgb_resized, Y, Cr, Cb = preprocess_single_image(image_path)

Y_tensor = torch.tensor(Y)

# Cloud stage

enhanced_images, weights = cloud.enhance_tensor(
    Y_tensor,
    image_path,
    targets
)

print("Dynamic weights:", weights)


# Edge stage

rgb_images = edge.reconstruct_rgb(enhanced_images, Cr, Cb)

features = edge.extract_features(rgb_images)

orig_features = edge.extract_features([rgb_resized])[0]

weighted_feats = edge.apply_weights(features, weights)

combined_feats = edge.combine_features(orig_features, weighted_feats)

detections = edge.detect(rgb_resized)

print(detections)
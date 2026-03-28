import torch
import torchvision
import cv2
import numpy as np
from torchvision.models.detection.image_list import ImageList

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EdgeDetector:

    def __init__(self):

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )

        self.model = self.model.to(device)
        self.model.eval()

        self.backbone = self.model.backbone


    # -----------------------------
    # Step 1: reconstruct RGB
    # -----------------------------
    def reconstruct_rgb(self, enhanced_y_list, Cr, Cb):

        rgb_images = []

        for y_tensor in enhanced_y_list:

            y = y_tensor.squeeze().cpu().numpy()
            y = (y * 255).astype(np.uint8)

            merged = cv2.merge([y, Cr, Cb])
            rgb = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

            rgb_images.append(rgb)

        return rgb_images


    # -----------------------------
    # Step 2: extract features
    # -----------------------------
    def extract_features(self, images):

        features = []

        for img in images:

            img = cv2.resize(img, (800, 800)) # the boxes get an offset due to resizing, so we stay at 244 X 244
            # img = cv2.resize(img, (224,224))

            tensor = torch.tensor(img).permute(2,0,1).float()/255
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                feat = self.backbone(tensor)

            features.append(feat)

        return features


    # -----------------------------
# Step 3: compute weighted features (PAPER-CORRECT)
# -----------------------------
    def compute_weighted_features(self, orig_feat, features, weights):

        all_features = []

        # original image (weight = 1)
        all_features.append(orig_feat)

        # enhanced images
        for feat, w in zip(features, weights):

            weighted = {}

            for k in feat:
                weighted[k] = feat[k] * float(w)

            all_features.append(weighted)

        return all_features

    # -----------------------------
    # Step 5: run detector
    # -----------------------------
    def detect(self, image):

        tensor = torch.tensor(image).permute(2,0,1).float()/255
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(tensor)

        return output
    
class BackboneOverride(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        return self.features

class WeightedFasterRCNN:

    def __init__(self):

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )

        self.model = self.model.to(device)
        self.model.eval()

    def detect_with_features(self, image_tensor, fused_features):

        # Step 1: create ImageList
        image_sizes = [(image_tensor.shape[-2], image_tensor.shape[-1])]
        images = ImageList(image_tensor, image_sizes)

        # Step 2: RPN
        proposals, _ = self.model.rpn(images, fused_features)

        # Step 3: ROI heads
        detections, _ = self.model.roi_heads(
            fused_features,
            proposals,
            image_sizes,
        )

        # 🚨 Step 4: POSTPROCESS (YOU MISSED THIS)
        detections = self.model.transform.postprocess(
            detections,
            image_sizes,
            image_sizes
        )

        return detections
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

            img = cv2.resize(img, (800, 800))

            tensor = torch.tensor(img).permute(2, 0, 1).float() / 255
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                feat = self.backbone(tensor)

            features.append(feat)

        return features


    # -----------------------------
    # Step 3: compute weighted features (eq. 7)
    # orig_feat gets weight = 1
    # each enhanced feat gets weight = v_i
    # features stay SEPARATE (not summed)
    # -----------------------------
    def compute_weighted_features(self, orig_feat, features, weights):

        weighted_features = []

        # original image: weight = 1 (eq. 7, v_0 = 1)
        weighted_features.append(orig_feat)

        # enhanced images: weight = v_i
        for feat, w in zip(features, weights):

            weighted = {}

            for k in feat:
                weighted[k] = feat[k] * float(w)

            weighted_features.append(weighted)

        return weighted_features


    # -----------------------------
    # Step 4: sum weighted feature maps into one (eq. 7 final combination)
    # RPN will use orig_feat only (passed separately)
    # ROI heads use this combined map
    # -----------------------------
    def fuse_feature_maps(self, weighted_features):

        fused = {}

        for k in weighted_features[0]:
            fused[k] = sum(feat[k] for feat in weighted_features)

        return fused


    # -----------------------------
    # Step 5: detect using paper-correct two-stage approach:
    #   - RPN on original image features only
    #   - ROI heads on fused weighted features
    # -----------------------------
    def detect(self, orig_feat, fused_feat, image_tensor):

        image_sizes = [(image_tensor.shape[-2], image_tensor.shape[-1])]
        images = ImageList(image_tensor, image_sizes)

        with torch.no_grad():

            # RPN uses original image features only (paper Section III-C)
            proposals, _ = self.model.rpn(images, orig_feat)

            # ROI heads use fused weighted feature maps
            detections, _ = self.model.roi_heads(
                fused_feat,
                proposals,
                image_sizes,
            )

            # postprocess
            detections = self.model.transform.postprocess(
                detections,
                image_sizes,
                image_sizes
            )

        return detections
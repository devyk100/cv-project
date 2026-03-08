import torch
import torchvision
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EdgeDetector:

    def __init__(self):

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
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

            tensor = torch.tensor(img).permute(2,0,1).float()/255
            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                feat = self.backbone(tensor)

            features.append(feat)

        return features


    # -----------------------------
    # Step 3: apply weights
    # -----------------------------
    def apply_weights(self, features, weights):

        weighted = []

        for feat, w in zip(features, weights):

            wf = {}

            for k in feat:
                wf[k] = feat[k] * float(w)

            weighted.append(wf)

        return weighted


    # -----------------------------
    # Step 4: combine feature maps
    # -----------------------------
    def combine_features(self, original_feat, weighted_feats):

        fused = {}

        # original_feat is a dict from FPN:
        # {'0':..., '1':..., '2':..., '3':..., 'pool':...}

        for k in original_feat:

            total = original_feat[k].clone()

            for wf in weighted_feats:
                total += wf[k]

            fused[k] = total

        return fused


    # -----------------------------
    # Step 5: run detector
    # -----------------------------
    def detect(self, image):

        tensor = torch.tensor(image).permute(2,0,1).float()/255
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(tensor)

        return output

class WeightedFasterRCNN:

    def __init__(self):

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )

        self.model = self.model.to(device)
        self.model.eval()

    def detect_with_features(self, image_tensor, fused_features):

        """
        Run FasterRCNN using custom backbone features
        """

        # generate proposals
        proposals, _ = self.model.rpn(image_tensor, fused_features)

        detections, _ = self.model.roi_heads(
            fused_features,
            proposals,
            image_tensor.shape[-2:],
        )

        return detections

import torch
import cv2
import os
import numpy as np

# from models.enhancement_net import EnhancementNet # when using from edge_example.py
from enhancement.models.enhancement_net import EnhancementNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CloudEnhancer:

    def __init__(self, model_paths):

        self.models = []

        for path in model_paths:

            model = EnhancementNet()
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()

            self.models.append(model)

        print("Loaded", len(self.models), "enhancement subnetworks")

    def preprocess(self, image_path):

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (224,224))

        img = img.astype(np.float32) / 255.0

        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 0)

        return torch.tensor(img).to(device)

    def compute_loss(self, pred, target):

        return torch.mean((pred - target) ** 2).item()

    def enhance(self, image_path, target_dir):

        Y = self.preprocess(image_path)

        filename = os.path.basename(image_path)

        targets = []

        for t in target_dir:
            target_path = os.path.join(t, filename)

            target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
            target = cv2.resize(target, (224,224))

            target = target.astype(np.float32) / 255.0
            target = np.expand_dims(target, 0)
            target = np.expand_dims(target, 0)

            targets.append(torch.tensor(target).to(device))

        enhanced_images = []
        losses = []

        with torch.no_grad():

            for model, target in zip(self.models, targets):

                pred = model(Y)

                loss = self.compute_loss(pred, target)

                enhanced_images.append(pred)
                losses.append(loss)

        losses = np.array(losses)

        N = len(losses)

        weights = (1 - losses / np.sum(losses)) * (N / (N - 1))

        return enhanced_images, weights
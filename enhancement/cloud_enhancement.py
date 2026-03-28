import torch
import numpy as np
import os
import cv2

from enhancement.models.enhancement_net import EnhancementNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CloudEnhancer:

    def __init__(self, model_paths, target_dirs):

        self.models = []
        self.target_dirs = target_dirs
        self.criterion = torch.nn.MSELoss()

        for path in model_paths:

            model = EnhancementNet()
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)
            model.eval()

            self.models.append(model)

        print(f"Loaded {len(self.models)} enhancement subnetworks")

    # ----------------------------------
    # Core cloud enhancement stage
    # ----------------------------------
    def enhance_tensor(self, Y_tensor, image_name):

        """
        Y_tensor shape: (1,1,224,224)
        """

        Y_tensor = Y_tensor.to(device)

        enhanced_images = []
        losses = []

        with torch.no_grad():

            for i, model in enumerate(self.models):

                pred = model(Y_tensor)
                enhanced_images.append(pred)

                # -----------------------------
                # Load corresponding target
                # -----------------------------
                target_path = os.path.join(
                    self.target_dirs[i],
                    image_name
                )

                target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

                if target is None:
                    raise ValueError(f"Missing target: {target_path}")

                target = cv2.resize(target, (224, 224))
                target = target.astype(np.float32) / 255.0

                target = torch.tensor(target).unsqueeze(0).unsqueeze(0).to(device)

                # -----------------------------
                # Compute loss
                # -----------------------------
                loss = self.criterion(pred, target)
                losses.append(loss)

        losses = torch.stack(losses)

        N = len(losses)

        # -----------------------------
        # Paper formula
        # -----------------------------
        weights = (1 - losses / losses.sum()) * (N / (N - 1))

        return enhanced_images, weights.detach().cpu().numpy()
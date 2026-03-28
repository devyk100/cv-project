import torch
import numpy as np
import cv2

from enhancement.models.enhancement_net import EnhancementNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CloudEnhancer:

    def __init__(self, model_paths, target_dirs=None):

        self.models = []
        self.criterion = torch.nn.MSELoss()

        for path in model_paths:

            model = EnhancementNet()
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)
            model.eval()

            self.models.append(model)

        print(f"Loaded {len(self.models)} enhancement subnetworks")

    # ----------------------------------
    # Compute targets on-the-fly from Y
    # ----------------------------------
    def _compute_targets(self, y_np):

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

        return [
            clahe.apply(y_np),                                        # clahe
            cv2.equalizeHist(y_np),                                   # hist
            cv2.filter2D(y_np, -1, sharpen_kernel),                   # sharpen
            cv2.bilateralFilter(y_np, 9, 75, 75),                     # bilateral
            np.array(255 * (y_np / 255) ** 1.5, dtype=np.uint8),     # gamma
        ]

    # ----------------------------------
    # Core cloud enhancement stage
    # ----------------------------------
    def enhance_tensor(self, Y_tensor, image_name=None):

        """
        Y_tensor shape: (1,1,224,224)
        Targets are computed on-the-fly from Y so this works on any image,
        not just ones pre-saved in target folders.
        """

        Y_tensor = Y_tensor.to(device)

        # extract raw uint8 Y for filter computation
        y_np = (Y_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        targets_np = self._compute_targets(y_np)

        enhanced_images = []
        losses = []

        with torch.no_grad():

            for i, model in enumerate(self.models):

                pred = model(Y_tensor)
                enhanced_images.append(pred)

                # -----------------------------
                # Build target tensor
                # -----------------------------
                target = targets_np[i].astype(np.float32) / 255.0
                target = torch.tensor(target).unsqueeze(0).unsqueeze(0).to(device)

                # -----------------------------
                # Compute loss
                # -----------------------------
                loss = self.criterion(pred, target)
                losses.append(loss)

        losses = torch.stack(losses)

        N = len(losses)

        # -----------------------------
        # Paper formula (eq. 6)
        # -----------------------------
        weights = (1 - losses / losses.sum()) * (N / (N - 1))

        return enhanced_images, weights.detach().cpu().numpy()
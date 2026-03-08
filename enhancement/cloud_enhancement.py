import torch
import numpy as np
from enhancement.models.enhancement_net import EnhancementNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CloudEnhancer:

    def __init__(self, model_paths):

        self.models = []

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
    def enhance_tensor(self, Y_tensor):

        """
        Y_tensor shape: (1,1,224,224)
        """

        Y_tensor = Y_tensor.to(device)

        enhanced_images = []

        with torch.no_grad():
            for model in self.models:
                pred = model(Y_tensor)
                enhanced_images.append(pred)

        # compute dynamic weights using enhancement outputs
        scores = []

        for img in enhanced_images:
            arr = img.squeeze().cpu().numpy()
            scores.append(np.var(arr))  # contrast proxy

        scores = np.array(scores)

        if scores.sum() == 0:
            weights = np.ones(len(scores)) / len(scores)
        else:
            weights = scores / scores.sum()

        return enhanced_images, weights
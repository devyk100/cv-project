import cv2
import numpy as np


def preprocess_single_image(image_path):
    """
    Loads an image and prepares Y channel for enhancement.
    Returns:
        resized_rgb
        Y tensor (1x1x224x224)
        Cr
        Cb
    """

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    resized = cv2.resize(img, (224, 224))

    ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    Y = Y.astype(np.float32) / 255.0

    Y = np.expand_dims(Y, 0)
    Y = np.expand_dims(Y, 0)

    return resized, Y, Cr, Cb
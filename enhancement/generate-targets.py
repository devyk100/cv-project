import cv2
import os
import numpy as np

input_dir = "dataset/input"

targets = {
    "target_clahe": None,
    "target_hist": None,
    "target_sharpen": None,
    "target_bilateral": None,
    "target_gamma": None
}

for t in targets:
    os.makedirs(f"dataset/{t}", exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

for file in os.listdir(input_dir):

    path = os.path.join(input_dir, file)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    # CLAHE
    clahe_img = clahe.apply(img)

    # Histogram equalization
    hist_img = cv2.equalizeHist(img)

    # Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharpen_img = cv2.filter2D(img,-1,kernel)

    # Bilateral filter
    bilateral_img = cv2.bilateralFilter(img,9,75,75)

    # Gamma correction
    gamma = 1.5
    gamma_img = np.array(255*(img/255)**gamma,dtype='uint8')

    cv2.imwrite(f"dataset/target_clahe/{file}", clahe_img)
    cv2.imwrite(f"dataset/target_hist/{file}", hist_img)
    cv2.imwrite(f"dataset/target_sharpen/{file}", sharpen_img)
    cv2.imwrite(f"dataset/target_bilateral/{file}", bilateral_img)
    cv2.imwrite(f"dataset/target_gamma/{file}", gamma_img)

print("All targets generated.")
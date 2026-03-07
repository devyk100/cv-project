import cv2
import os

import cv2
import os

dataset = os.getenv("DATASET_PATH")
output_folder = os.getenv("OUTPUT_PATH")

for category in os.listdir(dataset):

    category_path = os.path.join(dataset, category)

    if not os.path.isdir(category_path):
        continue

    for file in os.listdir(category_path):

        path = os.path.join(category_path, file)

        image = cv2.imread(path)

        if image is None:
            continue

        resized = cv2.resize(image, (224,224))

        ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)

        Y, Cr, Cb = cv2.split(ycrcb)

        output_path = os.path.join(output_folder, "Y_" + file)

        cv2.imwrite(output_path, Y)

print("All images processed and saved.")
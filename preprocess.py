import cv2
import os
import numpy as np

# Updated to match your exact folder names
dataset_dir = "Photos"
output_dir = "Preprocessed"

def process_dataset():
    # Check if your Photos folder exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Could not find the '{dataset_dir}' folder. Make sure you are running this in the correct directory.")
        return

    print(f"Scanning '{dataset_dir}' for images...")

    # os.walk goes through all folders and subfolders automatically
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            # Only process image files
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            input_path = os.path.join(root, file)
            
            # Recreate your exact folder structure inside the 'Preprocessed' folder
            relative_path = os.path.relpath(root, dataset_dir)
            if relative_path == '.':
                out_category_path = output_dir
            else:
                out_category_path = os.path.join(output_dir, relative_path)
            
            os.makedirs(out_category_path, exist_ok=True)

            # Read the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"  - Skipping {file} (Could not read as an image)")
                continue

            # 1. Convert BGR to YCrCb to separate illumination from color [cite: 824]
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            # 2. Split into the Luminance (Y) and Chrominance (Cr, Cb) components [cite: 824]
            Y, Cr, Cb = cv2.split(ycrcb)

            # 3. Resize ONLY the Y (luminance) channel to 224x224 for fast transmission [cite: 825]
            Y_resized = cv2.resize(Y, (224, 224), interpolation=cv2.INTER_AREA)

            # --- Saving the Outputs ---
            
            # Save the Y channel (The small file that would go to the Cloud)
            output_path_Y = os.path.join(out_category_path, "Y_" + file)
            cv2.imwrite(output_path_Y, Y_resized)

            # Save the Cr and Cb channels (The color data kept on the Edge device)
            file_name_without_ext = os.path.splitext(file)[0]
            output_path_color = os.path.join(out_category_path, "ColorData_" + file_name_without_ext + ".npz")
            np.savez_compressed(output_path_color, Cr=Cr, Cb=Cb, original_shape=image.shape)
            
            print(f"  - Successfully processed: {file}")

    print(f"\nFinished! Check the '{output_dir}' folder.")

if __name__ == "__main__":
    process_dataset()
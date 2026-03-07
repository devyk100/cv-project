import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
class LowLightDataset(Dataset):

    def __init__(self, folder_path):
        self.folder = folder_path
        self.image_files = os.listdir(folder_path)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        img_path = os.path.join(self.folder, self.image_files[idx])

        image = Image.open(img_path).convert("L")   # grayscale
        image = self.transform(image)

        return image
# YOUR DATASET PATH
folder = "/home/satvik/Desktop/lowlight_env/preprocessed_images/"

dataset = LowLightDataset(folder)

# DataLoader for batching
loader = DataLoader(dataset, batch_size=8, shuffle=True)

print("Total Images:", len(dataset))

# test one batch
for batch in loader:
    print("Batch shape:", batch.shape)
    break

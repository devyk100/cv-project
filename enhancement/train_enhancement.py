import os
import cv2
import torch
import numpy as np
import argparse

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from models.enhancement_net import EnhancementNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class EnhancementDataset(Dataset):

    def __init__(self,input_dir,target_dir):

        self.input_dir = input_dir
        self.target_dir = target_dir
        self.files = os.listdir(input_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):

        file = self.files[idx]

        x_path = os.path.join(self.input_dir,file)
        y_path = os.path.join(self.target_dir,file)

        x = cv2.imread(x_path,cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(y_path,cv2.IMREAD_GRAYSCALE)

        x = x.astype(np.float32)/255.0
        y = y.astype(np.float32)/255.0

        x = np.expand_dims(x,0)
        y = np.expand_dims(y,0)

        return torch.tensor(x), torch.tensor(y)


parser = argparse.ArgumentParser()
parser.add_argument("--target_dir", required=True)
parser.add_argument("--model_name", required=True)

args = parser.parse_args()


dataset = EnhancementDataset("dataset/input", args.target_dir)

loader = DataLoader(dataset,batch_size=30,shuffle=True)


model = EnhancementNet().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(),lr=1e-3)


for epoch in range(25):

    total_loss = 0

    for x,y in loader:

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = criterion(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch:",epoch,"Loss:",total_loss/len(loader))


torch.save(model.state_dict(), args.model_name)

print("Model saved:",args.model_name)
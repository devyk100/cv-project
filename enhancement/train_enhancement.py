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

    def __init__(self, input_dir, target_dir):

        self.input_dir = input_dir
        self.target_dir = target_dir
        self.files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]

        x_path = os.path.join(self.input_dir, file)
        y_path = os.path.join(self.target_dir, file)

        x = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
        y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)

        if x is None:
            raise FileNotFoundError(f"Could not load input: {x_path}")
        if y is None:
            raise FileNotFoundError(f"Could not load target: {y_path}")

        # paper uses 224x224 fixed size (Section III-A)
        x = cv2.resize(x, (224, 224))
        y = cv2.resize(y, (224, 224))

        x = x.astype(np.float32) / 255.0
        y = y.astype(np.float32) / 255.0

        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)

        return torch.tensor(x), torch.tensor(y)


parser = argparse.ArgumentParser()
parser.add_argument("--target_dir", required=True)
parser.add_argument("--model_name", required=True)
parser.add_argument("--epochs", type=int, default=12)        # paper: 12 epochs
parser.add_argument("--batch_size", type=int, default=30)
parser.add_argument("--lr", type=float, default=0.01)        # paper: lr=0.01
parser.add_argument("--warmup_iters", type=int, default=500) # paper: 500 iter warmup

args = parser.parse_args()


dataset = EnhancementDataset("dataset/input", args.target_dir)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

model = EnhancementNet().to(device)

criterion = nn.MSELoss()

# paper: SGD, momentum=0.9, weight_decay=0.0001 (Section IV-E)
optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=0.9,
    weight_decay=0.0001
)

warmup_start_lr = args.lr / 3.0
total_iters = 0

def get_warmup_lr(iteration):
    if iteration >= args.warmup_iters:
        return args.lr
    return warmup_start_lr + (args.lr - warmup_start_lr) * (iteration / args.warmup_iters)


for epoch in range(args.epochs):

    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        # linear warmup for first 500 iterations (paper Section IV-E)
        if total_iters < args.warmup_iters:
            lr = get_warmup_lr(total_iters)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iters += 1

    print(f"Epoch: {epoch}  Loss: {total_loss / len(loader):.6f}  LR: {optimizer.param_groups[0]['lr']:.6f}")


torch.save(model.state_dict(), args.model_name)
print("Model saved:", args.model_name)
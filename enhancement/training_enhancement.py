import os
import cv2
import torch
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.enhancement_net import EnhancementNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


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

        x = cv2.resize(x, (224, 224))
        y = cv2.resize(y, (224, 224))

        # augmentation: random horizontal flip
        if np.random.rand() > 0.5:
            x = cv2.flip(x, 1)
            y = cv2.flip(y, 1)

        x = x.astype(np.float32) / 255.0
        y = y.astype(np.float32) / 255.0

        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)

        return torch.tensor(x), torch.tensor(y)


parser = argparse.ArgumentParser()
parser.add_argument("--target_dir",   required=True)
parser.add_argument("--model_name",   required=True)
parser.add_argument("--epochs",       type=int,   default=50)
parser.add_argument("--batch_size",   type=int,   default=256)   # L40S has 48GB, use it
parser.add_argument("--lr",           type=float, default=0.01)
parser.add_argument("--warmup_iters", type=int,   default=500)
parser.add_argument("--num_workers",  type=int,   default=8)     # parallel data loading
parser.add_argument("--val_split",    type=float, default=0.1)   # 10% validation

args = parser.parse_args()

# -----------------------------
# Dataset + split
# -----------------------------

full_dataset = EnhancementDataset("dataset/input", args.target_dir)

val_size   = int(len(full_dataset) * args.val_split)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True       # faster CPU->GPU transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

print(f"Train: {train_size} | Val: {val_size}")

# -----------------------------
# Model + optimizer
# -----------------------------

model     = EnhancementNet().to(device)
criterion = nn.MSELoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=0.9,
    weight_decay=0.0001
)

# cosine annealing after warmup — decays lr smoothly to near-zero by end
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

# -----------------------------
# Warmup
# -----------------------------

warmup_start_lr = args.lr / 3.0
total_iters     = 0

def apply_warmup(iteration):
    if iteration >= args.warmup_iters:
        return
    lr = warmup_start_lr + (args.lr - warmup_start_lr) * (iteration / args.warmup_iters)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

# -----------------------------
# Checkpointing
# -----------------------------

checkpoint_dir  = os.path.dirname(args.model_name) or "."
checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_{os.path.basename(args.model_name)}")

best_val_loss = float('inf')

# -----------------------------
# Training loop
# -----------------------------

for epoch in range(args.epochs):

    # --- train ---
    model.train()
    train_loss = 0.0

    for x, y in train_loader:

        apply_warmup(total_iters)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
        loss.backward()
        optimizer.step()

        train_loss  += loss.item()
        total_iters += 1

    # step scheduler after warmup is done
    if total_iters >= args.warmup_iters:
        scheduler.step()

    # --- validate ---
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred     = model(x)
            val_loss += criterion(pred, y).item()

    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch:3d}  Train: {avg_train:.6f}  Val: {avg_val:.6f}  LR: {current_lr:.6f}")

    # save best model
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), args.model_name)
        print(f"  -> Saved best model (val={best_val_loss:.6f})")

    # save checkpoint every 10 epochs so you can resume if needed
    if (epoch + 1) % 10 == 0:
        torch.save({
            "epoch":          epoch,
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "best_val_loss":  best_val_loss,
        }, checkpoint_path)

print(f"\nDone. Best val loss: {best_val_loss:.6f}")
print(f"Model saved: {args.model_name}")
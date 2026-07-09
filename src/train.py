import sys
sys.stdout.reconfigure(line_buffering=True)

import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["TORCH_HOME"] = os.path.expanduser("~/.cache/torch")

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

print(">>> train.py started")

# -------------------------
# Dataset
# -------------------------
class SegmentationDataset(Dataset):
    def __init__(self, csv_file, augment=False):
        self.df = pd.read_csv(csv_file)
        self.augment = augment

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            # Shadow / illumination robustness
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.3),
            # A.CoarseDropout(
            #     # max_holes=4,
            #     # holes=4,
            #     max_holes=4,
            #     max_height=32,
            #     max_width=32,
            #     fill_value=0,
            #     p=0.3
            # ),

            # Added to improve color robustness
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                p=0.7
                ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
                ),
            A.GaussNoise(
                # var_limit=(10.0, 50.0),
                # p=0.3
                p=0.3
            ),
            A.CoarseDropout(
                num_holes_range=(4, 4),
                hole_height_range=(32, 32),
                hole_width_range=(32, 32),
                fill=0,
                fill_mask=0,
                p=0.3
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.df.iloc[idx]["image_path"]).convert("RGB"))
        mask = np.array(Image.open(self.df.iloc[idx]["mask_path"]))

        if self.augment:
            out = self.transform(image=img, mask=mask)
        else:
            out = self.val_transform(image=img, mask=mask)

        return out["image"], out["mask"].long()


# -------------------------
# Model
# -------------------------
def get_model(num_classes):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    )
    return model

# -------------------------
# Quantitative metrics
# -------------------------
def compute_iou(preds, targets, num_classes):
    """
    preds: (N, H, W) predicted class indices
    targets: (N, H, W) ground truth class indices
    """
    ious = []

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union == 0:
            ious.append(float("nan"))  # class not present
        else:
            ious.append(intersection / union)

    return ious

# -------------------------
# Training
# -------------------------
def train():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 5
    BATCH_SIZE = 4
    EPOCHS = 40
    LR = 3e-4

    train_ds = SegmentationDataset(
        "./dataset/potsdam_2/csv/train.csv", augment=True)
    val_ds = SegmentationDataset(
        "./dataset/potsdam_2/csv/val.csv", augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    model = get_model(NUM_CLASSES).to(DEVICE)

    # Check if encoder is pretrained
    enc = model.encoder
    w = enc.conv1.weight.detach().cpu()
    print("[CHECK] encoder pretrained?", flush=True)
    print("conv1 mean:", w.mean().item(), flush=True)
    print("conv1 std :", w.std().item(), flush=True)

    #added weighted loss
    class_weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0], device=DEVICE)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = smp.losses.DiceLoss(mode="multiclass")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS
    )

    best_val_loss = 1e9

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)

            # print("Unique mask values:", torch.unique(masks))
            # print("Pred shape:", preds.shape)

            loss = ce_loss(preds, masks) + dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                
                # preds = model(imgs)
                # loss = ce_loss(preds, masks) + dice_loss(preds, masks)
                # val_loss += loss.item()

                logits = model(imgs)
                loss = ce_loss(logits, masks) + dice_loss(logits, masks)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())

        val_loss /= len(val_loader)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        ious = compute_iou(all_preds, all_targets, NUM_CLASSES)
        mean_iou = np.nanmean(ious)

        # print(f"Epoch {epoch+1}/{EPOCHS} | "
        #       f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"mIoU: {mean_iou:.4f}", flush=True
            )
        # Printing LR too
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  LR: {current_lr:.6f}", flush=True)
        
        for cls_idx, iou in enumerate(ious):
            print(f"  Class {cls_idx} IoU: {iou:.4f}", flush=True)

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), "best_unet.pth")
        best_miou = 0.0
        
        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), "best_unet4.pth")
            print("Saved best model", flush=True)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    train()
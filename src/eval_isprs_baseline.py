import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
IGNORE_VAL = 255

MODEL_PATH = "./models/best_unet4.pth"
VAL_CSV = "./dataset/potsdam_2/csv/val.csv"


# -------------------------
# MODEL
# -------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# -------------------------
# TRANSFORM
# -------------------------
transform = A.Compose([
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
])


# -------------------------
# LABEL MAPPING (CRITICAL)
# -------------------------
def map_to_2class(mask):
    """
    ISPRS → 2-class:
    building (2) → 1
    vegetation (3) + tree (4) → 2
    others → ignore
    """
    mask = mask.astype(np.int64)

    new_mask = np.full(mask.shape, IGNORE_VAL, dtype=np.int64)

    # strictly map only known classes
    new_mask[mask == 2] = 1  # building
    new_mask[(mask == 3) | (mask == 4)] = 2  # vegetation

    return new_mask


# -------------------------
# CONFUSION MATRIX
# -------------------------
empty_count = 0

def update_cm(cm, gt, pred):
    mask = (gt != IGNORE_VAL)
    gt = gt[mask]
    pred = pred[mask]
    valid = (gt >= 1) & (gt <= 2) & (pred >= 1) & (pred <= 2)
    gt = gt[valid]
    pred = pred[valid]
   

    # FIX: skip empty samples
    if gt.size == 0:
        return cm

    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)
    # print("GT unique after mapping:", np.unique(gt))

    label = 3 * gt + pred
    # if label.size > 0:
    #     print("max label:", label.max())
    count = np.bincount(label, minlength=9)

    cm += count.reshape(3, 3)
    if gt.size == 0:
        empty_count += 1
    return cm


# -------------------------
# METRICS
# -------------------------
def compute_metrics(cm):
    # ignore row/col 0 (ignored class)
    cm = cm[1:, 1:]  # keep only building + vegetation

    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return iou, precision, recall


# -------------------------
# MAIN EVAL
# -------------------------
df = pd.read_csv(VAL_CSV)
print(df.head())

cm = np.zeros((3, 3), dtype=np.int64)

print("Running ISPRS validation baseline...")
print("Empty samples skipped:", empty_count)
for i in range(len(df)):

    img_path = df.iloc[i]["image_path"]
    mask_path = df.iloc[i]["mask_path"]

    # load
    img = np.array(Image.open(img_path).convert("RGB"))
    gt = np.array(Image.open(mask_path))

    if gt.ndim == 3:
        gt = gt[..., 0]

    # map GT
    gt = map_to_2class(gt)

    # predict
    x = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1)[0].cpu().numpy()

    # map prediction
    pred = map_to_2class(pred)
    if i < 5:
        Image.fromarray(pred.astype(np.uint8)*100).save(f"pred_{i}.png")
        Image.fromarray(gt.astype(np.uint8)*100).save(f"gt_{i}.png")

    # update confusion matrix
    cm = update_cm(cm, gt, pred)

    if i % 50 == 0:
        print(f"{i}/{len(df)}")
    
    # with torch.no_grad():
    #     logits = model(x)
    #     pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
    
    # del x, logits
    # import gc
    # gc.collect()

# -------------------------
# RESULTS
# -------------------------
iou, precision, recall = compute_metrics(cm)

CLASS_NAMES = ["building", "vegetation"]

print("\n===== ISPRS BASELINE RESULTS =====")

print("\nConfusion Matrix (2-class):")
print(cm[1:,1:])

print("\nPer-class Metrics:")
for i, name in enumerate(CLASS_NAMES):
    print(
        f"{name:<12} "
        f"IoU: {iou[i]:.3f}   "
        f"Precision: {precision[i]:.3f}   "
        f"Recall: {recall[i]:.3f}"
    )

print("\nMean IoU:", np.mean(iou))


# -------------------------
# DISTRIBUTION
# -------------------------
gt_pixels = cm.sum(axis=1)[1:]
pred_pixels = cm.sum(axis=0)[1:]

print("\nGT distribution:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name}: {gt_pixels[i]}")

print("\nPrediction distribution:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name}: {pred_pixels[i]}")
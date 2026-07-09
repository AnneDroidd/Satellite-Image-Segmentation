# from operator import gt
import os
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
MODEL_PATH = "./models/best_unet4.pth"
VAL_CSV = "./dataset/potsdam_2/csv/val.csv"
OUT_DIR = "./inference_outputs_4"
PALETTE_JSON = "./potsdam_palette.json"
TREE_CLASS_INDEX = 4  # or whatever your tree index is

os.makedirs(OUT_DIR, exist_ok=True)
VIS_DIR = os.path.join(OUT_DIR, "side_by_side")
os.makedirs(VIS_DIR, exist_ok=True)

# -------------------------
# Load palette (index -> RGB)
# -------------------------
with open(PALETTE_JSON, "r") as f:
    raw = json.load(f)

# convert "R,G,B" -> index  to  index -> (R,G,B)
IDX_TO_RGB = {v: tuple(map(int, k.split(","))) for k, v in raw.items()}

# -------------------------
# Model
# -------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,   # IMPORTANT: weights come from .pth
    in_channels=3,
    classes=NUM_CLASSES
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------
# Preprocessing (same as training)
# -------------------------
transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# -------------------------
# Helper: class mask -> RGB image
# -------------------------
def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in IDX_TO_RGB.items():
        rgb[mask == cls] = color
    return rgb

# -------------------------
# Metrics
# -------------------------
conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

def update_confusion_matrix(cm, gt, pred, num_classes):
    mask = (gt >= 0) & (gt < num_classes)
    label = num_classes * gt[mask] + pred[mask]
    count = np.bincount(label, minlength=num_classes**2)
    cm += count.reshape(num_classes, num_classes)
    return cm

# -------------------------
# Inference on validation set
# -------------------------
df = pd.read_csv(VAL_CSV)

# run on a few samples (adjust if needed)
# NUM_SAMPLES = min(10, len(df))
NUM_SAMPLES = len(df)

for i in range(20):  # NUM_SAMPLES
    img_path = df.iloc[i]["image_path"]
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # load image
    img = np.array(Image.open(img_path).convert("RGB"))

    # preprocess
    x = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

    # predict
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

    # load ground truth mask newwwww
    mask_path = df.iloc[i]["mask_path"]  # make sure column exists
    gt = np.array(Image.open(mask_path))
    conf_matrix = update_confusion_matrix(conf_matrix, gt, pred, NUM_CLASSES)

    # colorize prediction + GT
    pred_rgb = mask_to_rgb(pred)
    gt_rgb = mask_to_rgb(gt)

    # save outputs
    Image.fromarray(img).save(os.path.join(OUT_DIR, f"{img_name}_input.png"))
    Image.fromarray(pred_rgb).save(os.path.join(OUT_DIR, f"{img_name}_segmented.png"))
    # -------------------------
    # Side-by-side visualization
    # -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(VIS_DIR, f"{img_name}_comparison.png"))
    plt.close()

print(f"[DONE] Saved {NUM_SAMPLES} segmented images to {OUT_DIR}")
tp = np.diag(conf_matrix)
fp = conf_matrix.sum(axis=0) - tp
fn = conf_matrix.sum(axis=1) - tp

iou = tp / (tp + fp + fn + 1e-6)
dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
pixel_acc = tp.sum() / conf_matrix.sum()

print("\nPer-class IoU:")
for i, val in enumerate(iou):
    print(f"Class {i}: {val:.4f}")

print(f"\nMean IoU: {iou.mean():.4f}")
print(f"Mean Dice: {dice.mean():.4f}")
print(f"Pixel Accuracy: {pixel_acc:.4f}")
print(f"\nTree IoU: {iou[TREE_CLASS_INDEX]:.4f}")
# -------------------------
# Save Confusion Matrix
# -------------------------
plt.figure(figsize=(8,6))

class_names = [
    "Background",
    "Impervious Surface",
    "Building",
    "Vegetation",
    "Tree"
]

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close()

print("Saved confusion matrix")
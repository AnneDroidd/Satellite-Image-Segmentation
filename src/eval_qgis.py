import os
import json
import numpy as np
from PIL import Image

import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/best_unet4.pth"
PALETTE_JSON = "potsdam_palette.json"

RGB_DIR = "./val_qgis_potsdam/rgb"
GT_DIR  = "./val_qgis_potsdam/mask_index"

OUT_DIR = "./val_qgis_potsdam/eval_outputs"

NUM_CLASSES = 5
IGNORE_VAL = 255

SAVE_PREVIEWS = True
PREVIEW_LIMIT = 200


# -------------------------
# CLASS METADATA
# -------------------------
CLASS_NAMES = {
    0: "background",
    1: "sealed_surface",
    2: "building",
    3: "vegetation",
    4: "tree"
}

# evaluation merge rule
MERGE_MAP = {4: 3, 0: 1}  # tree → vegetation, building → sealed_surface


# -------------------------
# UTILS
# -------------------------
def merge_classes(mask, mapping):
    mask = mask.copy()
    for src, dst in mapping.items():
        mask[mask == src] = dst
    return mask


def load_palette(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {v: tuple(map(int, k.split(","))) for k, v in raw.items()}


def mask_to_rgb(mask, palette):
    if palette is None:
        return np.zeros((*mask.shape, 3), dtype=np.uint8)

    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in palette.items():
        rgb[mask == cls] = color
    return rgb


def update_confusion_matrix(cm, gt, pred, num_classes, ignore_val):
    mask = (gt != ignore_val) & (gt >= 0) & (gt < num_classes)

    label = num_classes * gt[mask].astype(np.int64) + pred[mask].astype(np.int64)
    count = np.bincount(label, minlength=num_classes**2)

    cm += count.reshape(num_classes, num_classes)
    return cm


def compute_metrics(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    pixel_acc = tp.sum() / (cm.sum() + 1e-6)

    return iou, dice, pixel_acc


def frequency_weighted_iou(cm):
    freq = cm.sum(axis=1) / cm.sum()
    iou = np.diag(cm) / (
        cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-6
    )
    return (freq * iou).sum()

def precision_recall(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return precision, recall


def find_gt(rgb_name):
    stem = os.path.splitext(rgb_name)[0]
    return os.path.join(GT_DIR, f"{stem}_mask.png")

def compute_rgb_stats(image_dir):
    pixels = []

    for f in os.listdir(image_dir):
        if f.endswith(".png"):
            img = np.array(Image.open(os.path.join(image_dir, f)).convert("RGB"))
            pixels.append(img.reshape(-1, 3))

    pixels = np.concatenate(pixels, axis=0)

    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)

    return mean, std

# -------------------------
# INIT
# -------------------------
os.makedirs(OUT_DIR, exist_ok=True)

palette = load_palette(PALETTE_JSON)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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
# PREPROCESS
# -------------------------
transform = A.Compose([
    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),
    ToTensorV2()
])

# DOMAIN GAP VISUALIZATION
print("Computing RGB dataset statistics for domain gap visualization...")
mean, std = compute_rgb_stats(RGB_DIR)
print(f"\nRGB Mean: {mean}")
print(f"RGB Std: {std}")

# -------------------------
# EVALUATION
# -------------------------
conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

rgb_files = sorted([f for f in os.listdir(RGB_DIR) if f.endswith(".png")])

print("Dataset size:", len(rgb_files))

saved = 0
missing_gt = 0

for idx, fname in enumerate(rgb_files, start=1):

    rgb_path = os.path.join(RGB_DIR, fname)
    gt_path = find_gt(fname)

    if not os.path.exists(gt_path):
        missing_gt += 1
        continue

    img = np.array(Image.open(rgb_path).convert("RGB"))
    gt = np.array(Image.open(gt_path))

    if gt.ndim == 3:
        gt = gt[...,0]

    x = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

    # ---------- preview BEFORE merging ----------
    if SAVE_PREVIEWS and saved < PREVIEW_LIMIT:

        pred_rgb = mask_to_rgb(pred, palette)
        gt_rgb = mask_to_rgb(gt, palette)

        stem = os.path.splitext(fname)[0]

        Image.fromarray(img).save(f"{OUT_DIR}/{stem}_input.png")
        Image.fromarray(pred_rgb).save(f"{OUT_DIR}/{stem}_pred.png")
        Image.fromarray(gt_rgb).save(f"{OUT_DIR}/{stem}_gt.png")

        saved += 1

    # ---------- evaluation masks ----------
    pred_eval = merge_classes(pred, MERGE_MAP)
    gt_eval = merge_classes(gt, MERGE_MAP)

    # ---------- error map ----------
    error = (pred_eval != gt_eval).astype(np.uint8) * 255
    if SAVE_PREVIEWS and saved < PREVIEW_LIMIT:
        Image.fromarray(error).save(f"{OUT_DIR}/{stem}_error.png")

    conf_matrix = update_confusion_matrix(
        conf_matrix,
        gt_eval,
        pred_eval,
        NUM_CLASSES,
        IGNORE_VAL
    )

    print(f"{idx}/{len(rgb_files)}", end="\r")


print(f"\n[DONE] Processed {len(rgb_files)-missing_gt}/{len(rgb_files)}")


# -------------------------
# METRICS
# -------------------------
iou, dice, pixel_acc = compute_metrics(conf_matrix)

valid_classes = np.where(conf_matrix.sum(axis=1) > 0)[0]

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nPer-class metrics:")
for c in valid_classes:
    print(
        f"{CLASS_NAMES[c]:<12}"
        f"IoU: {iou[c]:.3f}   "
        f"Dice: {dice[c]:.3f}"
    )
#newwww
pred_pixels = conf_matrix.sum(axis=0)

print("\nPrediction distribution:")
for c in valid_classes:
    print(f"{CLASS_NAMES[c]}: {pred_pixels[c]}")
#----
miou = np.mean([iou[c] for c in valid_classes])

precision, recall = precision_recall(conf_matrix)
print("\nPer-class Precision and Recall:")
for c in valid_classes:
    print(
        f"{CLASS_NAMES[c]:<12}"
        f"Precision: {precision[c]:.3f}   "
        f"Recall: {recall[c]:.3f}"
    )

print("\nSummary metrics:")
print(f"Mean IoU: {miou:.4f}")
print(f"Pixel Accuracy: {pixel_acc:.4f}")
print(f"Frequency Weighted IoU: {frequency_weighted_iou(conf_matrix):.4f}")

print("\nGT pixel distribution:")
for c in valid_classes:
    pixels = conf_matrix[c].sum()
    print(f"{CLASS_NAMES[c]}: {pixels}")

print("\nRow-normalized confusion matrix:")
cm_norm = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-6)
print(np.round(cm_norm,3))
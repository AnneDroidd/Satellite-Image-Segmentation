import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# --------------------------------
# CONFIG
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "bamberg": BASE_DIR / "val_qgis_bamberg" / "eval_outputs",
    "erlangen": BASE_DIR / "val_qgis_erlangen" / "eval_outputs",
    "potsdam": BASE_DIR / "val_qgis_potsdam" / "eval_outputs"
}

SEALED_CLASS = 1


# --------------------------------
# SHADOW DETECTION (HSV)
# --------------------------------

def get_shadow_mask(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    H, S, V = cv2.split(hsv)

    S = S / 255.0
    V = V / 255.0

    shadow = (V < 0.45) & (S < 0.35)

    return shadow


# --------------------------------
# IOU
# --------------------------------

def compute_iou(pred, gt):

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:   # avoid tiny regions
        return np.nan

    return intersection / union


# --------------------------------
# VISUALIZATION
# --------------------------------

def visualize(
    img,
    shadow_mask,
    pred_sealed,
    gt_sealed,
    city,
    name,
    shadow_iou,
    non_shadow_iou
):

    IGNORE_COLOR = np.array([0,0,0])
    valid_mask = ~np.all(gt == IGNORE_COLOR, axis=-1)
    error_mask = (np.logical_xor(pred_sealed, gt_sealed) & valid_mask)

    overlay = img.copy()

    # red = shadow
    overlay[shadow_mask] = [255, 0, 0]

    # yellow = segmentation error
    overlay[error_mask] = [255, 255, 0]

    fig, axes = plt.subplots(2, 3, figsize=(15,10))

    # RGB image
    axes[0,0].imshow(img)
    axes[0,0].set_title("RGB Image")
    axes[0,0].axis("off")

    # Shadow mask
    axes[0,1].imshow(shadow_mask, cmap="gray")
    axes[0,1].set_title("Shadow Mask")
    axes[0,1].axis("off")

    # GT sealed
    axes[0,2].imshow(gt_sealed, cmap="gray")
    axes[0,2].set_title("GT Sealed")
    axes[0,2].axis("off")

    # Pred sealed
    axes[1,0].imshow(pred_sealed, cmap="gray")
    axes[1,0].set_title("Pred Sealed")
    axes[1,0].axis("off")

    # Error map
    axes[1,1].imshow(error_mask, cmap="hot")
    axes[1,1].set_title("Segmentation Errors")
    axes[1,1].axis("off")

    # Overlay
    axes[1,2].imshow(overlay)
    axes[1,2].set_title("Shadow + Error Overlay")
    axes[1,2].axis("off")

    fig.suptitle(
        f"{city} | {name}\n"
        f"Shadow IoU: {shadow_iou:.3f} | "
        f"Non-Shadow IoU: {non_shadow_iou:.3f}",
        fontsize=14
    )

    plt.tight_layout()

    save_dir = BASE_DIR / "shadow_debug"
    save_dir.mkdir(exist_ok=True)

    plt.savefig(
        save_dir / f"{city}_{name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------
# MAIN ANALYSIS
# --------------------------------

for city, folder in DATASETS.items():

    print(f"\nProcessing {city}")

    shadow_ious = []
    non_shadow_ious = []
    shadow_percentages = []
    shadow_errors = []
    non_shadow_errors = []

    images = sorted(folder.glob("*_input.png"))
    print(f"{city}: Found {len(images)} images")

    for img_path in images:

        name = img_path.name.replace("_input.png","")

        pred_path = folder / f"{name}_pred.png"
        gt_path = folder / f"{name}_gt.png"

        if not pred_path.exists() or not gt_path.exists():
            continue

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        shadow_mask = get_shadow_mask(img)
        shadow_percentages.append(shadow_mask.mean())

        pred = cv2.imread(str(pred_path))
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

        gt = cv2.imread(str(gt_path))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        SEALED_COLOR = np.array([255,255,255])

        pred_sealed = np.all(pred == SEALED_COLOR, axis=-1)
        gt_sealed = np.all(gt == SEALED_COLOR, axis=-1)

        # new
        # valid_shadow = shadow_mask & (gt_sealed | pred_sealed)
        # shadow_iou = compute_iou( pred_sealed[valid_shadow], gt_sealed[valid_shadow])
        # # new

        # # shadow_iou = compute_iou(
        # #     pred_sealed[shadow_mask],
        # #     gt_sealed[shadow_mask]
        # # )

        # non_shadow_iou = compute_iou(
        #     pred_sealed[~shadow_mask],
        #     gt_sealed[~shadow_mask]
        # )
        
        # shadow_region = shadow_mask
        # non_shadow_region = ~shadow_mask

        # shadow_iou = compute_iou(
        #     pred_sealed & shadow_region,
        #     gt_sealed & shadow_region
        # )

        # non_shadow_iou = compute_iou(
        #     pred_sealed & non_shadow_region,
        #     gt_sealed & non_shadow_region
        # )
        IGNORE_COLOR = np.array([0,0,0])
        valid_mask = ~np.all(gt == IGNORE_COLOR, axis=-1)
        shadow_region = shadow_mask & valid_mask
        non_shadow_region = (~shadow_mask) & valid_mask
        
        shadow_iou = compute_iou( pred_sealed[shadow_region], gt_sealed[shadow_region])
        non_shadow_iou = compute_iou(pred_sealed[non_shadow_region], gt_sealed[non_shadow_region])

        shadow_ious.append(shadow_iou)
        non_shadow_ious.append(non_shadow_iou)
        # shadow_error = np.logical_xor(pred_sealed, gt_sealed) & shadow_mask
        # non_shadow_error = np.logical_xor(pred_sealed, gt_sealed) & ~shadow_mask

        # shadow_error_rate = shadow_error.sum() / shadow_mask.sum()
        # non_shadow_error_rate = non_shadow_error.sum() / (~shadow_mask).sum()
        # shadow_error = np.logical_xor(pred_sealed, gt_sealed) & shadow_mask
        # non_shadow_error = np.logical_xor(pred_sealed, gt_sealed) & ~shadow_mask
        shadow_error = (np.logical_xor(pred_sealed, gt_sealed) & shadow_region)
        non_shadow_error = (np.logical_xor(pred_sealed, gt_sealed)& non_shadow_region)

        # if shadow_mask.sum() > 0:
        #     # shadow_errors.append(shadow_error.sum() / shadow_mask.sum())
        #     shadow_errors.append(shadow_region.sum())
        
        # if (~shadow_mask).sum() > 0:
        #     non_shadow_errors.append(
        #         non_shadow_error.sum() / (~shadow_mask).sum()
        #     )
        if shadow_region.sum() > 0: 
            shadow_errors.append( shadow_error.sum() / shadow_region.sum())
        
        if non_shadow_region.sum() > 0:
            non_shadow_errors.append(non_shadow_error.sum() / non_shadow_region.sum())
 
    print("Shadow error:", np.mean(shadow_errors))
    print("Non-shadow error:", np.mean(non_shadow_errors))

    print("Shadow pixels:", shadow_mask.sum())
    print("Sealed pixels GT:", gt_sealed.sum())
    print("Sealed pixels Pred:", pred_sealed.sum())
    

    print("Shadow IoU:", np.nanmean(shadow_ious))
    print("Non-Shadow IoU:", np.nanmean(non_shadow_ious))
    print("Shadow %:", np.mean(shadow_percentages))
    
    
import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# CONFIG
# -------------------------

DATASETS = {
    "Bamberg": "./val_qgis_bamberg/eval_outputs",
    "Erlangen": "./val_qgis_erlangen/eval_outputs",
    "Potsdam": "./val_qgis_potsdam/eval_outputs"
}

palette = {
    (0,0,0):0,
    (255,255,255):1,   # sealed
    (0,0,255):2,       # building
    (0,255,255):3,     # low veg
    (0,255,0):4,       # tree
}

CLS_SEALED = 1
CLS_BUILDING = 2
CLS_VEG = 3
IGNORE = 255


# -------------------------
# RGB → CLASS
# -------------------------

def rgb_to_class(mask):
    h, w, _ = mask.shape
    class_map = np.zeros((h, w), dtype=np.uint8)

    for rgb, cls in palette.items():
        matches = np.all(mask == rgb, axis=-1)
        class_map[matches] = cls

    return class_map


# -------------------------
# CONFUSION MATRIX
# -------------------------

def confusion_matrix(gt, pred, labels):
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)

    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            cm[i, j] = np.sum((gt == l1) & (pred == l2))

    return cm


# -------------------------
# MAIN LOOP
# -------------------------

records = []

for city, EVAL_DIR in DATASETS.items():

    print(f"\nProcessing {city}")

    # files = [f for f in os.listdir(EVAL_DIR) if f.endswith("_gt.png")]
    files = [f for f in os.listdir(EVAL_DIR) if "_gt" in f.lower()]

    print(f"{city}: Found {len(files)} tiles")

    for gt_file in sorted(files):

        idx = gt_file.replace("_gt.png", "")

        gt_path = os.path.join(EVAL_DIR, f"{idx}_gt.png")
        pred_path = os.path.join(EVAL_DIR, f"{idx}_pred.png")

        if not os.path.exists(pred_path):
            print("Missing prediction:", pred_path)
            continue

        # load
        gt = rgb_to_class(np.array(Image.open(gt_path)))
        pred = rgb_to_class(np.array(Image.open(pred_path)))

        # merge vegetation
        gt[gt == 4] = 3
        pred[pred == 4] = 3

        valid = gt != IGNORE
        gt = gt[valid]
        pred = pred[valid]

        if len(gt) == 0:
            continue

        total = len(gt)

        sealed_ratio = np.sum(gt == CLS_SEALED) / total
        building_ratio = np.sum(gt == CLS_BUILDING) / total
        veg_ratio = np.sum(gt == CLS_VEG) / total

        cm = confusion_matrix(
            gt,
            pred,
            labels=[CLS_SEALED, CLS_BUILDING, CLS_VEG]
        )

        ious = []

        for i in range(3):

            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            denom = tp + fp + fn

            if denom == 0:
                iou = np.nan
            else:
                iou = tp / denom

            ious.append(iou)

        records.append({
            "city": city,
            "tile": idx,
            "sealed_ratio": sealed_ratio,
            "building_iou": ious[1],
            "veg_iou": ious[2],
            "sealed_iou": ious[0]
        })


# -------------------------
# DATAFRAME
# -------------------------

df = pd.DataFrame(records)

print("\nOverall Performance:")
print(df.groupby("city")[["sealed_iou","building_iou","veg_iou"]].mean())


# -------------------------
# GRAPH 1: Vegetation vs Sealed (All Cities)
# -------------------------

plt.figure()

for city in df.city.unique():
    subset = df[df.city == city]
    plt.scatter(
        subset["sealed_ratio"],
        subset["veg_iou"],
        label=city,
        alpha=0.7
    )

plt.xlabel("Sealed Surface Ratio")
plt.ylabel("Vegetation IoU")
plt.title("Urban Density vs Vegetation Performance (All Cities)")
plt.legend()
plt.savefig("veg_vs_sealed_all_cities.png", dpi=300)
plt.close()


# -------------------------
# GRAPH 2: Building vs Sealed
# -------------------------

plt.figure()

for city in df.city.unique():
    subset = df[df.city == city]
    plt.scatter(
        subset["sealed_ratio"],
        subset["building_iou"],
        label=city,
        alpha=0.7
    )

plt.xlabel("Sealed Surface Ratio")
plt.ylabel("Building IoU")
plt.title("Urban Density vs Building Performance (All Cities)")
plt.legend()
plt.savefig("building_vs_sealed_all_cities.png", dpi=300)
plt.close()


# -------------------------
# GRAPH 3: Urban vs Rural
# -------------------------

threshold = df["sealed_ratio"].median()

urban = df[df.sealed_ratio > threshold]
rural = df[df.sealed_ratio <= threshold]

print("\nUrban performance:")
print(urban.groupby("city")[["sealed_iou","building_iou","veg_iou"]].mean())

print("\nRural performance:")
print(rural.groupby("city")[["sealed_iou","building_iou","veg_iou"]].mean())


plt.figure()

data = [df[df.city == c]["veg_iou"].dropna() for c in df.city.unique()]

plt.boxplot(data)
plt.xticks(range(1,len(df.city.unique())+1), df.city.unique())

plt.ylabel("Vegetation IoU")
plt.title("Vegetation Performance Per City")

plt.savefig("city_comparison.png", dpi=300)
plt.close()


print("\nGraphs saved:")
print(" - veg_vs_sealed_all_cities.png")
print(" - building_vs_sealed_all_cities.png")
print(" - city_comparison.png")
#!/usr/bin/env python3
"""
prepare_data.py

Tiles images and masks, converts RGB masks to integer class masks,
and writes train/val CSV splits (group-split by source image).

Usage example:
python prepare_data.py \
  --img_dir /path/to/potsdam/images \
  --mask_dir /path/to/potsdam/labels \
  --out_dir dataset/yourdata \
  --tile_size 256 --stride 128 \
  --val_split 0.2 \
  --palette palette.json

If --palette is omitted, the script will auto-generate generated_palette.json
in the output folder and use it (you should inspect & edit if needed).
"""

import os
import json
import argparse
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import csv

# -------------------------
# CONFIG
# -------------------------
TREE_CLASS = 4          # index for "tree" in palette
MIN_TREE_PIXELS = 50    # minimum pixels to keep a tile

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def parse_rgb_tuple(s):
    # parse "R,G,B" -> (R,G,B)
    parts = s.split(',')
    return tuple(int(p.strip()) for p in parts)

def scan_unique_colors(mask_paths, max_images=50):
    """Scan up to max_images masks and return a sorted list of unique RGB colors found."""
    colors = set()
    sampled = mask_paths[:max_images]
    for p in sampled:
        m = Image.open(p).convert("RGB")
        arr = np.asarray(m)
        h,w,_ = arr.shape
        flat = arr.reshape(-1,3)
        # convert to small unique set per image
        for c in np.unique(flat.reshape(-1,3), axis=0):
            colors.add(tuple(int(x) for x in c.tolist()))
    colors_list = sorted(list(colors))
    return colors_list

def save_palette_json(palette_dict, out_path):
    with open(out_path, "w") as f:
        # convert keys to "R,G,B" strings for JSON friendliness
        jsonable = { ",".join(map(str, map(int, map(int, k.split(','))))): v for k,v in palette_dict.items() } if isinstance(list(palette_dict.keys())[0], str) else {}
        if jsonable:
            json.dump(palette_dict, f, indent=2)
        else:
            # palette_dict may already have tuple keys; convert them
            conv = { ",".join(map(str, k)): int(v) for k,v in palette_dict.items() }
            json.dump(conv, f, indent=2)
    print(f"[INFO] Saved palette JSON to {out_path}")

def load_palette_json(pal_path):
    with open(pal_path, "r") as f:
        raw = json.load(f)
    # raw keys should be "R,G,B" strings, values ints
    palette = { tuple(int(x) for x in k.split(',')): int(v) for k,v in raw.items() }
    return palette

def rgb_to_index_mask(rgb_arr, palette, default_index=0):
    """
    rgb_arr: HxWx3 uint8 numpy array
    palette: dict {(r,g,b): index}
    Returns HxW uint8/uint16 depending on max index
    """
    h,w,_ = rgb_arr.shape
    flat = rgb_arr.reshape(-1,3)
    packed = (flat[:,0].astype(np.int32) << 16) + (flat[:,1].astype(np.int32) << 8) + flat[:,2].astype(np.int32)
    # build mapping from packed int -> idx for speed
    mapping = {}
    for (r,g,b), idx in palette.items():
        key = (int(r) << 16) + (int(g) << 8) + int(b)
        mapping[key] = int(idx)
    default = int(default_index)
    out_flat = np.full((flat.shape[0],), default, dtype=np.int32)
    for key, idx in mapping.items():
        matches = (packed == key)
        if matches.any():
            out_flat[matches] = idx
    out = out_flat.reshape(h,w)
    # choose dtype based on max index
    maxidx = out.max()
    if maxidx < 256:
        return out.astype(np.uint8)
    elif maxidx < 65536:
        return out.astype(np.uint16)
    else:
        return out.astype(np.int32)


def tile_and_save_image(im_path, out_dir, prefix,
                        tile_size=256, stride=128,
                        keep_indices=None):
    im = Image.open(im_path).convert("RGB")
    W, H = im.size

    tile_idx = 0

    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            if keep_indices is not None and tile_idx not in keep_indices:
                tile_idx += 1
                continue

            tile = im.crop((x, y, x + tile_size, y + tile_size))
            out_path = os.path.join(out_dir, f"{prefix}_{tile_idx:05d}.png")
            tile.save(out_path)

            tile_idx += 1


def tile_and_save_mask(mask_path, out_dir, prefix, palette,
                       tile_size=256, stride=128):
    mask = Image.open(mask_path).convert("RGB")
    W, H = mask.size
    arr = np.asarray(mask)

    kept_indices = []
    tile_idx = 0

    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            sub = arr[y:y+tile_size, x:x+tile_size, :]
            idx_mask = rgb_to_index_mask(sub, palette, default_index=0)

            # ---- FILTER DECISION ----
            if np.sum(idx_mask == TREE_CLASS) < MIN_TREE_PIXELS:
                tile_idx += 1
                continue


            out = Image.fromarray(idx_mask)
            out_path = os.path.join(out_dir, f"{prefix}_{tile_idx:05d}.png")
            out.save(out_path)

            kept_indices.append(tile_idx)
            tile_idx += 1

    return kept_indices

def gather_pairs(img_dir, mask_dir, img_exts=(".png",".tif",".jpg",".jpeg")):
    images = []
    for ext in img_exts:
        images += sorted(glob(os.path.join(img_dir, f"*{ext}")))
    masks = []
    for ext in img_exts:
        masks += sorted(glob(os.path.join(mask_dir, f"*{ext}")))
    # naive pair matching by basename prefix — better if file naming matches
    images = sorted(images)
    masks = sorted(masks)
    if len(images) != len(masks):
        print(f"[WARN] Found {len(images)} images and {len(masks)} masks. Proceeding with pair-wise zipped list. Ensure they match.")
    pairs = list(zip(images, masks))
    return pairs

def write_csv_pairs(pairs, csv_path):
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path", "orig_image_id"])
        for img, m, orig in pairs:
            writer.writerow([img, m, orig])

def main(args):
    random.seed(args.seed)
    ensure_dir(args.out_dir)
    tiles_img_dir = os.path.join(args.out_dir, "tiles", "images")
    tiles_mask_dir = os.path.join(args.out_dir, "tiles", "masks")
    csv_dir = os.path.join(args.out_dir, "csv")
    ensure_dir(tiles_img_dir)
    ensure_dir(tiles_mask_dir)
    ensure_dir(csv_dir)

    # gather raw image/mask pairs
    pairs = gather_pairs(args.img_dir, args.mask_dir)
    if len(pairs) == 0:
        raise SystemExit("[ERROR] No image/mask pairs found. Check paths and extensions.")

    # load or auto-generate palette
    palette = None
    if args.palette:
        print(f"[INFO] Loading palette from {args.palette}")
        palette = load_palette_json(args.palette)
    else:
        print("[INFO] No palette JSON supplied. Scanning masks to discover unique colors...")
        mask_paths = [m for _,m in pairs]
        unique_colors = scan_unique_colors(mask_paths, max_images=args.palette_scan_max)
        print(f"[INFO] Discovered {len(unique_colors)} unique colors (showing up to first 50):")
        for c in unique_colors[:50]:
            print(c)
        # create auto mapping and save it
        palette = {}
        for i, c in enumerate(unique_colors):
            palette[c] = int(i)  # assign incremental index; user should edit this file if necessary
        gen_pal_path = os.path.join(args.out_dir, "generated_palette.json")
        # gen_pal_path = os.path.join(args.out_dir, "potsdam_palette.json")
        # convert keys to "R,G,B"
        json_pal = { ",".join(map(str,k)): v for k,v in palette.items() }
        with open(gen_pal_path, "w") as f:
            json.dump(json_pal, f, indent=2)
        print(f"[WARN] Auto-generated palette saved to {gen_pal_path}")
        print("[WARN] Inspect generated_palette.json and edit class indices if needed, then re-run with --palette <path> for correct semantic mapping.")

    # tile each pair; keep record of produced tiles
    produced_pairs = []
    print("[INFO] Tiling images and masks...")
    for i, (img_path, mask_path) in enumerate(tqdm(pairs, desc="pairs")):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        prefix = f"{basename}"
        # n_img = tile_and_save_image(img_path, tiles_img_dir, prefix, tile_size=args.tile_size, stride=args.stride)
        # n_mask = tile_and_save_mask(mask_path, tiles_mask_dir, prefix, palette, tile_size=args.tile_size, stride=args.stride)
        # if n_img != n_mask:
        #     print(f"[WARN] For {basename} produced {n_img} image tiles but {n_mask} mask tiles. Check mask colors / palette.")
        # # append produced tile paths to list
        # for idx in range(n_img):
        #     produced_pairs.append((
        #         os.path.join(tiles_img_dir, f"{prefix}_{idx:05d}.png"),
        #         os.path.join(tiles_mask_dir, f"{prefix}_{idx:05d}.png"),
        #         basename
        #     ))
        kept_indices = tile_and_save_mask(mask_path,
                                          tiles_mask_dir,
                                          prefix,
                                          palette,
                                          tile_size=args.tile_size,
                                          stride=args.stride
        )
        tile_and_save_image(img_path,
                            tiles_img_dir,
                            prefix,
                            tile_size=args.tile_size,
                            stride=args.stride,
                            keep_indices=kept_indices
        )
        for idx in kept_indices:
            produced_pairs.append((
                os.path.join(tiles_img_dir, f"{prefix}_{idx:05d}.png"),
                os.path.join(tiles_mask_dir, f"{prefix}_{idx:05d}.png"),
                basename
            ))


    print(f"[INFO] Produced {len(produced_pairs)} tile pairs.")

    # group-split by original image id (basename) to create train/val splits
    groups = {}
    for imgp, mskp, orig in produced_pairs:
        groups.setdefault(orig, []).append((imgp, mskp, orig))
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    n_val_groups = max(1, int(len(group_keys) * args.val_split))
    val_group_keys = set(group_keys[:n_val_groups])
    train_pairs = []
    val_pairs = []
    for k in group_keys:
        if k in val_group_keys:
            val_pairs.extend(groups[k])
        else:
            train_pairs.extend(groups[k])

    train_csv = os.path.join(csv_dir, "train.csv")
    val_csv = os.path.join(csv_dir, "val.csv")
    write_csv_pairs(train_pairs, train_csv)
    write_csv_pairs(val_pairs, val_csv)

    print(f"[INFO] Wrote {len(train_pairs)} train tiles to {train_csv}")
    print(f"[INFO] Wrote {len(val_pairs)} val tiles to {val_csv}")

    # save palette used (convert tuple keys to "R,G,B")
    # palette_json_path = os.path.join(args.out_dir, "used_palette.json")
    palette_json_path = os.path.join(args.out_dir, "potsdam_palette.json")
    conv = { ",".join(map(str,k)): int(v) for k,v in palette.items() }
    with open(palette_json_path, "w") as f:
        json.dump(conv, f, indent=2)
    print(f"[INFO] Saved used palette to {palette_json_path}")

    print("[DONE] Data preparation finished.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", required=True, help="Path to folder with raw RGB images")
    p.add_argument("--mask_dir", required=True, help="Path to folder with raw RGB mask images")
    p.add_argument("--out_dir", required=True, help="Output base folder (tiles, csv, palette)")
    p.add_argument("--tile_size", type=int, default=256, help="Tile size (default 256)")
    p.add_argument("--stride", type=int, default=128, help="Tile stride (default 128 for 50%% overlap)")
    p.add_argument("--val_split", type=float, default=0.2, help="Fraction of groups (original images) for val (default 0.2)")
    p.add_argument("--palette", type=str, default=None, help="Path to palette JSON file mapping 'R,G,B' -> index. If omitted, script auto-generates palette.")
    p.add_argument("--palette_scan_max", type=int, default=50, help="How many mask files to scan to discover unique colors when palette omitted.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)

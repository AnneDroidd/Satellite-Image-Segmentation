from qgis.core import (
    QgsProject, QgsRectangle, QgsMapSettings, QgsMapRendererParallelJob,
    QgsFeatureRequest, QgsWkbTypes
)
from qgis.utils import iface
from PyQt5.QtCore import QSize, QEventLoop
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtWidgets import QApplication
import os, time
import numpy as np
from PIL import Image

# =========================
# CONFIG — EDIT THESE
# =========================
OUTPUT_DIR = r"C:\/Users\sroy\Documents\Study Materials\/0 Image Segmentation Project\val_qgis_erlangen"  # <-- change
CITY_GROUP_NAME = "CityErlangen"  # <-- change
TREE_POINTS_LAYER_NAME = "treeLocations"

TREE_PREFIX = "greenAtta"     # trees + immediate surrounding (treat as tree)
VEG_PREFIX  = "greenDeta"     # vegetation
BLD_PREFIX  = "buildings"     # buildings

TILE_PX = 256
DPI = 96

# "bit of surrounding" in map units (meters if CRS is metric)
TILE_SIZE_MAP_UNITS = 30.0

# If you only want the trees you selected in QGIS:
EXPORT_ONLY_SELECTED = False

# Small wait helps XYZ/WMS tiles render fully
TILE_WAIT_SECONDS = 0.3
# =========================

# Class indices (match your training convention: TREE_CLASS=4; NUM_CLASSES=5)
CLS_CLOSED_SURFACE = 1
CLS_BUILDING       = 2
CLS_VEGETATION     = 3
CLS_TREE           = 4
CLS_IGNORE = 255

def find_layer_in_group(group_name: str, layer_name: str):
    root = QgsProject.instance().layerTreeRoot()
    grp = root.findGroup(group_name)
    if grp is None:
        return None
    for node in grp.findLayers():
        lyr = node.layer()
        if lyr and lyr.name() == layer_name:
            return lyr
    return None

def layers_by_prefix(prefix: str):
    return [lyr for lyr in QgsProject.instance().mapLayers().values()
            if lyr.name().startswith(prefix)]

def render_qimage(layers, extent: QgsRectangle, background: QColor, draw_labels: bool):
    settings = QgsMapSettings()
    settings.setLayers(layers)
    settings.setBackgroundColor(background)
    settings.setOutputSize(QSize(TILE_PX, TILE_PX))
    settings.setExtent(extent)
    settings.setOutputDpi(DPI)
    settings.setDestinationCrs(QgsProject.instance().crs())

    settings.setFlag(QgsMapSettings.Antialiasing, True)
    settings.setFlag(QgsMapSettings.UseAdvancedEffects, True)
    settings.setFlag(QgsMapSettings.HighQualityImageTransforms, True)
    settings.setFlag(QgsMapSettings.DrawLabeling, draw_labels)

    job = QgsMapRendererParallelJob(settings)
    loop = QEventLoop()
    job.finished.connect(loop.quit)
    job.start()
    loop.exec_()

    if TILE_WAIT_SECONDS > 0:
        time.sleep(TILE_WAIT_SECONDS)
        QApplication.processEvents()

    return job.renderedImage()

def qimage_to_binary_mask(img: QImage, threshold=10):
    """
    Convert rendered QImage (white shapes on black bg) to binary mask (0/1)
    using brightness threshold.
    """
    # Force ARGB32 for predictable byte layout
    img = img.convertToFormat(QImage.Format_ARGB32)
    w, h = img.width(), img.height()

    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((h, img.bytesPerLine() // 4, 4))[:, :w, :]

    # QImage.Format_ARGB32 is stored as BGRA on little endian
    b = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    r = arr[..., 2].astype(np.int16)

    # simple brightness
    bright = (r + g + b) // 3
    return (bright > threshold).astype(np.uint8)

def main():
    project = QgsProject.instance()

    # --- 1) Export centers = Erlangen labelled tree points
    tree_pts = find_layer_in_group(CITY_GROUP_NAME, TREE_POINTS_LAYER_NAME)
    if not tree_pts:
        raise RuntimeError(f"Couldn't find '{TREE_POINTS_LAYER_NAME}' under group '{CITY_GROUP_NAME}'.")

    # --- 2) Mask layers
    tree_layers = layers_by_prefix(TREE_PREFIX)
    veg_layers  = layers_by_prefix(VEG_PREFIX)
    bld_layers  = layers_by_prefix(BLD_PREFIX)

    if not tree_layers:
        raise RuntimeError(f"No layers found starting with '{TREE_PREFIX}'.")
    if not veg_layers:
        print(f"[WARN] No layers found starting with '{VEG_PREFIX}'. Vegetation class will be empty.")
    if not bld_layers:
        print(f"[WARN] No layers found starting with '{BLD_PREFIX}'. Building class will be empty.")

    print("Tree mask layers:", [l.name() for l in tree_layers])
    print("Veg  mask layers:", [l.name() for l in veg_layers])
    print("Bld  mask layers:", [l.name() for l in bld_layers])

    # --- 3) RGB layers: take what is visible in canvas, but remove annotation layers
    canvas_layers = iface.mapCanvas().layers()
    remove_ids = {l.id() for l in (tree_layers + veg_layers + bld_layers)}
    remove_ids.add(tree_pts.id())  # don't draw points on RGB

    rgb_layers = [l for l in canvas_layers if l.id() not in remove_ids]
    print("\nRGB layers used:", [l.name() for l in rgb_layers])

    # --- Output dirs
    rgb_dir = os.path.join(OUTPUT_DIR, "rgb")
    msk_dir = os.path.join(OUTPUT_DIR, "mask_index")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    # --- Features to export
    if EXPORT_ONLY_SELECTED:
        fids = tree_pts.selectedFeatureIds()
        feats = list(tree_pts.getFeatures(QgsFeatureRequest().setFilterFids(fids)))
        print(f"\nExporting ONLY selected labelled trees: {len(feats)}")
    else:
        feats = list(tree_pts.getFeatures())
        print(f"\nExporting ALL Erlangen labelled trees: {len(feats)}")

    half = TILE_SIZE_MAP_UNITS / 2.0

    for i, f in enumerate(feats, start=1):
        fid = f.id()
        geom = f.geometry()
        if not geom or geom.isEmpty():
            continue

        if QgsWkbTypes.geometryType(geom.wkbType()) == QgsWkbTypes.PointGeometry:
            c = geom.asPoint()
        else:
            c = geom.centroid().asPoint()

        extent = QgsRectangle(c.x() - half, c.y() - half, c.x() + half, c.y() + half)

        # --- RGB tile
        rgb_img = render_qimage(rgb_layers, extent, QColor(255,255,255), draw_labels=False)
        rgb_path = os.path.join(rgb_dir, f"{fid}.png")
        rgb_img.save(rgb_path, "PNG", 100)

        # --- Render class masks as binary (white on black)
        # IMPORTANT: style these layers in QGIS as SOLID WHITE fill (no outline) for clean masks.
        tree_img = render_qimage(tree_layers, extent, QColor(0,0,0), draw_labels=False)
        tree_bin = qimage_to_binary_mask(tree_img)

        if veg_layers:
            veg_img = render_qimage(veg_layers, extent, QColor(0,0,0), draw_labels=False)
            veg_bin = qimage_to_binary_mask(veg_img)
        else:
            veg_bin = np.zeros((TILE_PX, TILE_PX), dtype=np.uint8)

        if bld_layers:
            bld_img = render_qimage(bld_layers, extent, QColor(0,0,0), draw_labels=False)
            bld_bin = qimage_to_binary_mask(bld_img)
        else:
            bld_bin = np.zeros((TILE_PX, TILE_PX), dtype=np.uint8)

        # create circular region mask
        # yy, xx = np.ogrid[:TILE_PX, :TILE_PX]
        # center = TILE_PX // 2
        # radius = TILE_PX // 2
        # circle_mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
        meters_per_pixel = TILE_SIZE_MAP_UNITS / TILE_PX
        radius_m = 7.5
        radius_px = int(radius_m / meters_per_pixel)
        yy, xx = np.ogrid[:TILE_PX, :TILE_PX]
        center = TILE_PX // 2
        circle_mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius_px ** 2
        
        # --- Compose 5-class index mask (uint8)
        # out = np.full((TILE_PX, TILE_PX), CLS_CLOSED_SURFACE, dtype=np.uint8)
        out = np.full((TILE_PX, TILE_PX), CLS_IGNORE, dtype=np.uint8)

        # # priority: building -> vegetation -> tree
        # out[bld_bin == 1]  = CLS_BUILDING
        # out[veg_bin == 1]  = CLS_VEGETATION
        # out[tree_bin == 1] = CLS_TREE

        # inside circle = sealed surface initially
        out[circle_mask] = CLS_CLOSED_SURFACE
        
        # overwrite with actual classes (priority order)
        out[circle_mask & (bld_bin == 1)]  = CLS_BUILDING
        out[circle_mask & (veg_bin == 1)]  = CLS_VEGETATION
        out[circle_mask & (tree_bin == 1)] = CLS_TREE

        mask_path = os.path.join(msk_dir, f"{fid}_mask.png")
        Image.fromarray(out, mode="L").save(mask_path)

        if i % 25 == 0 or i == 1:
            print(f"  Exported {i}/{len(feats)}")

    print("\nDONE.")
    print("RGB tiles:", rgb_dir)
    print("Index masks:", msk_dir)
    print(f"Mask classes used: closed={CLS_IGNORE}, sealed={CLS_CLOSED_SURFACE} building={CLS_BUILDING}, veg={CLS_VEGETATION}, tree={CLS_TREE}")

main()
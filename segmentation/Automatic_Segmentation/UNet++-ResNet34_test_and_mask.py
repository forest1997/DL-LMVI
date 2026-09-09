"""
UNet++ (ResNet34) + Cellpose + Robust Lipid Detection (TUNED DOWN)

Changes vs previous:
✅ Lipids restricted to CYTOPLASM (reduces false positives in nucleus/edges)
✅ Lipid thresholds tightened (less background)
✅ Cleanup changed to OPENING (shrinks/removes noise instead of growing blobs)
✅ Combined overlay lipid display toned down (lower alpha, less/no dilation, no sparkle edge)

Outputs per image:

A) Masks (binary PNGs):
  Outputs2/FinalMasks/
    Cell_binary/
    Cell_outline/
    Cytoplasm_binary/
    Nucleus_binary/
    Nucleolus_binary/
    Lipid_binary/

B) Visualizations:
  1) 2-panel: [Original | Combined Overlay]
     Outputs2/Visualizations2/Original_vs_AllCombinedOverlay/

  2) Separate panels:
     Outputs2/Visualizations2/Panels_Separate/
        Original/
        Cell/
        Cytoplasm/
        Nucleus/
        Nucleolus/
        Lipids/
        Combined/

  3) Lipid-only RGB (black background):
     Outputs2/Visualizations2/LipidOnly/

  4) Optional Merge vs Zoom:
     Outputs2/Visualizations2/Merge_vs_ZoomedIn/

  5) ✅ NEW: Full prediction image (Combined overlay only)
     Outputs2/Visualizations2/FullPrediction/
"""

import os
import glob
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import torch
import torch.nn as nn

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(PIPELINE_ROOT)
os.environ.setdefault(
    "CELLPOSE_LOCAL_MODELS_PATH",
    os.path.join(PIPELINE_ROOT, "cellpose_models"),
)

# =============== CELLPOSE IMPORT (avoid name conflict) ===============
# IMPORTANT: your file must NOT be named "cellpose.py"
try:
    from cellpose import io as cp_io
    from cellpose import models as cp_models
except ImportError:
    print("❌ ERROR: Could not import cellpose.")
    print("Make sure cellpose is installed and your script is NOT named 'cellpose.py'.")
    sys.exit(1)

# ===================== CONFIG =====================
PREDICT_DATA_ROOT = os.environ.get(
    "CELLSEG_PREDICT_DATA_ROOT",
    os.path.join(WORKSPACE_ROOT, "Cell segment"),
)
ORIGINAL_FOLDER = os.environ.get("CELLSEG_ORIGINAL_FOLDER", "OA")

MODEL_PATH = os.path.join(
    PIPELINE_ROOT,
    "trained_models",
    "best_unetpp_resnet34.pth",
)
UNET_IMG_SIZE = 512
UNET_THRESHOLD = 0.5
MASK_THRESHOLD = 127

# ---- Cellpose ----
CELLPOSE_MODEL_TYPE = "cyto3"
CELLPOSE_CHANNELS = [2, 3]       # [cyto, nuclei] : 0=gray, 1=R, 2=G, 3=B
CELLPOSE_DIAMETER = 30           # set 0 or None for auto
CELLPOSE_FLOW_THRESHOLD = 0.4
CELLPOSE_CELLPROB_THRESHOLD = 0.0
CELLPOSE_BATCH_SIZE = 1

# ---- Lipid detection (TUNED DOWN) ----
# narrower hue and stricter S/V reduces background
LIPID_H_LO = 15 / 360.0
LIPID_H_HI = 70 / 360.0
LIPID_S_MIN = 0.25
LIPID_V_MIN = 0.25
LIPID_CLEANUP = True

# stricter yellow-score gate reduces false positives
LIPID_YELLOW_SCORE_THR = 0.18
LIPID_MIN_RG = 0.25

# ---- Combined overlay lipid display (TUNED DOWN) ----
COLOR_LIPID_OVERLAY = (255, 220, 0)   # bright yellow
LIPID_ALPHA_COMBINED = 160            # was 240
LIPID_DISPLAY_DILATE_K = 1            # was 3; 1 = basically no dilation

LIPID_SPARKLE_EDGE = False            # was True (makes it look “over expressed”)
LIPID_EDGE_ALPHA = 220
LIPID_EDGE_DILATE_K = 3

# ---- Zoom settings ----
MAKE_ZOOM = True
ZOOM_PAD = 40
ZOOM_MIN_SIZE = 96
ZOOM_SCALE = 2.0
ZOOM_BOX_COLOR = (255, 42, 42)  # red

# ---- Output root ----
OUT_ROOT = os.environ.get(
    "CELLSEG_OUTPUT_ROOT",
    os.path.join(PREDICT_DATA_ROOT, "UNetPP_ResNet34_results", ORIGINAL_FOLDER),
)

# Save “predictions”
PRED_ROOT = os.path.join(OUT_ROOT, "Predictions_UNetPP")
PRED_NUC_DIR  = os.path.join(PRED_ROOT, "Nucleus")
PRED_NUCL_DIR = os.path.join(PRED_ROOT, "Nucleolus")

# Cellpose instances
CELLPOSE_MASK_DIR = os.path.join(OUT_ROOT, "Cellpose", "CellInstanceMasks")

# Final masks
FINAL_ROOT = os.path.join(OUT_ROOT, "FinalMasks")
FINAL_CELL_BIN_DIR     = os.path.join(FINAL_ROOT, "Cell_binary")
FINAL_CELL_OUTLINE_DIR = os.path.join(FINAL_ROOT, "Cell_outline")
FINAL_CYTO_BIN_DIR     = os.path.join(FINAL_ROOT, "Cytoplasm_binary")
FINAL_NUC_BIN_DIR      = os.path.join(FINAL_ROOT, "Nucleus_binary")
FINAL_NUCL_BIN_DIR     = os.path.join(FINAL_ROOT, "Nucleolus_binary")
FINAL_LIPID_BIN_DIR    = os.path.join(FINAL_ROOT, "Lipid_binary")

# Visualizations
VIS_ROOT = os.path.join(OUT_ROOT, "Visualizations2")
VIS_2PANEL = os.path.join(VIS_ROOT, "Original_vs_AllCombinedOverlay")

VIS_PANELS_ROOT   = os.path.join(VIS_ROOT, "Panels_Separate")
VIS_ORIG_DIR      = os.path.join(VIS_PANELS_ROOT, "Original")
VIS_CELL_DIR      = os.path.join(VIS_PANELS_ROOT, "Cell")
VIS_CYTO_DIR      = os.path.join(VIS_PANELS_ROOT, "Cytoplasm")
VIS_NUC_DIR       = os.path.join(VIS_PANELS_ROOT, "Nucleus")
VIS_NUCL_DIR      = os.path.join(VIS_PANELS_ROOT, "Nucleolus")
VIS_LIPID_DIR     = os.path.join(VIS_PANELS_ROOT, "Lipids")
VIS_COMBINED_DIR  = os.path.join(VIS_PANELS_ROOT, "Combined")

VIS_LIPID_ONLY_DIR = os.path.join(VIS_ROOT, "LipidOnly")
VIS_MERGE_ZOOM     = os.path.join(VIS_ROOT, "Merge_vs_ZoomedIn")

# ✅ NEW: Full prediction folder (full combined overlay image)
VIS_FULL_PRED_DIR  = os.path.join(VIS_ROOT, "FullPrediction")

SKIP_IF_EXISTS = True

# ===================== COLORS =====================
COLOR_CELL_FILL      = (0, 200, 200)     # teal
COLOR_CYTO_FILL      = (0, 200, 200)     # teal
COLOR_NUC_FILL       = (30, 91, 255)     # blue
COLOR_NUCL_FILL      = (255, 79, 216)    # magenta
COLOR_CELL_OUTLINE   = (255, 212, 0)     # yellow


# ===================== MODELS =====================
def build_plain_unet():
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    class Down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
        def forward(self, x): return self.net(x)

    class Up(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        def forward(self, x1, x2):
            x1 = self.up(x1)
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            x1 = nn.functional.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inc = DoubleConv(3, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 512)
            self.up1 = Up(1024, 256)
            self.up2 = Up(512, 128)
            self.up3 = Up(256, 64)
            self.up4 = Up(128, 64)
            self.outc = nn.Conv2d(64, 2, 1)
        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            return self.outc(x)

    return UNet()

def build_unetpp():
    import segmentation_models_pytorch as smp
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=2,
        activation=None
    )

def build_model_from_weights(state_dict):
    keys = list(state_dict.keys())
    if any(k.startswith("encoder.") for k in keys) or any(k.startswith("segmentation_head.") for k in keys):
        print("✅ Detected UnetPlusPlus weights")
        return build_unetpp()
    if any(k.startswith("inc.net.") for k in keys) and any(k.startswith("outc.") for k in keys):
        print("✅ Detected plain UNet weights")
        return build_plain_unet()
    print("⚠️ Unknown weights format — trying UnetPlusPlus first, else plain UNet.")
    try:
        return build_unetpp()
    except Exception:
        return build_plain_unet()


# ===================== HELPERS =====================
def mkdirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def list_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)

def resize_pad(img_pil, size, is_mask=False):
    w, h = img_pil.size
    scale = size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    interp = Image.NEAREST if is_mask else Image.BILINEAR
    img_resized = img_pil.resize((nw, nh), interp)

    canvas_mode = "L" if img_resized.mode == "L" else "RGB"
    canvas = Image.new(canvas_mode, (size, size), 0)
    left = (size - nw) // 2
    top = (size - nh) // 2
    canvas.paste(img_resized, (left, top))

    meta = {"orig": (w, h), "scaled": (nw, nh), "pad": (left, top)}
    return canvas, meta

def unpad_and_resize(mask_sq, meta):
    left, top = meta["pad"]
    nw, nh = meta["scaled"]
    w, h = meta["orig"]
    crop = mask_sq.crop((left, top, left + nw, top + nh))
    return crop.resize((w, h), Image.NEAREST)

def pil_to_torch(img_pil):
    arr = np.array(img_pil).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)

def save_binary_png(mask_bool, path):
    Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L").save(path)

def save_uint16_label(mask_int, path_tif):
    Image.fromarray(mask_int.astype(np.uint16), mode="I;16").save(path_tif)

def add_label(img_rgb, text):
    img = img_rgb.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", size=max(16, img.size[1] // 30))
    except OSError:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = font.getsize(text)

    x = (img.size[0] - tw) // 2
    y = 6
    pad = 6
    draw.rectangle((x - pad, y - pad, x + tw + pad, y + th + pad), fill=(0, 0, 0, 180))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    return Image.alpha_composite(img, overlay).convert("RGB")

def mask_to_rgba(mask_bool, color_rgb, alpha):
    h, w = mask_bool.shape
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[..., 0] = color_rgb[0]
    layer[..., 1] = color_rgb[1]
    layer[..., 2] = color_rgb[2]
    layer[..., 3] = mask_bool.astype(np.uint8) * alpha
    return Image.fromarray(layer, mode="RGBA")

def instance_outline_from_labels(lbl):
    m = lbl
    outline = np.zeros_like(m, dtype=bool)
    outline |= (m != np.roll(m,  1, axis=0)) & (m > 0)
    outline |= (m != np.roll(m, -1, axis=0)) & (m > 0)
    outline |= (m != np.roll(m,  1, axis=1)) & (m > 0)
    outline |= (m != np.roll(m, -1, axis=1)) & (m > 0)
    outline[0, :] = False
    outline[-1, :] = False
    outline[:, 0] = False
    outline[:, -1] = False
    return outline

def dilate_bool(mask_bool, k=3):
    m = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
    m = m.filter(ImageFilter.MaxFilter(k))
    return (np.array(m) > 127)


# ===================== LIPID DETECTION =====================
def rgb_to_hsv_np(rgb_u8):
    rgb = rgb_u8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc

    s = np.zeros_like(maxc)
    s[maxc > 1e-6] = delta[maxc > 1e-6] / (maxc[maxc > 1e-6] + 1e-6)

    h = np.zeros_like(maxc)
    mask = delta > 1e-6

    r_m = r[mask]; g_m = g[mask]; b_m = b[mask]
    max_m = maxc[mask]; d_m = delta[mask] + 1e-6

    rc = (max_m - r_m) / d_m
    gc = (max_m - g_m) / d_m
    bc = (max_m - b_m) / d_m

    r_is_max = (r_m == max_m)
    g_is_max = (g_m == max_m)
    b_is_max = (b_m == max_m)

    h_tmp = np.zeros_like(rc)
    h_tmp[r_is_max] = (bc - gc)[r_is_max]
    h_tmp[g_is_max] = 2.0 + (rc - bc)[g_is_max]
    h_tmp[b_is_max] = 4.0 + (gc - rc)[b_is_max]

    h_val = (h_tmp / 6.0) % 1.0
    h[mask] = h_val
    return h, s, v

def extract_orange_lipid_mask(orig_pil,
                              h_lo=LIPID_H_LO, h_hi=LIPID_H_HI,
                              s_min=LIPID_S_MIN, v_min=LIPID_V_MIN,
                              cleanup=LIPID_CLEANUP,
                              yellow_score_thr=LIPID_YELLOW_SCORE_THR,
                              min_rg=LIPID_MIN_RG):
    rgb = np.array(orig_pil.convert("RGB"))
    h, s, v = rgb_to_hsv_np(rgb)

    r = rgb[..., 0].astype(np.float32) / 255.0
    g = rgb[..., 1].astype(np.float32) / 255.0
    b = rgb[..., 2].astype(np.float32) / 255.0

    hsv_gate = (h >= h_lo) & (h <= h_hi) & (s >= s_min) & (v >= v_min)
    yellow_score = ((r + g) * 0.5) - b
    score_gate = (yellow_score >= yellow_score_thr) & (r >= min_rg) & (g >= min_rg)

    mask = hsv_gate & score_gate

    if cleanup:
        # ✅ OPENING (Min then Max) to remove noise / shrink instead of growing blobs
        m = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
        mask = (np.array(m) > 127)

    return mask

def apply_mask_black_background(orig_pil, mask_bool):
    rgb = np.array(orig_pil.convert("RGB"))
    out = np.zeros_like(rgb)
    out[mask_bool] = rgb[mask_bool]
    return Image.fromarray(out, mode="RGB")


# ===================== OVERLAYS =====================
def make_structure_overlay(orig_rgb, structure_mask, cell_labels, cell_union,
                           fill_color, fill_alpha,
                           show_cell_tint=True, cell_tint_alpha=20):
    base = orig_rgb.convert("RGBA")
    H, W = np.array(orig_rgb).shape[:2]
    cell_outline = instance_outline_from_labels(cell_labels)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    if show_cell_tint:
        overlay = Image.alpha_composite(overlay, mask_to_rgba(cell_union, COLOR_CELL_FILL, cell_tint_alpha))

    overlay = Image.alpha_composite(overlay, mask_to_rgba(structure_mask, fill_color, fill_alpha))
    overlay = Image.alpha_composite(overlay, mask_to_rgba(cell_outline, COLOR_CELL_OUTLINE, 240))
    return Image.alpha_composite(base, overlay).convert("RGB")

def make_all_combined_overlay(orig_rgb, cell_labels, cell_union, cyto, nuc, nucl, lipid=None):
    base = orig_rgb.convert("RGBA")
    H, W = np.array(orig_rgb).shape[:2]

    cell_outline = instance_outline_from_labels(cell_labels)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    overlay = Image.alpha_composite(overlay, mask_to_rgba(cell_union, COLOR_CELL_FILL, 35))
    overlay = Image.alpha_composite(overlay, mask_to_rgba(cyto,      COLOR_CYTO_FILL, 130))
    overlay = Image.alpha_composite(overlay, mask_to_rgba(nuc,       COLOR_NUC_FILL, 150))
    overlay = Image.alpha_composite(overlay, mask_to_rgba(nucl,      COLOR_NUCL_FILL, 190))

    if lipid is not None:
        lipid_disp = dilate_bool(lipid, k=LIPID_DISPLAY_DILATE_K)
        overlay = Image.alpha_composite(overlay, mask_to_rgba(lipid_disp, COLOR_LIPID_OVERLAY, LIPID_ALPHA_COMBINED))

        if LIPID_SPARKLE_EDGE:
            lip_d = dilate_bool(lipid_disp, k=LIPID_EDGE_DILATE_K)
            lip_edge = lip_d & (~lipid_disp)
            overlay = Image.alpha_composite(overlay, mask_to_rgba(lip_edge, (255, 255, 255), LIPID_EDGE_ALPHA))

    overlay = Image.alpha_composite(overlay, mask_to_rgba(cell_outline, COLOR_CELL_OUTLINE, 240))
    return Image.alpha_composite(base, overlay).convert("RGB")


# ===================== ZOOM HELPERS =====================
def mask_bbox(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1, y1)

def expand_bbox(bbox, W, H, pad, min_size):
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    bw = (x1 - x0 + 1)
    bh = (y1 - y0 + 1)
    size = max(bw, bh, min_size) + 2 * pad

    half = size // 2
    nx0 = max(0, cx - half)
    ny0 = max(0, cy - half)
    nx1 = min(W - 1, cx + half)
    ny1 = min(H - 1, cy + half)

    if nx1 <= nx0: nx1 = min(W - 1, nx0 + 1)
    if ny1 <= ny0: ny1 = min(H - 1, ny0 + 1)
    return (nx0, ny0, nx1, ny1)

def draw_box(img_rgb, bbox, color=(255, 0, 0), width=3):
    out = img_rgb.copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = bbox
    for i in range(width):
        draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=color)
    return out


# ===================== SKIP CHECK =====================
def all_outputs_exist(base):
    out_2panel = os.path.join(VIS_2PANEL, base + "_orig_vs_all_overlay.png")

    out_orig  = os.path.join(VIS_ORIG_DIR,     base + "_original.png")
    out_cell  = os.path.join(VIS_CELL_DIR,     base + "_cell.png")
    out_cyto  = os.path.join(VIS_CYTO_DIR,     base + "_cytoplasm.png")
    out_nuc   = os.path.join(VIS_NUC_DIR,      base + "_nucleus.png")
    out_nucl  = os.path.join(VIS_NUCL_DIR,     base + "_nucleolus.png")
    out_lipid = os.path.join(VIS_LIPID_DIR,    base + "_lipids.png")
    out_comb  = os.path.join(VIS_COMBINED_DIR, base + "_combined.png")
    out_lipid_only = os.path.join(VIS_LIPID_ONLY_DIR, base + "_lipid_only.png")

    # ✅ NEW: full prediction image file
    out_full_pred = os.path.join(VIS_FULL_PRED_DIR, base + "_full_predict.png")

    cell_bin   = os.path.join(FINAL_CELL_BIN_DIR,     base + "_cell.png")
    cell_out   = os.path.join(FINAL_CELL_OUTLINE_DIR, base + "_cell_outline.png")
    cyto_bin   = os.path.join(FINAL_CYTO_BIN_DIR,     base + "_cytoplasm.png")
    nuc_bin    = os.path.join(FINAL_NUC_BIN_DIR,      base + "_nucleus.png")
    nucl_bin   = os.path.join(FINAL_NUCL_BIN_DIR,     base + "_nucleolus.png")
    lipid_bin  = os.path.join(FINAL_LIPID_BIN_DIR,    base + "_lipid.png")

    inst_tif   = os.path.join(CELLPOSE_MASK_DIR,      base + "_cell_instances.tif")
    pred_nuc   = os.path.join(PRED_NUC_DIR,           base + "_nucleus.png")
    pred_nucl  = os.path.join(PRED_NUCL_DIR,          base + "_nucleolus.png")

    need = [
        out_2panel,
        out_orig, out_cell, out_cyto, out_nuc, out_nucl, out_lipid, out_comb,
        out_lipid_only,
        out_full_pred,  # ✅ NEW
        cell_bin, cell_out, cyto_bin, nuc_bin, nucl_bin, lipid_bin,
        inst_tif, pred_nuc, pred_nucl
    ]
    if MAKE_ZOOM:
        need.append(os.path.join(VIS_MERGE_ZOOM, base + "_merge_vs_zoom.png"))
    return all(os.path.exists(p) for p in need)


# ===================== PREDICT =====================
def unet_predict_masks(model, device, orig_pil):
    sq, meta = resize_pad(orig_pil, UNET_IMG_SIZE, is_mask=False)
    x = pil_to_torch(sq).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0]  # 2 x S x S

    nuc_sq = Image.fromarray(((probs[0].cpu().numpy() > UNET_THRESHOLD).astype(np.uint8) * 255), "L")
    nucl_sq = Image.fromarray(((probs[1].cpu().numpy() > UNET_THRESHOLD).astype(np.uint8) * 255), "L")

    nuc_back = unpad_and_resize(nuc_sq, meta)
    nucl_back = unpad_and_resize(nucl_sq, meta)

    nuc_bool = (np.array(nuc_back) > MASK_THRESHOLD)
    nucl_bool = (np.array(nucl_back) > MASK_THRESHOLD)
    return nuc_bool, nucl_bool


# ===================== MAIN =====================
def main():
    mkdirs(
        OUT_ROOT,
        PRED_NUC_DIR, PRED_NUCL_DIR,
        CELLPOSE_MASK_DIR,
        FINAL_CELL_BIN_DIR, FINAL_CELL_OUTLINE_DIR,
        FINAL_CYTO_BIN_DIR, FINAL_NUC_BIN_DIR, FINAL_NUCL_BIN_DIR,
        FINAL_LIPID_BIN_DIR,
        VIS_2PANEL,
        VIS_ORIG_DIR, VIS_CELL_DIR, VIS_CYTO_DIR, VIS_NUC_DIR, VIS_NUCL_DIR, VIS_LIPID_DIR, VIS_COMBINED_DIR,
        VIS_LIPID_ONLY_DIR,
        VIS_MERGE_ZOOM,
        VIS_FULL_PRED_DIR,  # ✅ NEW
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = (device.type == "cuda")
    print("Using device:", device)
    if use_gpu:
        print("✅ GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ CPU only (will be slower).")

    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found:", MODEL_PATH)
        sys.exit(1)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model = build_model_from_weights(state_dict).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Loaded weights:", MODEL_PATH)

    print(f"Initializing Cellpose model_type='{CELLPOSE_MODEL_TYPE}' (gpu={use_gpu}) ...")
    try:
        cp_model = cp_models.CellposeModel(gpu=use_gpu, model_type=CELLPOSE_MODEL_TYPE)
    except Exception as e:
        print("⚠️ Failed to load", CELLPOSE_MODEL_TYPE, "->", e)
        print("Falling back to 'cyto3' ...")
        cp_model = cp_models.CellposeModel(gpu=use_gpu, model_type="cyto3")

    img_dir = os.path.join(PREDICT_DATA_ROOT, ORIGINAL_FOLDER)
    img_paths = list_images(img_dir)
    print(f"Found {len(img_paths)} images in: {img_dir}")
    if not img_paths:
        return

    for path in img_paths:
        name = os.path.basename(path)
        base = os.path.splitext(name)[0]
        print("\n=== Processing:", name, "===")

        if SKIP_IF_EXISTS and all_outputs_exist(base):
            print("Skipping (all outputs exist)")
            continue

        orig_pil = Image.open(path).convert("RGB")
        W, H = orig_pil.size

        # 1) UNet++ nucleus + nucleolus
        nuc_bool, nucl_bool = unet_predict_masks(model, device, orig_pil)

        # 2) Cellpose instance segmentation
        img_np = cp_io.imread(path)
        cell_labels, _, _ = cp_model.eval(
            img_np,
            batch_size=CELLPOSE_BATCH_SIZE,
            channels=CELLPOSE_CHANNELS,
            diameter=CELLPOSE_DIAMETER,
            flow_threshold=CELLPOSE_FLOW_THRESHOLD,
            cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD,
        )
        cell_labels = cell_labels.astype(np.int32)

        # 3) Hierarchy
        cell_union   = (cell_labels > 0)
        nuc_in_cells = nuc_bool & cell_union
        nucl_in_nuc  = nucl_bool & nuc_in_cells
        cyto         = cell_union & (~nuc_in_cells)
        cell_outline = instance_outline_from_labels(cell_labels)

        # 4) Lipids from color (TUNED DOWN)
        lipid_mask = extract_orange_lipid_mask(orig_pil)

        # ✅ KEY FIX: restrict to CYTOPLASM (not whole cell)
        lipid_in_cells = lipid_mask & cyto

        # Save "preds"
        save_binary_png(nuc_in_cells, os.path.join(PRED_NUC_DIR,  base + "_nucleus.png"))
        save_binary_png(nucl_in_nuc,  os.path.join(PRED_NUCL_DIR, base + "_nucleolus.png"))

        # Save instance labels
        save_uint16_label(cell_labels, os.path.join(CELLPOSE_MASK_DIR, base + "_cell_instances.tif"))

        # Save FINAL masks
        save_binary_png(cell_union,      os.path.join(FINAL_CELL_BIN_DIR,     base + "_cell.png"))
        save_binary_png(cell_outline,    os.path.join(FINAL_CELL_OUTLINE_DIR, base + "_cell_outline.png"))
        save_binary_png(cyto,            os.path.join(FINAL_CYTO_BIN_DIR,     base + "_cytoplasm.png"))
        save_binary_png(nuc_in_cells,    os.path.join(FINAL_NUC_BIN_DIR,      base + "_nucleus.png"))
        save_binary_png(nucl_in_nuc,     os.path.join(FINAL_NUCL_BIN_DIR,     base + "_nucleolus.png"))
        save_binary_png(lipid_in_cells,  os.path.join(FINAL_LIPID_BIN_DIR,    base + "_lipid.png"))

        # Lipid-only RGB (black background)
        apply_mask_black_background(orig_pil, lipid_in_cells).save(
            os.path.join(VIS_LIPID_ONLY_DIR, base + "_lipid_only.png")
        )

        # 5) overlays
        overlay_all = make_all_combined_overlay(
            orig_rgb=orig_pil,
            cell_labels=cell_labels,
            cell_union=cell_union,
            cyto=cyto,
            nuc=nuc_in_cells,
            nucl=nucl_in_nuc,
            lipid=lipid_in_cells
        )

        # ✅ NEW: save full prediction image (combined overlay only)
        out_full_pred = os.path.join(VIS_FULL_PRED_DIR, base + "_full_predict.png")
        overlay_all.save(out_full_pred)
        print("Saved:", out_full_pred)

        overlay_cell = make_structure_overlay(
            orig_pil, cell_union, cell_labels, cell_union,
            fill_color=COLOR_CELL_FILL, fill_alpha=70,
            show_cell_tint=False
        )
        overlay_cyto = make_structure_overlay(
            orig_pil, cyto, cell_labels, cell_union,
            fill_color=COLOR_CYTO_FILL, fill_alpha=130,
            show_cell_tint=True
        )
        overlay_nuc = make_structure_overlay(
            orig_pil, nuc_in_cells, cell_labels, cell_union,
            fill_color=COLOR_NUC_FILL, fill_alpha=150,
            show_cell_tint=True
        )
        overlay_nucl = make_structure_overlay(
            orig_pil, nucl_in_nuc, cell_labels, cell_union,
            fill_color=COLOR_NUCL_FILL, fill_alpha=190,
            show_cell_tint=True
        )
        overlay_lipid = make_structure_overlay(
            orig_pil, lipid_in_cells, cell_labels, cell_union,
            fill_color=(255, 140, 0), fill_alpha=180,
            show_cell_tint=True
        )

        # Save separate panels
        add_label(orig_pil.copy(),        "Original").save(os.path.join(VIS_ORIG_DIR,     base + "_original.png"))
        add_label(overlay_cell.copy(),    "Cell").save(os.path.join(VIS_CELL_DIR,        base + "_cell.png"))
        add_label(overlay_cyto.copy(),    "Cytoplasm").save(os.path.join(VIS_CYTO_DIR,   base + "_cytoplasm.png"))
        add_label(overlay_nuc.copy(),     "Nucleus").save(os.path.join(VIS_NUC_DIR,      base + "_nucleus.png"))
        add_label(overlay_nucl.copy(),    "Nucleolus").save(os.path.join(VIS_NUCL_DIR,   base + "_nucleolus.png"))
        add_label(overlay_lipid.copy(),   "Lipids").save(os.path.join(VIS_LIPID_DIR,     base + "_lipids.png"))
        add_label(overlay_all.copy(),     "Combined").save(os.path.join(VIS_COMBINED_DIR, base + "_combined.png"))

        # 2-panel Original | Combined
        p1 = add_label(orig_pil.copy(), "Original")
        p2 = add_label(overlay_all.copy(), "Combined Overlay (lipids tuned down)")
        two = Image.new("RGB", (W * 2, H))
        two.paste(p1, (0, 0))
        two.paste(p2, (W, 0))
        out_2panel = os.path.join(VIS_2PANEL, base + "_orig_vs_all_overlay.png")
        two.save(out_2panel)
        print("Saved:", out_2panel)

        # Optional Merge vs Zoom (ROI lipid->nucl->nuc->cell)
        if MAKE_ZOOM:
            bbox = mask_bbox(lipid_in_cells)
            if bbox is None:
                bbox = mask_bbox(nucl_in_nuc)
            if bbox is None:
                bbox = mask_bbox(nuc_in_cells)
            if bbox is None:
                bbox = mask_bbox(cell_union)

            if bbox is not None:
                roi = expand_bbox(bbox, W, H, pad=ZOOM_PAD, min_size=ZOOM_MIN_SIZE)
                merge_box = draw_box(overlay_all, roi, color=ZOOM_BOX_COLOR, width=3)

                x0, y0, x1, y1 = roi
                crop = overlay_all.crop((x0, y0, x1 + 1, y1 + 1))
                crop_zoom = crop.resize((int(crop.size[0] * ZOOM_SCALE), int(crop.size[1] * ZOOM_SCALE)), Image.NEAREST)

                zoom_panel = Image.new("RGB", (W, H), (0, 0, 0))
                z = crop_zoom
                if z.size[0] > W or z.size[1] > H:
                    z = z.resize((min(W, z.size[0]), min(H, z.size[1])), Image.NEAREST)
                zx = (W - z.size[0]) // 2
                zy = (H - z.size[1]) // 2
                zoom_panel.paste(z, (zx, zy))

                m1 = add_label(merge_box, "Merge (ROI box)")
                m2 = add_label(zoom_panel, "Zoomed-in")

                mz = Image.new("RGB", (W * 2, H))
                mz.paste(m1, (0, 0))
                mz.paste(m2, (W, 0))

                out_zoom = os.path.join(VIS_MERGE_ZOOM, base + "_merge_vs_zoom.png")
                mz.save(out_zoom)
                print("Saved:", out_zoom)

        cov_nuc = 100.0 * nuc_in_cells.mean()
        cov_nucl = 100.0 * nucl_in_nuc.mean()
        cov_lip = 100.0 * lipid_in_cells.mean()
        n_cells = int(cell_labels.max())
        print(f"cells={n_cells} | nuc%={cov_nuc:.2f} | nucl%={cov_nucl:.2f} | lipid%={cov_lip:.3f}")

    print("\n✅ DONE")
    print("FinalMasks folder:", FINAL_ROOT)
    print("2-panel outputs:", VIS_2PANEL)
    print("Separate panels root:", VIS_PANELS_ROOT)
    print("Lipid-only:", VIS_LIPID_ONLY_DIR)
    print("✅ Full prediction folder:", VIS_FULL_PRED_DIR)
    if MAKE_ZOOM:
        print("Merge vs Zoom:", VIS_MERGE_ZOOM)


if __name__ == "__main__":
    main()

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from scipy import ndimage
from scipy.stats import entropy as sp_entropy
from skimage.feature import graycomatrix, graycoprops
from torchvision.models import efficientnet_b0
import segmentation_models_pytorch as smp

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent

SEG_CKPT = _ROOT / "Segmentation_Dataset/Segmentation_Branch_Outputs/05_training_runs/checkpoints/best_efficientnetb0_unet.pth"
CLS_CKPT = _ROOT / "diabetes_pipeline_outputs/05_segmented_training_lighting_robust/best_segmented_lighting_robust_model.pth"
HYBRID_MODEL_PATH = _ROOT / "diabetes_pipeline_outputs/11_hybrid_probability_fusion/11_hybrid_model.joblib"
SCALER_PATH = _ROOT / "diabetes_pipeline_outputs/11_hybrid_probability_fusion/11_hybrid_scaler.joblib"
FEAT_COLS_PATH = _ROOT / "diabetes_pipeline_outputs/10_hybrid_image_features/10_model_feature_columns.json"
HYBRID_SUMMARY_PATH = _ROOT / "diabetes_pipeline_outputs/11_hybrid_probability_fusion/11_hybrid_best_model_summary.json"

# ── Constants ─────────────────────────────────────────────────────────────────
SEG_SIZE = 384
CLS_SIZE = 224
SEG_THRESHOLD = 0.5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MIN_TONGUE_PIXELS = 100
ERO_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Models container ──────────────────────────────────────────────────────────
@dataclass
class Models:
    seg: Any
    cls: Any
    hybrid: Any
    scaler: Any
    feature_cols: list
    threshold: float


# ── Load all models (called once at startup) ──────────────────────────────────
def load_models() -> Models:
    seg = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    ckpt = torch.load(SEG_CKPT, map_location=DEVICE)
    seg.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    seg.to(DEVICE).eval()

    cls = efficientnet_b0(weights=None)
    cls.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(cls.classifier[1].in_features, 1),
    )
    ckpt = torch.load(CLS_CKPT, map_location=DEVICE)
    cls.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    cls.to(DEVICE).eval()

    hybrid = joblib.load(HYBRID_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEAT_COLS_PATH) as f:
        feature_cols = json.load(f)

    threshold = 0.3
    if HYBRID_SUMMARY_PATH.exists():
        with open(HYBRID_SUMMARY_PATH) as f:
            summary = json.load(f)
        threshold = float(summary.get("best_threshold", 0.3))

    return Models(
        seg=seg,
        cls=cls,
        hybrid=hybrid,
        scaler=scaler,
        feature_cols=feature_cols,
        threshold=threshold,
    )


# ── Transforms ────────────────────────────────────────────────────────────────
def _pad_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), (0, 0, 0))
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


_seg_tf = T.Compose([
    T.Lambda(_pad_square),
    T.Resize((SEG_SIZE, SEG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

_cls_tf = T.Compose([
    T.Lambda(_pad_square),
    T.Resize((CLS_SIZE, CLS_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# ── Preprocessing ─────────────────────────────────────────────────────────────
def _clahe(img_pil: Image.Image) -> Image.Image:
    img_rgb = np.array(img_pil.convert("RGB"))
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB))


# ── Segmentation ──────────────────────────────────────────────────────────────
def _segment(img_pil: Image.Image, seg_model: Any) -> np.ndarray:
    w, h = img_pil.size
    inp = _seg_tf(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(seg_model(inp)).cpu().numpy()[0, 0]
    mask = (prob > SEG_THRESHOLD).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    labeled, num = ndimage.label(mask)
    if num > 0:
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        mask = (labeled == np.argmax(sizes) + 1).astype(np.uint8)
    return mask


def _crop_masked(img_rgb: np.ndarray, mask: np.ndarray):
    bb = img_rgb * mask[:, :, np.newaxis]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return bb, (0, img_rgb.shape[0], 0, img_rgb.shape[1])
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return bb[y1:y2 + 1, x1:x2 + 1].copy(), (int(y1), int(y2 + 1), int(x1), int(x2 + 1))


# ── QC ────────────────────────────────────────────────────────────────────────
def _qc(mask: np.ndarray, shape) -> dict:
    h, w = shape[:2]
    fg = int(mask.sum())
    qc = {"mask_fg_ratio": round(fg / (h * w), 6), "failed_no_mask": fg == 0}
    if fg > 0:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        ec = sum([y1 == 0, y2 == h - 1, x1 == 0, x2 == w - 1])
        qc["edge_touch"] = ec >= 3
        qc["large_mask"] = qc["mask_fg_ratio"] > 0.85
        qc["small_mask"] = qc["mask_fg_ratio"] < 0.02
        qc["tiny_bbox"] = ((x2 - x1 + 1) * (y2 - y1 + 1)) / (h * w) < 0.02
    else:
        qc["edge_touch"] = qc["large_mask"] = False
        qc["small_mask"] = qc["tiny_bbox"] = True
    return qc


def _assess_retake(qc: dict):
    if qc["failed_no_mask"] or qc["small_mask"] or qc["tiny_bbox"]:
        return True, False, "Please retake with tongue centered and fully visible."
    if qc["large_mask"]:
        return False, True, "Prediction generated, but review recommended (large mask)."
    if qc.get("edge_touch", False):
        return False, False, "Prediction generated. Segmentation touched edges; review recommended."
    return False, False, "Image accepted."


# ── CNN predict ───────────────────────────────────────────────────────────────
def _cnn_predict(crop_masked_pil: Image.Image, cls_model: Any) -> float:
    clahe_img = _clahe(crop_masked_pil)
    inp = _cls_tf(clahe_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = cls_model(inp)
    return float(torch.sigmoid(logit).item())


# ── Feature extraction ────────────────────────────────────────────────────────
def _color_ratio(hsv_px: np.ndarray, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi) -> float:
    h, s, v = hsv_px[:, 0], hsv_px[:, 1], hsv_px[:, 2]
    return float(
        ((h >= h_lo) & (h <= h_hi) & (s >= s_lo) & (s <= s_hi) & (v >= v_lo) & (v <= v_hi)).sum()
        / max(len(h), 1)
    )


def _extract_features(img_rgb: np.ndarray, mask: np.ndarray) -> Optional[dict]:
    img_rgb = np.array(_clahe(Image.fromarray(img_rgb)))
    eroded = cv2.erode(mask, ERO_KERNEL, iterations=1)
    if eroded.sum() < MIN_TONGUE_PIXELS:
        eroded = mask
    tongue_px = img_rgb[mask == 1]
    if len(tongue_px) < MIN_TONGUE_PIXELS:
        return None

    feat = {}

    for i, ch in enumerate(["r", "g", "b"]):
        feat[f"rgb_mean_{ch}"] = float(tongue_px[:, i].mean())
        feat[f"rgb_std_{ch}"] = float(tongue_px[:, i].std())

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hsv_px = hsv[mask == 1]
    for i, ch in enumerate(["h", "s", "v"]):
        feat[f"hsv_mean_{ch}"] = float(hsv_px[:, i].mean())
        feat[f"hsv_std_{ch}"] = float(hsv_px[:, i].std())

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab_px = lab[mask == 1]
    for i, ch in enumerate(["l", "a", "b"]):
        feat[f"lab_mean_{ch}"] = float(lab_px[:, i].mean())
        feat[f"lab_std_{ch}"] = float(lab_px[:, i].std())

    feat["pink_red_ratio"] = (
        _color_ratio(hsv_px, 0, 15, 30, 255, 50, 255)
        + _color_ratio(hsv_px, 160, 180, 30, 255, 50, 255)
    )
    feat["white_coating_ratio"] = _color_ratio(hsv_px, 0, 180, 0, 40, 160, 255)
    feat["yellow_ratio"] = _color_ratio(hsv_px, 15, 35, 30, 255, 100, 255)
    feat["dark_tongue_pixel_ratio"] = float((hsv_px[:, 2] < 60).mean())
    feat["bright_tongue_pixel_ratio"] = float((hsv_px[:, 2] > 220).mean())
    feat["saturation_mean"] = float(hsv_px[:, 1].mean())
    feat["saturation_std"] = float(hsv_px[:, 1].std())
    feat["value_mean"] = float(hsv_px[:, 2].mean())
    feat["value_std"] = float(hsv_px[:, 2].std())

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    inner_px = gray[eroded == 1]
    feat["laplacian_variance_inner"] = float(
        cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var()
    )
    hist, _ = np.histogram(inner_px, bins=64, range=(0, 256), density=True)
    feat["entropy_inner"] = float(sp_entropy(hist + 1e-12))

    iy, ix = np.where(eroded == 1)
    patch = gray[iy.min():iy.max() + 1, ix.min():ix.max() + 1]
    patch_q = (patch // 4).astype(np.uint8)
    try:
        glcm = graycomatrix(patch_q, [1], [0], 64, symmetric=True, normed=True)
        feat["glcm_contrast_inner"] = float(graycoprops(glcm, "contrast")[0, 0])
        feat["glcm_homogeneity_inner"] = float(graycoprops(glcm, "homogeneity")[0, 0])
        feat["glcm_energy_inner"] = float(graycoprops(glcm, "energy")[0, 0])
        feat["glcm_correlation_inner"] = float(graycoprops(glcm, "correlation")[0, 0])
    except Exception:
        for k in ["glcm_contrast_inner", "glcm_homogeneity_inner", "glcm_energy_inner", "glcm_correlation_inner"]:
            feat[k] = 0.0

    return feat


# ── Risk band ─────────────────────────────────────────────────────────────────
def _risk_band(prob: float, threshold: float) -> str:
    if prob < 0.20:
        return "Low image-based risk"
    if prob < threshold:
        return "Borderline / uncertain"
    if prob < 0.70:
        return "Elevated image-based risk"
    return "High image-based risk"


# ── Public inference entry point ──────────────────────────────────────────────
def run_inference(img_pil: Image.Image, models: Models) -> dict:
    img_rgb = np.array(img_pil)
    mask = _segment(img_pil, models.seg)
    crop_masked, _ = _crop_masked(img_rgb, mask)
    qc = _qc(mask, img_rgb.shape)
    retake_req, retake_rec, message = _assess_retake(qc)

    base = {
        "retake_required": retake_req,
        "retake_recommended": retake_rec,
        "message": message,
        "mask_fg_ratio": qc["mask_fg_ratio"],
        "threshold": models.threshold,
        "cnn_probability": None,
        "hybrid_probability": None,
        "predicted_class": None,
        "risk_band": None,
    }

    if retake_req:
        return base

    cnn_prob = _cnn_predict(Image.fromarray(crop_masked), models.cls)
    base["cnn_probability"] = round(cnn_prob, 6)

    features = _extract_features(img_rgb, mask)
    if features is None:
        base["retake_required"] = True
        base["message"] = "Feature extraction failed. Please retake."
        return base

    hybrid_input = [cnn_prob] + [features.get(f, 0.0) for f in models.feature_cols]
    hybrid_input_scaled = models.scaler.transform([hybrid_input])
    hybrid_prob = float(models.hybrid.predict_proba(hybrid_input_scaled)[0, 1])
    pred_class = "diabetes" if hybrid_prob >= models.threshold else "non_diabetes"

    base["hybrid_probability"] = round(hybrid_prob, 6)
    base["predicted_class"] = pred_class
    base["risk_band"] = _risk_band(hybrid_prob, models.threshold)

    return base

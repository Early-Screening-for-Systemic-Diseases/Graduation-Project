"""
preprocessing_utils.py — Shared preprocessing functions for the skin lesion pipeline.

Contains:
1. detect_and_crop_vignette() — OpenCV-based, used by 08/08b for one-time cropping
2. preprocess_image_with_vignette_crop() — OpenCV-based, used by 08 for validation
3. preprocess_image_tf() — Pure TF, used by 10/11/12 for training from pre-cropped images
"""

import os
import numpy as np
import cv2

# Vignette constants
VIGNETTE_DARK_THRESHOLD = 40
VIGNETTE_MORPH_KERNEL_SIZE = 15
VIGNETTE_OCCUPANCY_THRESHOLD = 0.65
VIGNETTE_SAFETY_PADDING_PCT = 0.03
VIGNETTE_MIN_VALID_FRACTION = 0.25
VIGNETTE_MAX_VALID_FRACTION = 0.92
VIGNETTE_MIN_CROP_FRACTION = 0.35


def detect_and_crop_vignette(img_bgr,
                              dark_thresh=VIGNETTE_DARK_THRESHOLD,
                              morph_k=VIGNETTE_MORPH_KERNEL_SIZE,
                              occupancy_thresh=VIGNETTE_OCCUPANCY_THRESHOLD,
                              safety_pad_pct=VIGNETTE_SAFETY_PADDING_PCT,
                              min_valid_frac=VIGNETTE_MIN_VALID_FRACTION,
                              max_valid_frac=VIGNETTE_MAX_VALID_FRACTION,
                              min_crop_frac=VIGNETTE_MIN_CROP_FRACTION):
    h, w = img_bgr.shape[:2]
    total_pixels = h * w
    debug = {"valid_fraction": 0.0, "skip_reason": None}

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mask = (v_channel > dark_thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        debug["skip_reason"] = "no_valid_component"
        return img_bgr, False, None, mask, debug

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = ((labels == largest_label) * 255).astype(np.uint8)

    valid_fraction = np.count_nonzero(mask) / total_pixels
    debug["valid_fraction"] = round(valid_fraction, 4)

    if valid_fraction < min_valid_frac:
        debug["skip_reason"] = f"valid_fraction_too_low ({valid_fraction:.3f} < {min_valid_frac})"
        return img_bgr, False, None, mask, debug
    if valid_fraction > max_valid_frac:
        debug["skip_reason"] = f"valid_fraction_too_high ({valid_fraction:.3f} > {max_valid_frac})"
        return img_bgr, False, None, mask, debug

    row_occupancy = np.count_nonzero(mask, axis=1) / w
    col_occupancy = np.count_nonzero(mask, axis=0) / h

    valid_rows = np.where(row_occupancy >= occupancy_thresh)[0]
    valid_cols = np.where(col_occupancy >= occupancy_thresh)[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        debug["skip_reason"] = "no_rows_cols_above_occupancy"
        return img_bgr, False, None, mask, debug

    y_start, y_end = int(valid_rows[0]), int(valid_rows[-1]) + 1
    x_start, x_end = int(valid_cols[0]), int(valid_cols[-1]) + 1

    pad_y = int(h * safety_pad_pct)
    pad_x = int(w * safety_pad_pct)
    y_start = min(y_start + pad_y, h - 1)
    y_end = max(y_end - pad_y, y_start + 1)
    x_start = min(x_start + pad_x, w - 1)
    x_end = max(x_end - pad_x, x_start + 1)

    crop_area = (y_end - y_start) * (x_end - x_start)
    crop_frac = crop_area / total_pixels
    if crop_frac < min_crop_frac:
        debug["skip_reason"] = f"crop_too_small ({crop_frac:.3f} < {min_crop_frac})"
        return img_bgr, False, None, mask, debug

    crop_box = (y_start, y_end, x_start, x_end)
    cropped = img_bgr[y_start:y_end, x_start:x_end].copy()
    debug["skip_reason"] = None

    return cropped, True, crop_box, mask, debug


def preprocess_image_with_vignette_crop(image_path_str, target_size=224):
    """OpenCV-based full preprocessing. Used by Section 08 for validation only."""
    img_bgr = cv2.imread(image_path_str)
    if img_bgr is None:
        return np.zeros((target_size, target_size, 3), dtype=np.float32)

    img_bgr, _, _, _, _ = detect_and_crop_vignette(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    h, w = img_rgb.shape[:2]
    scale = target_size / min(h, w)
    new_h = int(np.ceil(h * scale))
    new_w = int(np.ceil(w * scale))
    img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    start_y = (new_h - target_size) // 2
    start_x = (new_w - target_size) // 2
    img_rgb = img_rgb[start_y:start_y + target_size, start_x:start_x + target_size]

    img_rgb = img_rgb / 255.0
    return img_rgb.astype(np.float32)
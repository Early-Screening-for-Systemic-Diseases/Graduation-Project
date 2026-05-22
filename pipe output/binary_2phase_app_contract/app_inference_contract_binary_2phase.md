# App Inference Contract — Binary Cancer-Risk Model

**Version:** binary_2phase_v1
**Date:** 2026-05-22
**Model:** EfficientNetB0 (binary head, two-phase transfer learning)
**Checkpoint:** `best_model_binary_2phase.pt`

---

## Purpose

Provide a standardised interface for mobile/web apps to query the binary
cancer-risk screening model. The model outputs a **cancer-risk probability**
and a **risk level** for a single dermoscopic (or similar) skin lesion image.

This is a **screening support tool only**. It is not a clinical diagnosis.

---

## Input Format

| Field        | Requirement                                          |
|------------- |------------------------------------------------------|
| Image type   | RGB JPEG or PNG                                      |
| Recommended  | Dermoscopic image (macroscopic may be out-of-scope)  |
| Resize       | Resize to 256 px on shortest side, CenterCrop 224   |
| Normalization| ImageNet mean/std (see config JSON)                  |

**Inference transform (exact):**
```
Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)
```

---

## Output Schema

| Field                                 | Type    | Description                          |
|---------------------------------------|---------|--------------------------------------|
| `cancer_risk_probability`             | float   | Model softmax prob for cancer_risk   |
| `cancer_risk_percent`                 | float   | Probability × 100                    |
| `risk_level`                          | string  | lower / moderate / higher            |
| `binary_prediction_at_youden_threshold` | int   | 1 = model-positive, 0 = model-negative |
| `recommended_action`                  | string  | Human-readable guidance              |
| `disclaimer`                          | string  | Medical disclaimer (always included) |
| `model_version`                       | string  | Model identifier                     |
| `threshold_version`                   | string  | Threshold identifier                 |

---

## Risk Thresholds

| Boundary            | Value  | Meaning                                      |
|---------------------|--------|----------------------------------------------|
| lower → moderate    | 0.1700 | High-recall caution boundary (val ≥90% recall) |
| moderate → higher   | 0.4219 | Youden J action threshold (balanced)         |

---

## Risk Level Wording

**Lower** (`probability < 0.17`):
> Model-estimated cancer risk is lower, but this does not rule out disease. Consult a clinician if the lesion changes, bleeds, hurts, or concerns you.

**Moderate** (`0.17 ≤ probability < 0.4219`):
> Model-estimated cancer risk is moderate. Consider medical review, especially if the lesion is new, changing, symptomatic, or clinically concerning.

**Higher** (`probability ≥ 0.4219`):
> Model-estimated cancer risk is higher. A dermatologist or qualified clinician should review this lesion.

---

## Disclaimer (always include in app output)

> This AI output is for screening support only and is not a medical diagnosis. It should not replace professional medical evaluation.

---

## Example Output

Input: dermoscopic lesion image

```json
{
  "cancer_risk_probability": 0.72,
  "cancer_risk_percent": 72.0,
  "risk_level": "higher_model_estimated_risk",
  "binary_prediction_at_youden_threshold": 1,
  "recommended_action": "Model-estimated cancer risk is higher. A dermatologist or qualified clinician should review this lesion.",
  "disclaimer": "This AI output is for screening support only and is not a medical diagnosis. It should not replace professional medical evaluation.",
  "model_version": "binary_2phase_v1",
  "threshold_version": "youden_j_v1"
}
```

---

## Model Limitations

- Trained on dermoscopic images only. Performance on smartphone or
  non-dermoscopic images is unknown and likely reduced.
- Only three lesion types in training: NV, MEL, BCC. Other lesion types
  are out-of-distribution and will produce unreliable probabilities.
- This model does not provide a diagnosis.
- Thresholds were selected on a held-out validation set and evaluated once
  on a held-out test set. Real-world performance may differ.
- Grad-CAM visualisations are qualitative only and do not prove clinical reasoning.

---

## Grad-CAM

Grad-CAM explainability visualisations are handled in a separate notebook:
`14_real_image_binary_inference_gradcam.ipynb`. They are **not** part of the
production app inference output.

---

## Test Performance (reference)

| Metric          | Value   |
|-----------------|---------|
| ROC-AUC         | 0.92758 |
| PR-AUC          | 0.90013 |
| Cancer recall (Youden J=0.4219)  | 0.84796 |
| Specificity (Youden J=0.4219)    | 0.84900 |
| Cancer F1 (Youden J=0.4219)      | 0.80895 |

---

*This contract was generated automatically by
`13_binary_2phase_app_inference_contract.ipynb`.*

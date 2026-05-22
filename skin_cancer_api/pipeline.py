from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import efficientnet_b0

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
MODEL_PATH = _ROOT / "pipe output" / "pytorch_training_binary_2phase" / "best_model_binary_2phase.pt"

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOUDEN_THRESH = 0.4219
HIGH_RECALL_THRESH = 0.17
MODEL_VERSION = "binary_2phase_v1"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

WORDING = {
    "lower": (
        "Model-estimated cancer risk is lower, but this does not rule out disease. "
        "Consult a clinician if the lesion changes, bleeds, hurts, or concerns you."
    ),
    "moderate": (
        "Model-estimated cancer risk is moderate. Consider medical review, especially "
        "if the lesion is new, changing, symptomatic, or clinically concerning."
    ),
    "higher": (
        "Model-estimated cancer risk is higher. A dermatologist or qualified clinician "
        "should review this lesion."
    ),
}

DISCLAIMER = (
    "This AI output is for screening support only and is not a medical diagnosis. "
    "It should not replace professional medical evaluation."
)

# ── Transform ─────────────────────────────────────────────────────────────────
_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model() -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_feat, 2),
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    return model


# ── Risk helpers ──────────────────────────────────────────────────────────────
def _risk_level(prob: float) -> str:
    if prob < HIGH_RECALL_THRESH:
        return "lower"
    if prob < YOUDEN_THRESH:
        return "moderate"
    return "higher"


# ── Public inference entry point ──────────────────────────────────────────────
def run_inference(img_pil: Image.Image, model: nn.Module) -> dict:
    inp = _transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)
    prob = float(probs[0, 1].item())
    risk = _risk_level(prob)

    return {
        "cancer_risk_probability":              round(prob, 6),
        "cancer_risk_percent":                  round(prob * 100, 2),
        "risk_level":                           f"{risk}_model_estimated_risk",
        "binary_prediction_at_youden_threshold": int(prob >= YOUDEN_THRESH),
        "recommended_action":                   WORDING[risk],
        "disclaimer":                           DISCLAIMER,
        "model_version":                        MODEL_VERSION,
        "threshold_version":                    "youden_j_v1",
    }

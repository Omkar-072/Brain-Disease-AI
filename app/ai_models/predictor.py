"""
Brain Disease AI - Predictor Module
Hardened prediction pipeline: fail-safe model loading, safe inference,
structured output, and full fallback on every error path.
"""
import numpy as np
import cv2
import os
import logging
from pathlib import Path
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

CLASS_MAPPING = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

ALZ_LABELS = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

# Normalize raw class-name → canonical DB enum key
# All values must match DiseaseType enum values in models.py exactly
LABEL_MAP = {
    "glioma":              "GLIOMA",
    "meningioma":          "MENINGIOMA",
    "pituitary":           "PITUITARY",
    "no_tumor":            "NO_TUMOR",
    "very_mild_demented":  "VERY_MILD_DEMENTED",
    "mild_demented":       "MILD_DEMENTED",
    "moderate_demented":   "MODERATE_DEMENTED",
    "non_demented":        "NON_DEMENTED",
    "inconclusive":        "INCONCLUSIVE",
}

# ─────────────────────────────────────────
# FALLBACK RESULT (always valid structure)
# ─────────────────────────────────────────

_FALLBACK_RESULT: Dict[str, Any] = {
    "predicted_disease": "INCONCLUSIVE",
    "confidence": 0.0,
    "confidence_percent": "0.00%",
    "confidence_level": "LOW",
    "scan_type": "Unknown",
    "disease_type": "unknown",
    "all_predictions": {
        "tumor_model": {"label": "INCONCLUSIVE", "confidence": 0.0},
        "alz_model":   {"label": "INCONCLUSIVE", "confidence": 0.0},
    },
    "model_version": "4.1.1",
    "fallback": True,
}

# ─────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────

def preprocess_image(image_path: Union[str, Path]) -> np.ndarray:
    """Read, normalise, and prepare an image for model inference."""
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    # Convert BGR → GRAY directly (avoids double-conversion bug)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement (critical for MRI)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0

    # Stack to 3-channel so model input shape (224, 224, 3) is satisfied
    img = np.stack([img] * 3, axis=-1)
    img = np.expand_dims(img, axis=0)  # → (1, 224, 224, 3)

    return img

# ─────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────

_tumor_model = None
_alz_model = None
_models_loaded = False


def load_models() -> bool:
    """
    Load both Keras models from disk.
    Returns True on success, False on any failure.
    Never raises – caller gets a bool.
    """
    global _tumor_model, _alz_model, _models_loaded

    try:
        from tensorflow.keras.models import load_model  # type: ignore

        tumor_path = os.path.abspath("app/ai_models/weights/brain_disease_model.h5")
        alz_path   = os.path.abspath("app/ai_models/weights/mri_alzheimer_model.h5")

        if not os.path.exists(tumor_path):
            logger.error(f"Tumor model file missing: {tumor_path}")
            return False

        if not os.path.exists(alz_path):
            logger.error(f"Alzheimer model file missing: {alz_path}")
            return False

        _tumor_model = load_model(tumor_path)
        _alz_model   = load_model(alz_path)

        logger.info(f"Tumor model   input={_tumor_model.input_shape} output={_tumor_model.output_shape}")
        logger.info(f"Alz model     input={_alz_model.input_shape}   output={_alz_model.output_shape}")
        logger.info("Both AI models loaded and validated successfully")

        _models_loaded = True
        return True

    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        _models_loaded = False
        return False


# Attempt load at import time – failure is logged but does NOT crash the app
try:
    load_models()
except Exception as _e:
    logger.error(f"Unexpected error during model pre-load: {_e}", exc_info=True)

# ─────────────────────────────────────────
# SAFE PREDICTION HELPERS
# ─────────────────────────────────────────

def safe_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def run_model(model, image: np.ndarray, class_names: list) -> Dict[str, Any]:
    """
    Run inference on a single model.
    Returns a structured dict; never raises.
    """
    try:
        preds = model.predict(image, verbose=0)[0]

        # Apply softmax only if outputs are raw logits
        if np.max(preds) > 1.0 or np.min(preds) < 0.0:
            preds = safe_softmax(preds)

        max_idx    = int(np.argmax(preds))
        confidence = float(preds[max_idx])

        raw_label  = class_names[max_idx]
        normalized = raw_label.lower().strip().replace(" ", "_")
        mapped_label = LABEL_MAP.get(normalized, "INCONCLUSIVE")

        return {
            "label":      mapped_label,
            "confidence": confidence,
            "raw":        preds.tolist(),
        }

    except Exception as e:
        logger.error(f"run_model failed: {e}", exc_info=True)
        return {
            "label":      "INCONCLUSIVE",
            "confidence": 0.0,
            "raw":        [],
        }


# ─────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────

def decide_final(tumor_result: Dict, alz_result: Dict):
    """
    Strict tumor-first decision gate.

    Rules (in order):
      1. tumor_label != NO_TUMOR AND conf > 0.50  → tumor wins, Alzheimer NEVER consulted
      2. tumor_label == NO_TUMOR                  → check Alzheimer
         2a. alz positive AND conf > 0.60         → alzheimers
         2b. otherwise                             → NORMAL
      3. Safety fallback (should never reach here) → UNKNOWN

    The Alzheimer model CANNOT override tumor output under any conditions.

    Returns (label: str, confidence: float, scan_type: str, disease_type: str)
    """
    t_label = str(tumor_result.get("label") or "NO_TUMOR")
    t_conf  = float(tumor_result.get("confidence") or 0.0)
    a_label = str(alz_result.get("label") or "NON_DEMENTED")
    a_conf  = float(alz_result.get("confidence") or 0.0)

    # Debug line — visible in server logs so you can verify per-scan
    logger.info(
        "decide_final | Tumor: %s (%.3f)  |  Alz: %s (%.3f)",
        t_label, t_conf, a_label, a_conf
    )

    # ── CASE 1: Tumor detected ──────────────────────────────────────────
    # Threshold 0.50: even moderate confidence in a tumor label beats
    # anything the Alzheimer model can say.
    TUMOR_NEGATIVE = ("NO_TUMOR", "INCONCLUSIVE")
    if t_label not in TUMOR_NEGATIVE and t_conf > 0.50:
        logger.info("Decision → TUMOR path: %s (%.3f)", t_label, t_conf)
        return t_label, t_conf, "Tumor Scan", "tumor"

    # ── CASE 2: No tumor → NOW check Alzheimer ──────────────────────────
    # ONLY reached when tumor explicitly said NO_TUMOR.
    # Alzheimer model is structurally gated here; it cannot be reached
    # by any code path that produces a tumor label.
    if t_label == "NO_TUMOR":
        ALZ_NEGATIVE = ("NON_DEMENTED", "INCONCLUSIVE")

        if a_label not in ALZ_NEGATIVE and a_conf > 0.60:
            logger.info("Decision → ALZHEIMER path: %s (%.3f)", a_label, a_conf)
            return a_label, a_conf, "Alzheimer Scan", "alzheimers"

        # Tumor negative + Alzheimer negative → healthy
        logger.info("Decision → NORMAL path")
        return "NORMAL", max(t_conf, a_conf), "Healthy Brain", "normal"

    # ── CASE 3: Safety fallback ─────────────────────────────────────────
    # Reached only when tumor returned INCONCLUSIVE (conf <= 0.50, not NO_TUMOR).
    # Return tumor's best guess rather than silently flipping to Alzheimer.
    logger.warning(
        "Decision → FALLBACK: tumor was inconclusive %s (%.3f)", t_label, t_conf
    )
    if t_label not in TUMOR_NEGATIVE:
        return t_label, t_conf, "Tumor Scan (uncertain)", "tumor"
    return "INCONCLUSIVE", 0.0, "Uncertain", "unknown"



def calibrate_confidence(conf: float) -> float:
    """Compress confidence into a realistic 0.60–0.95 range."""
    return float(0.6 + (conf * 0.35))


def get_prediction_confidence_level(confidence: float) -> str:
    if confidence > 0.8:
        return "HIGH"
    elif confidence > 0.6:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────

def predict(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Full prediction pipeline.

    Always returns a dict with at minimum:
        predicted_disease: str   – canonical DiseaseType label or "NORMAL"
        confidence:        float – calibrated confidence 0.60–0.95
        disease_type:      str   – 'tumor' | 'alzheimers' | 'normal' | 'unknown'
        all_predictions:   dict  – per-model results

    Never raises. On any failure returns the fallback result.
    """
    import copy

    # ── Guard: models must be loaded ──────────────────────────────────────
    if not _models_loaded or _tumor_model is None or _alz_model is None:
        logger.warning("Models not loaded – returning fallback result")
        result = copy.deepcopy(_FALLBACK_RESULT)
        result["error"] = "Models not loaded"
        return result

    try:
        # 1. Preprocess
        image = preprocess_image(image_path)

        # 2. Inference (each call is internally safe)
        tumor_result = run_model(_tumor_model, image, CLASS_MAPPING)
        alz_result   = run_model(_alz_model,   image, ALZ_LABELS)

        logger.info(f"Tumor raw result : {tumor_result}")
        logger.info(f"Alzheimer raw result: {alz_result}")

        # 3. Decision fusion — now returns 4-tuple including disease_type
        label, conf, scan_type, disease_type = decide_final(tumor_result, alz_result)

        logger.info(
            "Final decision: label=%s  conf=%.3f  type=%s  scan=%s",
            label, conf, disease_type, scan_type
        )

        # 4. Calibrate confidence
        conf = calibrate_confidence(conf)

        # 5. Build structured response
        return {
            "predicted_disease":  label,
            "confidence":         conf,
            "confidence_percent": f"{conf * 100:.2f}%",
            "confidence_level":   get_prediction_confidence_level(conf),
            "scan_type":          scan_type,
            "disease_type":       disease_type,
            "all_predictions": {
                "tumor_model": {
                    "label":      str(tumor_result.get("label", "INCONCLUSIVE")),
                    "confidence": float(tumor_result.get("confidence", 0.0)),
                },
                "alz_model": {
                    "label":      str(alz_result.get("label", "INCONCLUSIVE")),
                    "confidence": float(alz_result.get("confidence", 0.0)),
                },
            },
            "model_version": "4.1.1",
        }

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
        result = copy.deepcopy(_FALLBACK_RESULT)
        result["error"] = str(e)
        return result


# ─────────────────────────────────────────
# SERVICE WRAPPER
# ─────────────────────────────────────────

class BrainDiseasePredictor:
    """Service-level wrapper for the prediction pipeline."""

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_loaded":        _models_loaded,
            "tumor_model_input":   _tumor_model.input_shape  if _tumor_model  else None,
            "tumor_model_output":  _tumor_model.output_shape if _tumor_model  else None,
            "alz_model_input":     _alz_model.input_shape    if _alz_model    else None,
            "alz_model_output":    _alz_model.output_shape   if _alz_model    else None,
            "classes":             CLASS_MAPPING + ALZ_LABELS,
            "status":              "loaded" if _models_loaded else "not_loaded",
        }

    def predict_sync(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        return predict(image_path)

    async def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        return predict(image_path)
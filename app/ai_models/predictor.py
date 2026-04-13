import numpy as np
import cv2
import os
import logging
import asyncio

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# 1. LABEL CONFIG 
# ─────────────────────────────────────────
# NOTE: Make sure the ALZ_CLASSES order exactly matches what Colab printed!
TUMOR_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
STROKE_CLASSES = ["No Stroke", "Stroke"] 
ALZ_CLASSES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

LABEL_MAP = {
    "glioma": "GLIOMA",
    "meningioma": "MENINGIOMA",
    "pituitary": "PITUITARY",
    "no_tumor": "NO_TUMOR",
    "stroke": "STROKE",
    "no_stroke": "NO_STROKE", 
    "very_mild_demented": "VERY_MILD_DEMENTED",
    "mild_demented": "MILD_DEMENTED",
    "moderate_demented": "MODERATE_DEMENTED",
    "non_demented": "NON_DEMENTED",
    "inconclusive": "INCONCLUSIVE",
}

# ─────────────────────────────────────────
# 2. MODEL STORE
# ─────────────────────────────────────────
_models = {
    "mri_tumor": None,
    "ct_stroke": None,
    "pet_alzheimer": None
}

_models_loaded = False

def load_models():
    global _models, _models_loaded
    try:
        from tensorflow.keras.models import load_model

        paths = {
            "mri_tumor": "app/ai_models/weights/brain_disease_model.h5",
            "ct_stroke": "app/ai_models/weights/ct_stroke_model.h5",
            "pet_alzheimer": "app/ai_models/weights/pet_alzheimer_model.h5",
        }

        for key, path in paths.items():
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                logger.warning(f"[{key.upper()}] model missing: {abs_path}")
                continue

            _models[key] = load_model(abs_path)
            logger.info(f"[{key.upper()}] model loaded successfully")

        _models_loaded = True
        return True

    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        return False

# Initialize models on startup
load_models()

# ─────────────────────────────────────────
# 3. CORE UTILITIES
# ─────────────────────────────────────────
def preprocess_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Invalid image")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # ALL models now safely use 0-1 scaling
    img = img.astype("float32") / 255.0 
        
    img = np.expand_dims(img, axis=0)
    return img

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_model(model, image, classes):
    if model is None:
        return {"label": "MODEL_OFFLINE", "confidence": 0.0}

    preds = model.predict(image, verbose=0)[0]
    
    print("========================================")
    print(f"RAW AI OUTPUT: {preds}")

    # Fallback for old 1-node binary models
    if preds.size == 1:
        prob = float(preds[0])
        idx = 0 if prob > 0.5 else 1
        conf = prob if idx == 0 else (1.0 - prob)
    # Modern Categorical / 2-node Binary models
    else:
        if np.max(preds) > 1:
            preds = softmax(preds)
        idx = int(np.argmax(preds))
        conf = float(preds[idx])

    label = classes[idx].lower().replace(" ", "_")
    label = LABEL_MAP.get(label, "INCONCLUSIVE")

    print(f"FINAL DECISION: {label} with {conf*100:.2f}% raw confidence")
    print("========================================")

    return {"label": label, "confidence": conf}

# ─────────────────────────────────────────
# 4. PERFECTLY ISOLATED PIPELINES
# ─────────────────────────────────────────
def predict_mri(image):
    # MRI is ONLY for Tumors
    result = run_model(_models["mri_tumor"], image, TUMOR_CLASSES)
    if result["label"] == "MODEL_OFFLINE": return "INCONCLUSIVE", 0.0, "system_error"
    return result["label"], result["confidence"], "tumor"

def predict_ct(image):
    # CT is ONLY for Strokes
    result = run_model(_models["ct_stroke"], image, STROKE_CLASSES)
    if result["label"] == "MODEL_OFFLINE": return "INCONCLUSIVE", 0.0, "system_error"
    return result["label"], result["confidence"], "stroke"

def predict_pet(image):
    # PET is ONLY for Alzheimer's
    result = run_model(_models["pet_alzheimer"], image, ALZ_CLASSES)
    if result["label"] == "MODEL_OFFLINE": return "INCONCLUSIVE", 0.0, "system_error"
    return result["label"], result["confidence"], "alzheimers"

# ─────────────────────────────────────────
# 5. UI HELPERS
# ─────────────────────────────────────────
def calibrate_confidence(conf: float) -> float:
    if conf == 0.0: return 0.0
    return float(0.6 + (conf * 0.35))

def get_prediction_confidence_level(conf: float) -> str:
    if conf == 0.0: return "ERROR"
    if conf > 0.8: return "HIGH"
    elif conf > 0.6: return "MEDIUM"
    return "LOW"

# ─────────────────────────────────────────
# 6. MAIN ENTRY & WRAPPER
# ─────────────────────────────────────────
def _process_scan(image_path, scan_type="MRI"):
    try:
        scan_type = scan_type.upper()
        logger.info(f"====== ROUTING SCAN AS: {scan_type} ======")

        # Preprocess the image (scaling is now universally True)
        image = preprocess_image(image_path)

        if scan_type == "MRI":
            label, conf, dtype = predict_mri(image)
        elif scan_type == "CT":
            label, conf, dtype = predict_ct(image)
        elif scan_type == "PET":
            label, conf, dtype = predict_pet(image)
        else:
            return {"predicted_disease": "INCONCLUSIVE", "confidence": 0.0, "scan_type": scan_type}

        calibrated_conf = calibrate_confidence(conf)

        return {
            "predicted_disease": label,
            "confidence": calibrated_conf,
            "confidence_percent": f"{calibrated_conf*100:.2f}%",
            "confidence_level": get_prediction_confidence_level(calibrated_conf),
            "scan_type": scan_type,
            "disease_type": dtype,
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return {
            "predicted_disease": "INCONCLUSIVE", 
            "confidence": 0.0, 
            "confidence_percent": "0.00%",
            "confidence_level": "ERROR"
        }

class BrainDiseasePredictor:
    def get_model_info(self):
        return {
            "models_loaded": _models_loaded,
            "available_models": {k: v is not None for k, v in _models.items()}
        }

    async def predict(self, image_path, scan_type="MRI"):
        return await asyncio.to_thread(_process_scan, image_path, scan_type)
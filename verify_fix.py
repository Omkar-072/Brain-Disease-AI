import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("."))

from app.ai_models.predictor import _process_scan, run_model, TUMOR_CLASSES, STROKE_CLASSES
import numpy as np

print("--- Testing Stroke Logic (Binary 1-node) ---")
class MockModel:
    def __init__(self, val): self.val = val
    def predict(self, x, verbose=0): return np.array([[self.val]])

# High probability should be Stroke (idx 1)
model_high = MockModel(0.9)
res = run_model(model_high, None, STROKE_CLASSES)
print(f"High Prob (0.9): {res}")

# Low probability should be No Stroke (idx 0)
model_low = MockModel(0.1)
res = run_model(model_low, None, STROKE_CLASSES)
print(f"Low Prob (0.1): {res}")

print("\n--- Testing MRI Dual Check (Simulated) ---")
# To test this fully without real weights, I'd need to mock the load_models or weights.
# But I can check the logic in _process_scan if I patch the model calls.

from unittest.mock import patch

with patch('app.ai_models.predictor.predict_mri_tumor') as mock_t, \
     patch('app.ai_models.predictor.predict_mri_alzheimer') as mock_a, \
     patch('app.ai_models.predictor.preprocess_image') as mock_p:
    
    mock_p.return_value = np.zeros((1, 224, 224, 3))
    
    # CASE 1: Both normal
    mock_t.return_value = ("NO_TUMOR", 0.95, "tumor")
    mock_a.return_value = ("NON_DEMENTED", 0.92, "alzheimers")
    res = _process_scan("dummy.jpg", "MRI")
    print(f"Both Normal: label={res['predicted_disease']}, type={res['disease_type']}")
    
    # CASE 2: Tumor found
    mock_t.return_value = ("GLIOMA", 0.85, "tumor")
    mock_a.return_value = ("NON_DEMENTED", 0.98, "alzheimers")
    res = _process_scan("dummy.jpg", "MRI")
    print(f"Tumor Found: label={res['predicted_disease']}, type={res['disease_type']}")

    # CASE 3: Alzheimer found (No Tumor)
    mock_t.return_value = ("NO_TUMOR", 0.99, "tumor")
    mock_a.return_value = ("MILD_DEMENTED", 0.75, "alzheimers")
    res = _process_scan("dummy.jpg", "MRI")
    print(f"Alzheimer Found: label={res['predicted_disease']}, type={res['disease_type']}")

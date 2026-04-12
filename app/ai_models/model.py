"""
Brain Disease AI - CNN Model Architecture
Deep learning model for brain disease classification
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Disease classes
DISEASE_CLASSES = [
    "healthy",
    "stroke",
    "epilepsy",
    "alzheimer",
    "parkinson",
    "brain_tumor"
]

NUM_CLASSES = len(DISEASE_CLASSES)
INPUT_SHAPE = (224, 224, 3)


def create_cnn_model_tensorflow():
    """
    Create a CNN model using TensorFlow/Keras for brain disease classification.
    Uses transfer learning with EfficientNetB0 as base.
    
    Returns:
        Compiled Keras model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import EfficientNetB0
        
        # Load pre-trained EfficientNetB0 (without top layers)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=INPUT_SHAPE
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("TensorFlow CNN model created successfully")
        return model
        
    except ImportError as e:
        logger.error(f"TensorFlow not available: {e}")
        return None


def create_cnn_model_pytorch():
    """
    Create a CNN model using PyTorch for brain disease classification.
    Uses transfer learning with ResNet18 as base.
    
    Returns:
        PyTorch model
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
        
        class BrainDiseaseClassifier(nn.Module):
            def __init__(self, num_classes=NUM_CLASSES):
                super(BrainDiseaseClassifier, self).__init__()
                
                # Load pre-trained ResNet18
                self.resnet = models.resnet18(pretrained=True)
                
                # Freeze early layers
                for param in list(self.resnet.parameters())[:-10]:
                    param.requires_grad = False
                
                # Replace the final fully connected layer
                num_features = self.resnet.fc.in_features
                self.resnet.fc = nn.Sequential(
                    nn.Linear(num_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                return self.resnet(x)
        
        model = BrainDiseaseClassifier()
        logger.info("PyTorch CNN model created successfully")
        return model
        
    except ImportError as e:
        logger.error(f"PyTorch not available: {e}")
        return None


def create_simple_cnn_model():
    """
    Create a simple CNN model without transfer learning.
    Useful when pre-trained weights are not available.
    
    Returns:
        Keras model
    """
    try:
        from tensorflow.keras import layers, models
        
        model = models.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth Conv Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fifth Conv Block
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Simple CNN model created successfully")
        return model
        
    except ImportError as e:
        logger.error(f"Failed to create simple CNN: {e}")
        return None


class ModelWrapper:
    """Wrapper class to handle both TensorFlow and PyTorch models"""
    
    def __init__(self, framework: str = "tensorflow"):
        self.framework = framework
        self.model = None
        self.device = None
        
        if framework == "tensorflow":
            self.model = create_cnn_model_tensorflow()
            if self.model is None:
                self.model = create_simple_cnn_model()
        elif framework == "pytorch":
            self.model = create_cnn_model_pytorch()
            if self.model is not None:
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = self.model.to(self.device)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Make prediction on a single image.
        
        Args:
            image: Preprocessed image array of shape (H, W, C)
            
        Returns:
            Probability distribution over classes
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Add batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        if self.framework == "tensorflow":
            predictions = self.model.predict(image, verbose=0)
            return predictions[0]
        
        elif self.framework == "pytorch":
            import torch
            
            # Convert to tensor and adjust dimensions (B, C, H, W)
            tensor = torch.FloatTensor(image).permute(0, 3, 1, 2)
            tensor = tensor.to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            return probabilities.cpu().numpy()[0]
    
    def load_weights(self, weights_path: str) -> bool:
        """Load model weights from file"""
        try:
            if self.framework == "tensorflow":
                self.model.load_weights(weights_path)
            elif self.framework == "pytorch":
                import torch
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            logger.info(f"Weights loaded from {weights_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return False
    
    def save_weights(self, weights_path: str) -> bool:
        """Save model weights to file"""
        try:
            if self.framework == "tensorflow":
                self.model.save_weights(weights_path)
            elif self.framework == "pytorch":
                import torch
                torch.save(self.model.state_dict(), weights_path)
            logger.info(f"Weights saved to {weights_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")
            return False


def get_class_name(class_index: int) -> str:
    """Get disease class name from index"""
    if 0 <= class_index < len(DISEASE_CLASSES):
        return DISEASE_CLASSES[class_index]
    return "unknown"


def get_class_index(class_name: str) -> int:
    """Get index from disease class name"""
    try:
        return DISEASE_CLASSES.index(class_name.lower())
    except ValueError:
        return -1

"""
Brain Disease AI - Model Training Script
Script for training the CNN model on brain scan datasets
"""
import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disease classes
DISEASE_CLASSES = ["healthy", "stroke", "epilepsy", "alzheimer", "parkinson", "brain_tumor"]
NUM_CLASSES = len(DISEASE_CLASSES)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 50


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from directory structure.
    Expected structure:
        data_dir/
            train/
                healthy/
                stroke/
                epilepsy/
                ...
            val/
                healthy/
                stroke/
                ...
    
    Args:
        data_dir: Root directory containing train and val folders
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=DISEASE_CLASSES
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=DISEASE_CLASSES
    )
    
    return train_generator, val_generator


def create_model():
    """Create the CNN model for training"""
    from app.ai_models.model import create_cnn_model_tensorflow
    return create_cnn_model_tensorflow()


def train_model(
    data_dir: str,
    output_dir: str = "app/ai_models/weights",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE
):
    """
    Train the brain disease classification model.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save trained weights
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
    )
    
    logger.info("Starting model training...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading dataset from {data_dir}")
    train_gen, val_gen = load_dataset(data_dir)
    
    # Create model
    logger.info("Creating model...")
    model = create_model()
    model.summary()
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        ModelCheckpoint(
            filepath=str(output_path / "brain_disease_model.h5"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(output_path / f"logs/{timestamp}"),
            histogram_freq=1
        )
    ]
    
    # Train
    logger.info(f"Training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(str(output_path / "brain_disease_model_final.h5"))
    logger.info(f"Model saved to {output_path}")
    
    # Save training history
    import json
    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    
    # Print final metrics
    logger.info("\n=== Training Complete ===")
    logger.info(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model, history


def evaluate_model(model_path: str, test_dir: str):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to saved model weights
        test_dir: Directory containing test data
    """
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Load test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=DISEASE_CLASSES,
        shuffle=False
    )
    
    # Evaluate
    logger.info("Evaluating model...")
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    logger.info(f"Test Loss: {loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate predictions
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    
    # Classification report
    logger.info("\n=== Classification Report ===")
    print(classification_report(true_classes, predicted_classes, target_names=DISEASE_CLASSES))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=DISEASE_CLASSES,
                yticklabels=DISEASE_CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    logger.info("Confusion matrix saved to confusion_matrix.png")
    
    return accuracy


def create_sample_data(output_dir: str, samples_per_class: int = 10):
    """
    Create sample placeholder data for testing.
    WARNING: This creates random images - NOT for production use!
    
    Args:
        output_dir: Directory to create sample data
        samples_per_class: Number of samples per class
    """
    from PIL import Image
    import random
    
    output_path = Path(output_dir)
    
    for split in ["train", "val"]:
        for disease in DISEASE_CLASSES:
            class_dir = output_path / split / disease
            class_dir.mkdir(parents=True, exist_ok=True)
            
            num_samples = samples_per_class if split == "train" else samples_per_class // 2
            
            for i in range(num_samples):
                # Create random grayscale image (simulating brain scan)
                img_array = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
                img = Image.fromarray(img_array, mode='L')
                img = img.convert('RGB')
                
                img.save(class_dir / f"{disease}_{i:04d}.jpg")
    
    logger.info(f"Sample data created in {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Brain Disease Classification Model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="app/ai_models/weights", help="Output directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing model")
    parser.add_argument("--create-sample", action="store_true", help="Create sample data")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data(args.data_dir)
    elif args.evaluate:
        evaluate_model(
            model_path=f"{args.output_dir}/brain_disease_model.h5",
            test_dir=args.data_dir
        )
    else:
        train_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

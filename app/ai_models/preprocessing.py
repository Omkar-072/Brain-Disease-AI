"""
Brain Disease AI - Image Preprocessing Utilities
Handles image loading, preprocessing, and augmentation for brain scans
"""
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Target image size for model input
TARGET_SIZE = (224, 224)


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file path.
    Supports common image formats (JPEG, PNG) and attempts to handle DICOM/NIfTI.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy array of the image
    """
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    ext = path.suffix.lower()
    
    try:
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # Standard image formats
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        elif ext == '.dcm':
            # DICOM format - try to use pydicom if available
            try:
                import pydicom
                dcm = pydicom.dcmread(str(path))
                image = dcm.pixel_array
                # Normalize to 0-255
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                # Convert to RGB if grayscale
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            except ImportError:
                logger.warning("pydicom not installed. Treating as standard image.")
                image = cv2.imread(str(path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
        elif ext in ['.nii', '.nii.gz']:
            # NIfTI format - try to use nibabel if available
            try:
                import nibabel as nib
                nii = nib.load(str(path))
                data = nii.get_fdata()
                # Take middle slice if 3D
                if len(data.shape) == 3:
                    mid_slice = data.shape[2] // 2
                    image = data[:, :, mid_slice]
                else:
                    image = data
                # Normalize to 0-255
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                # Convert to RGB if grayscale
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            except ImportError:
                logger.warning("nibabel not installed. Cannot process NIfTI files.")
                raise ValueError("NIfTI support not available. Please install nibabel.")
        else:
            # Try loading as standard image
            image = cv2.imread(str(path))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unsupported image format: {ext}")
                
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        raise


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Resize image to target dimensions.
    
    Args:
        image: Input image array
        target_size: Tuple of (width, height)
        
    Returns:
        Resized image array
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        image: Input image array (0-255)
        
    Returns:
        Normalized image array (0-1)
    """
    return image.astype(np.float32) / 255.0


def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Apply denoising to reduce noise in the image.
    
    Args:
        image: Input image array
        strength: Denoising strength (higher = more smoothing)
        
    Returns:
        Denoised image array
    """
    # Ensure image is in uint8 format for denoising
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Apply non-local means denoising
    if len(image.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    else:
        denoised = cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    
    return denoised


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image array
        
    Returns:
        Contrast-enhanced image array
    """
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Convert to LAB color space
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
    else:
        l_channel = image
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    
    # Merge back if color image
    if len(image.shape) == 3:
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    else:
        enhanced = enhanced_l
    
    return enhanced


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur to reduce high-frequency noise.
    
    Args:
        image: Input image array
        kernel_size: Size of the Gaussian kernel (must be odd)
        
    Returns:
        Blurred image array
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def preprocess_brain_scan(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = TARGET_SIZE,
    denoise: bool = True,
    enhance: bool = True
) -> np.ndarray:
    """
    Complete preprocessing pipeline for brain scan images.
    
    Steps:
    1. Load image
    2. Denoise (optional)
    3. Enhance contrast (optional)
    4. Resize to target dimensions
    5. Normalize pixel values
    
    Args:
        image_path: Path to the image file
        target_size: Target dimensions (width, height)
        denoise: Whether to apply denoising
        enhance: Whether to apply contrast enhancement
        
    Returns:
        Preprocessed image array ready for model input
    """
    # Load image
    image = load_image(image_path)
    
    # Apply denoising
    if denoise:
        image = denoise_image(image)
    
    # Enhance contrast
    if enhance:
        image = enhance_contrast(image)
    
    # Resize
    image = resize_image(image, target_size)
    
    # Normalize
    image = normalize_image(image)
    
    return image


def prepare_batch(images: list, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Prepare a batch of images for model inference.
    
    Args:
        images: List of image paths or arrays
        target_size: Target dimensions
        
    Returns:
        Batch array of shape (N, H, W, C)
    """
    batch = []
    
    for img in images:
        if isinstance(img, (str, Path)):
            processed = preprocess_brain_scan(img, target_size)
        else:
            processed = resize_image(img, target_size)
            processed = normalize_image(processed)
        batch.append(processed)
    
    return np.array(batch)


def augment_image(image: np.ndarray) -> list:
    """
    Generate augmented versions of an image for training.
    
    Args:
        image: Input image array
        
    Returns:
        List of augmented images
    """
    augmented = [image]
    
    # Horizontal flip
    augmented.append(cv2.flip(image, 1))
    
    # Vertical flip
    augmented.append(cv2.flip(image, 0))
    
    # Rotation variations
    rows, cols = image.shape[:2]
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented.append(rotated)
    
    # Brightness variations
    for factor in [0.8, 1.2]:
        adjusted = np.clip(image * factor, 0, 1 if image.max() <= 1 else 255)
        augmented.append(adjusted.astype(image.dtype))
    
    return augmented

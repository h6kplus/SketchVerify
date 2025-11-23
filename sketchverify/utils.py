"""
Utility functions for image processing and encoding
"""
import base64
import re
import numpy as np
from PIL import Image
import cv2
import io


def encode_image(image_path):
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_pil_image(pil_image):
    """Encode PIL Image to base64"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=90)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def normalize_number(number: str) -> str:
    """Normalize numbers by zero-padding them"""
    return re.sub(r'\d+', lambda x: x.group().zfill(4), number)


def composite_images_with_mask(original_image, generated_image, mask):
    """Composite generated and original images using mask."""
    original_np = np.array(original_image)
    generated_np = np.array(generated_image)

    # Handle mask format
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    else:
        mask_np = mask

    # Convert mask to binary if needed
    if mask_np.max() > 1:
        mask_np = (mask_np > 127).astype(np.float32)
    else:
        mask_np = mask_np.astype(np.float32)

    # Handle RGB mask (convert to single channel)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    # Ensure mask is same size as images
    if mask_np.shape != original_np.shape[:2]:
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(original_image.size, Image.Resampling.NEAREST)
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0

    # Expand mask to 3 channels for RGB composition
    mask_3ch = np.stack([mask_np, mask_np, mask_np], axis=-1)

    # Composite: mask areas from generated, non-mask areas from original
    composited = generated_np * mask_3ch + original_np * (1 - mask_3ch)

    return Image.fromarray(composited.astype(np.uint8))


def apply_mask_feathering(mask, feather_radius=3):
    """Apply feathering to mask edges for smoother blending."""
    # Convert to uint8 if needed
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask

    # Apply Gaussian blur for feathering
    feathered = cv2.GaussianBlur(mask_uint8, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)

    return feathered.astype(np.float32) / 255.0


# Coordinate system guide for prompts
COORDS_GUIDE = """
**CAMERA-SPACE COORDINATES (use in ALL sections)**
- Work strictly in the 2D image plane.
- Origin (0,0) at TOP-LEFT; x increases RIGHT, y increases DOWN.
- Use NORMALIZED units [0,1] when quantifying positions/gaps/sizes.
  - Examples: "x≈0.62", "y≈0.28", "gap ~0.04 of frame width (0.04w)", "height ~0.12 of frame height (0.12h)".
- Allowed relations: left_of / right_of / above / below / overlapping / touching / on / inside / near (~distance in normalized units),
  occluding / partially occluded / aligned_with / facing_toward.
- Allowed sector terms: "upper-left (x<0.5,y<0.5)", "center (~0.4–0.6 on both axes)", "lower-right (x>0.5,y>0.5)".
- FORBIDDEN in captions: 3D/world units ("meters", "depth", "away from camera") unless inferable from clear monocular cues—prefer camera-space wording instead.
"""

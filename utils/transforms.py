"""Data augmentation: multi-view generation via center crop + CLAHE."""

import random
import cv2
import numpy as np
import torch


def generate_view_b(image, crop_ratio=0.75):
    """Generate view B: center crop to crop_ratio + CLAHE + resize back.

    Args:
        image: numpy array (H, W) or (H, W, C)
        crop_ratio: center crop ratio (0.7~0.8)

    Returns:
        view_b: numpy array with same shape as input
    """
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    squeeze = False
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
        squeeze = True

    h, w = image.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)

    # Center crop
    y1 = (h - ch) // 2
    x1 = (w - cw) // 2
    cropped = image[y1:y1 + ch, x1:x1 + cw]

    # CLAHE
    if cropped.ndim == 3:
        result = np.zeros_like(cropped)
        for c in range(cropped.shape[2]):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result[:, :, c] = clahe.apply(
                (cropped[:, :, c] * 255).astype(np.uint8)
            ).astype(np.float32) / 255.0
        cropped = result
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cropped = clahe.apply(
            (cropped * 255).astype(np.uint8)
        ).astype(np.float32) / 255.0

    # Resize back to original size
    view_b = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    if squeeze:
        view_b = np.squeeze(view_b)

    if view_b.ndim == 2:
        view_b = view_b[np.newaxis, :, :]
    else:
        view_b = view_b.transpose(2, 0, 1)

    return view_b.astype(np.float32)

import math
import cv2
import numpy as np
import config

import re

def is_diagram_caption(text):
    if not text:
        return False
    text = text.strip()
    if re.match(r'^(figure|fig\.?|diagram)\s*[\d\.]+', text, re.IGNORECASE):
        return True
    return False

def calculate_center(bbox):
    """Calculate the center point of a bounding box. bbox: [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_distance(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def is_overlapping(bbox1, bbox2):
    """
    Check if two bounding boxes overlap (axis-aligned).
    bbox format: [x1, y1, x2, y2]
    """
    if bbox1[0] > bbox2[2] or bbox2[0] > bbox1[2]:
        return False
    if bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]:
        return False
    return True


def does_person_block_board(person_bbox, whiteboard_roi):
    """Returns True if a person's bounding box overlaps the whiteboard ROI."""
    return is_overlapping(person_bbox, whiteboard_roi)


def get_absolute_roi(roi, frame_w, frame_h):
    """
    Convert ROI coordinates into absolute coordinates for the original frame dimensions.
    Handles float percentages (0.0 to 1.0) and absolute coordinates (scaled from 640x480).
    """
    x1, y1, x2, y2 = roi
    # If all values are in range [0, 1], treat as percentages
    if all(0.0 <= val <= 1.0 for val in roi):
        ax1 = int(x1 * frame_w)
        ay1 = int(y1 * frame_h)
        ax2 = int(x2 * frame_w)
        ay2 = int(y2 * frame_h)
    else:
        # Otherwise treat as coordinates relative to 640x480 res and scale
        scale_x = frame_w / 640.0
        scale_y = frame_h / 480.0
        ax1 = int(x1 * scale_x)
        ay1 = int(y1 * scale_y)
        ax2 = int(x2 * scale_x)
        ay2 = int(y2 * scale_y)
    
    # Keep within boundaries
    ax1 = max(0, min(frame_w - 1, ax1))
    ay1 = max(0, min(frame_h - 1, ay1))
    ax2 = max(0, min(frame_w, ax2))
    ay2 = max(0, min(frame_h, ay2))
    return [ax1, ay1, ax2, ay2]


def preprocess_for_ocr(roi_img):
    """
    Preprocess the cropped ROI frame based on config.OCR_PREPROCESSING_MODE.
    If the mode is 'multi', it returns a dict of {'grayscale': img, 'otsu': img, 'adaptive': img}.
    Otherwise, returns a single preprocessed image.
    """
    mode = getattr(config, 'OCR_PREPROCESSING_MODE', 'multi')
    
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # 2. Denoise with a light bilateral filter to keep text edges crisp
    denoised = cv2.bilateralFilter(gray, 5, 50, 50)

    # 3. Apply CLAHE for local contrast / glare reduction
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(denoised)

    # 4. Upscale (Improves OCR accuracy significantly)
    scale = getattr(config, 'OCR_RESIZE_SCALE', 2.0)
    if scale != 1.0:
        h, w = cl_img.shape[:2]
        cl_img = cv2.resize(cl_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # 5. Sharpen (Enhances stroke definition)
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(cl_img, -1, sharpen_kernel)

    def get_otsu(img):
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def get_adaptive(img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    if mode == 'multi':
        return {
            'grayscale': sharpened,
            'otsu': get_otsu(sharpened),
            'adaptive': get_adaptive(sharpened)
        }
    elif mode == 'otsu':
        return get_otsu(sharpened)
    elif mode == 'adaptive':
        return get_adaptive(sharpened)
    elif mode == 'fast':
        return sharpened  # Fast uses grayscale + CLAHE + light sharpening
    else:
        return sharpened


def resize_keep_aspect_ratio(image, max_width=None, max_height=None, inter=cv2.INTER_LINEAR):
    """
    Resizes an image while preserving the aspect ratio.
    If both max_width and max_height are provided, scales to fit within the bounding box.
    """
    (h, w) = image.shape[:2]

    if max_width is None and max_height is None:
        return image

    if max_width is not None and max_height is not None:
        r = min(max_width / float(w), max_height / float(h))
    elif max_width is not None:
        r = max_width / float(w)
    else:
        r = max_height / float(h)

    dim = (max(1, int(w * r)), max(1, int(h * r)))
    return cv2.resize(image, dim, interpolation=inter)

import os
import json

# Video and Display
# For Mobile IP Camera (e.g., using "IP Webcam" app on Android):
# 1. Connect phone and PC to the same WiFi.
# 2. Open IP Webcam app and tap "Start server".
# 3. Enter the URL shown on your phone here, appending '/video'.
# Example: DEFAULT_VIDEO_SOURCE = "http://192.168.1.100:8080/video"
DEFAULT_VIDEO_SOURCE = "http://172.24.37.79:8080/video"
RESIZE_DIM = (640, 480)   # Frame resize dimensions for YOLO tracking
FRAME_SKIP = 3            # Process every N frames for YOLO to save CPU

# ROI Configuration [x1, y1, x2, y2]
# Percentage-based ROI by default so it works on any camera resolution.
# Example: [0.05, 0.08, 0.95, 0.88] means 5% to 95% width, 8% to 88% height.
WHITEBOARD_ROI = [0.05, 0.08, 0.95, 0.88]

# Tracking & Activity Configuration
MOVEMENT_THRESHOLD = 15.0      # Minimum movement distance to be considered "Active"
STABILITY_THRESHOLD = 10.0     # Max average scene movement to allow OCR (scene must be stable)
SMOOTHING_WINDOW = 5           # Number of frames to calculate rolling average of movement
BOARD_STABLE_SECONDS = 2.0     # Duration in seconds the board must remain stable before starting OCR checks

# OCR Configuration
PROCESS_OCR_INTERVAL = 30      # Fallback frame interval for OCR checks
OCR_COOLDOWN = 5.0             # Seconds to wait after successful text extraction before running OCR again
OCR_PREPROCESSING_MODE = 'fast' # 'fast' (default), 'grayscale', 'otsu', 'adaptive', or 'multi' (runs all three and selects best)

# OCR Quality Tuning
OCR_CONFIDENCE_THRESHOLD = 0.45  # Confidence threshold for OCR text detection
OCR_LINE_Y_THRESHOLD = 25        # Vertical pixel grouping threshold for reconstructing visual lines
OCR_MIN_TEXT_LENGTH = 3          # Discard any text fragment shorter than this
OCR_CONFIRMATION_FRAMES = 2      # Confirm the text across this many stable reads before saving
OCR_RESIZE_SCALE = 1.5           # Upscaling factor for OCR preprocessing image
OCR_MAX_WIDTH = 1200             # Downscale the ROI if it exceeds this width before passing to EasyOCR

# Display Configuration
ROTATE_180 = False               # Change to True if the video is upside down
FLIP_HORIZONTAL = False          # Change to True if the text reads backwards like a mirror

# Output & Logging Configuration
OUTPUT_DIR = 'output'
NOTES_FILE = os.path.join(OUTPUT_DIR, 'notes.txt')
ACTIVITY_FILE = os.path.join(OUTPUT_DIR, 'activity.csv')
SNAPSHOTS_DIR = os.path.join(OUTPUT_DIR, 'board_snapshots')
LOG_BUFFER_SIZE = 25           # Write to CSV only when buffer reaches this size

# ─────────────────────────────────────────────────────────
# Override with local settings if available
# ─────────────────────────────────────────────────────────
LOCAL_SETTINGS_PATH = "local_settings.json"

if os.path.exists(LOCAL_SETTINGS_PATH):
    try:
        with open(LOCAL_SETTINGS_PATH, 'r', encoding='utf-8') as f:
            local_config = json.load(f)
            
        if "WHITEBOARD_ROI" in local_config:
            WHITEBOARD_ROI = local_config["WHITEBOARD_ROI"]
        if "OCR_PREPROCESSING_MODE" in local_config:
            OCR_PREPROCESSING_MODE = local_config["OCR_PREPROCESSING_MODE"]
            
        print(f"[Config] Successfully loaded runtime settings from {LOCAL_SETTINGS_PATH}")
    except Exception as e:
        print(f"[Warning] Failed to load {LOCAL_SETTINGS_PATH}: {e}. Falling back to default config.")

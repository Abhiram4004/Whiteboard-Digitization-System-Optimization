import re
import difflib
import collections
import os
import cv2
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Any

import easyocr

import config
from utils import does_person_block_board, preprocess_for_ocr, resize_keep_aspect_ratio, is_diagram_caption


@dataclass
class OCRFrameResult:
    text: Optional[str]
    status_msg: str
    candidate_text: str
    roi_image: Any
    color_snapshot_path: Optional[str]
    cleaned_snapshot_path: Optional[str]
    timestamp: str


def _is_garbage(text):
    text = text.strip()
    
    if is_diagram_caption(text):
        return False
        
    if len(text) < config.OCR_MIN_TEXT_LENGTH:
        return True

    # Reject lines with less than 2 alphabetic characters for normal OCR notes
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 2:
        return True

    if not any(c.isalnum() for c in text):
        return True
    if "`" in text:
        return True

    valid_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 =+-*/()[]{}<>;,._?$%'"
    valid_count = sum(1 for c in text if c in valid_chars)
    if valid_count / max(1, len(text)) < 0.85:
        return True

    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    garbage_words = {"hentna", "telate", "topie"}
    if any(w in garbage_words for w in words):
        return True
    return False


def clean_ocr_text(text):
    corrections = {
        r'\b0[fF]\b': 'of',
        r'\b[tT]0\b': 'to',
        r'\bCompleton\b': 'Completion',
        r'\bcompleton\b': 'completion',
        r'\bAvrival\b': 'Arrival',
        r'\bavrival\b': 'arrival',
        r'\bWalting\b': 'Waiting',
        r'\bwalting\b': 'waiting',
        r'\bQucuc\b': 'Queue',
        r'\bqucuc\b': 'queue',
        r'\bSchedluling\b': 'Scheduling',
        r'\bschedluling\b': 'scheduling',
        r'\b[sS]2[ _]?[jJ]ob[ _]?[nN]ext\b': 'Shortest Job Next'
    }
    cleaned = text
    for pattern, repl in corrections.items():
        cleaned = re.sub(pattern, repl, cleaned)
    return cleaned


def _similarity(a, b):
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def group_ocr_results_by_line(results):
    if not results:
        return []

    items = []
    for bbox, text, _conf in results:
        ys = [pt[1] for pt in bbox]
        xs = [pt[0] for pt in bbox]
        y_center = sum(ys) / len(ys)
        x_left   = min(xs)
        items.append({'y_center': y_center, 'x_left': x_left, 'text': text.strip()})

    line_groups = []
    for item in items:
        placed = False
        for group in line_groups:
            avg_y = sum(g['y_center'] for g in group) / len(group)
            if abs(item['y_center'] - avg_y) <= config.OCR_LINE_Y_THRESHOLD:
                group.append(item)
                placed = True
                break
        if not placed:
            line_groups.append([item])

    line_groups.sort(key=lambda g: sum(item['y_center'] for item in g) / len(g))

    result_lines = []
    for group in line_groups:
        sorted_group = sorted(group, key=lambda item: item['x_left'])
        line_text = ' '.join(item['text'] for item in sorted_group).strip()
        if line_text:
            result_lines.append(line_text)

    return result_lines


class OCREngine:
    def __init__(self, languages=None):
        if languages is None:
            languages = ['en']
        print("[OCR Engine] Loading EasyOCR models...")
        self.reader = easyocr.Reader(languages, gpu=False)
        self._confirm_buf = collections.deque(maxlen=config.OCR_CONFIRMATION_FRAMES)
        self._saved_lines = collections.deque(maxlen=200)
        self.latest_candidate = ""
        self.last_diagram_save_time = 0.0

    def clear_confirm_buffer(self):
        self._confirm_buf.clear()
        self.latest_candidate = ""

    def clear_saved_lines(self):
        self._saved_lines.clear()

    def capture_snapshot(self, frame, whiteboard_roi):
        x1, y1, x2, y2 = whiteboard_roi
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(fw, int(x2)), min(fh, int(y2))
        roi_img = frame[y1:y2, x1:x2]

        if roi_img.size == 0:
            return None, None

        if not os.path.exists(config.SNAPSHOTS_DIR):
            os.makedirs(config.SNAPSHOTS_DIR)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        color_path = os.path.join(config.SNAPSHOTS_DIR, f"{timestamp}.png")
        bw_path = os.path.join(config.SNAPSHOTS_DIR, f"{timestamp}_cleaned.png")

        cv2.imwrite(color_path, roi_img)

        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_img = clahe.apply(gray)
        _, thresh = cv2.threshold(cl_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(bw_path, thresh)

        return color_path, bw_path

    def _is_already_saved(self, line):
        return any(_similarity(line, saved) >= 0.92 for saved in self._saved_lines)

    def _confirmed_lines(self, candidate_lines):
        confirmed = []
        for line in candidate_lines:
            match_count = sum(
                1 for buf_lines in self._confirm_buf
                if any(_similarity(line, bl) >= 0.75 for bl in buf_lines)
            )
            if match_count >= config.OCR_CONFIRMATION_FRAMES:
                confirmed.append(line)
        return confirmed

    def process_frame(self, frame, whiteboard_roi, person_bboxes) -> OCRFrameResult:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        def make_result(text=None, msg="Error", roi_img=None, c_path=None, bw_path=None):
            return OCRFrameResult(
                text=text,
                status_msg=msg,
                candidate_text=self.latest_candidate,
                roi_image=roi_img,
                color_snapshot_path=c_path,
                cleaned_snapshot_path=bw_path,
                timestamp=timestamp
            )

        is_blocked = any(
            does_person_block_board(bbox, whiteboard_roi) for bbox in person_bboxes
        )
        if is_blocked:
            self._confirm_buf.clear()
            return make_result(msg="Whiteboard is blocked")

        x1, y1, x2, y2 = whiteboard_roi
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(fw, int(x2)), min(fh, int(y2))
        roi_img = frame[y1:y2, x1:x2]

        if roi_img.size == 0:
            return make_result(msg="Invalid ROI dimensions")

        if hasattr(config, 'OCR_MAX_WIDTH') and config.OCR_MAX_WIDTH:
            if roi_img.shape[1] > config.OCR_MAX_WIDTH:
                roi_img = resize_keep_aspect_ratio(roi_img, max_width=config.OCR_MAX_WIDTH)

        processed_roi = preprocess_for_ocr(roi_img)
        chosen_img = None

        if isinstance(processed_roi, dict):
            all_candidate_runs = []
            for key, img in processed_roi.items():
                try:
                    raw_results = self.reader.readtext(img, detail=1, paragraph=False)
                except Exception as exc:
                    print(f"[OCR Error] EasyOCR failed on variant '{key}': {exc}")
                    raw_results = []

                filtered = [
                    (bbox, text, conf) for bbox, text, conf in raw_results
                    if conf >= config.OCR_CONFIDENCE_THRESHOLD and not _is_garbage(text)
                ]

                candidate_lines = group_ocr_results_by_line(filtered)
                candidate_lines = [clean_ocr_text(l) for l in candidate_lines]
                candidate_lines = [l for l in candidate_lines if not _is_garbage(l)]

                if filtered:
                    avg_conf = sum(conf for _, _, conf in filtered) / len(filtered)
                    total_len = sum(len(text) for _, text, _ in filtered)
                    score = avg_conf * total_len
                else:
                    score = 0.0

                all_candidate_runs.append({
                    'score': score,
                    'candidate_lines': candidate_lines,
                    'filtered': filtered,
                    'img': img
                })

            all_candidate_runs.sort(key=lambda x: x['score'], reverse=True)
            best_run = all_candidate_runs[0]
            
            candidate_lines = best_run['candidate_lines']
            filtered = best_run['filtered']
            chosen_img = best_run['img']
        else:
            chosen_img = processed_roi
            try:
                raw_results = self.reader.readtext(chosen_img, detail=1, paragraph=False)
            except Exception as exc:
                print(f"[OCR Error] EasyOCR failed: {exc}")
                return make_result(msg="OCR processing error", roi_img=chosen_img)

            filtered = [
                (bbox, text, conf) for bbox, text, conf in raw_results
                if conf >= config.OCR_CONFIDENCE_THRESHOLD and not _is_garbage(text)
            ]

            candidate_lines = group_ocr_results_by_line(filtered)
            candidate_lines = [clean_ocr_text(l) for l in candidate_lines]
            candidate_lines = [l for l in candidate_lines if not _is_garbage(l)]

        if not filtered:
            self._confirm_buf.clear()
            self.latest_candidate = ""
            return make_result(msg="No text passed confidence/quality filter", roi_img=chosen_img)

        if not candidate_lines:
            self._confirm_buf.clear()
            self.latest_candidate = ""
            return make_result(msg="No valid lines after grouping", roi_img=chosen_img)

        self.latest_candidate = '\n'.join(candidate_lines)
        self._confirm_buf.append(candidate_lines)

        if len(self._confirm_buf) < config.OCR_CONFIRMATION_FRAMES:
            return make_result(msg=f"Buffering confirmation ({len(self._confirm_buf)}/{config.OCR_CONFIRMATION_FRAMES})", roi_img=chosen_img)

        confirmed = self._confirmed_lines(candidate_lines)
        if not confirmed:
            return make_result(msg="Lines not yet confirmed across frames", roi_img=chosen_img)

        new_lines = []
        is_diagram = any(is_diagram_caption(l) for l in confirmed)
        current_time = time.time()
        
        bypass_dedupe = is_diagram and (current_time - self.last_diagram_save_time > 15.0)
        
        for l in confirmed:
            if bypass_dedupe or not self._is_already_saved(l):
                new_lines.append(l)

        if not new_lines:
            return make_result(msg="All confirmed lines already saved", roi_img=chosen_img)

        if is_diagram and bypass_dedupe:
            self.last_diagram_save_time = current_time
        elif is_diagram and new_lines:
            self.last_diagram_save_time = current_time

        self._saved_lines.extend(new_lines)
        self._confirm_buf.clear()
        
        color_path, bw_path = self.capture_snapshot(frame, whiteboard_roi)
        return make_result(
            text='\n'.join(new_lines), 
            msg="Success", 
            roi_img=chosen_img, 
            c_path=color_path, 
            bw_path=bw_path
        )

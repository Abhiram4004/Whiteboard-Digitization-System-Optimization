import time
from ocr_engine import OCREngine
from main import SharedState, YOLOTrackerThread, OCRWorkerThread
import cv2
import numpy as np
import config

def test():
    # Make a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Figure 6.7", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    state = SharedState()
    # Force stability
    state.is_stable = True
    state.is_blocked = False
    
    ocr_thread = OCRWorkerThread(state)
    
    # Simulate YOLO thread triggering OCR multiple times to fill buffer
    for i in range(5):
        print(f"Triggering OCR pass {i+1}...")
        with state.lock:
            state.ocr_request_frame = frame.copy()
            state.ocr_request_metrics = []
            state.active_ocr_job_id = i
            state.ocr_request_trigger.set()
            
        time.sleep(2.0) # wait for OCR
        
        print(f"State after pass {i+1}:")
        print(f"  Note Entries: {len(state.note_entries)}")
        print(f"  OCR Busy: {state.ocr_busy}")
        print(f"  Candidate: {state.candidate_text}")
        print(f"  Status: {state.status_msg}")
        
    ocr_thread.stop()

if __name__ == "__main__":
    test()

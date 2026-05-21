from ultralytics import YOLO
import cv2
import collections
from utils import calculate_center, calculate_distance
import config

import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

_original_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_load


class ActivityTracker:
    def __init__(self, model_path='yolov8n.pt'):
        # Load the YOLOv8 model for tracking (default: n for CPU performance)
        self.model = YOLO(model_path)
        self.movement_threshold = config.MOVEMENT_THRESHOLD
        self.smoothing_window = config.SMOOTHING_WINDOW
        
        # Store previous centers for each track_id to calculate instantaneous movement
        self.previous_centers = {}
        # Store a rolling history of movements for each track_id for smoothing
        self.movement_history = collections.defaultdict(lambda: collections.deque(maxlen=self.smoothing_window))
        
    def track_frame(self, frame):
        """
        Process a single frame for tracking.
        Returns the annotated frame, a list of metrics for each person, and scene_movement.
        """
        annotated_frame = frame.copy()
        metrics = []
        scene_movement = 0.0
        
        try:
            # Run tracking. persist=True tells YOLO to track across frames
            results = self.model.track(frame, persist=True, classes=[0], verbose=False)
        except Exception as e:
            print(f"[Tracker Error] YOLO inference failed: {e}")
            return annotated_frame, metrics, scene_movement
            
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            total_movement = 0.0
            
            for box, track_id in zip(boxes, track_ids):
                # Calculate center
                center = calculate_center(box)
                inst_movement = 0.0
                
                # Calculate instantaneous movement
                if track_id in self.previous_centers:
                    prev_center = self.previous_centers[track_id]
                    inst_movement = calculate_distance(center, prev_center)
                
                self.previous_centers[track_id] = center
                
                # Add to history and calculate smoothed movement
                self.movement_history[track_id].append(inst_movement)
                history = self.movement_history[track_id]
                smoothed_movement = sum(history) / len(history) if history else 0.0
                
                total_movement += smoothed_movement
                
                # Active teaching logic based on smoothed movement
                status = "Active" if smoothed_movement > self.movement_threshold else "Idle"
                
                metrics.append({
                    'person_id': track_id,
                    'bbox': box,
                    'movement': round(smoothed_movement, 2),
                    'status': status
                })
                
                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if status == "Active" else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID: {track_id} | {status} | Mov: {smoothed_movement:.1f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Scene movement is average of all tracked persons' smoothed movement
            if track_ids:
                scene_movement = total_movement / len(track_ids)

        return annotated_frame, metrics, scene_movement

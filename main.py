import cv2
import argparse
import time
import os
import threading
import pandas as pd
from datetime import datetime
import json
import subprocess
import platform
import re

import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk

import config
from utils import does_person_block_board, get_absolute_roi, resize_keep_aspect_ratio
from tracker import ActivityTracker
from ocr_engine import OCREngine, OCRFrameResult


def save_local_settings():
    try:
        settings = {
            "WHITEBOARD_ROI": config.WHITEBOARD_ROI,
            "OCR_PREPROCESSING_MODE": config.OCR_PREPROCESSING_MODE
        }
        with open("local_settings.json", "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        print("[Config] Saved to local_settings.json")
    except Exception as e:
        print(f"[Warning] Failed to save local_settings.json: {e}")


def open_file_safe(path):
    try:
        if platform.system() == 'Windows':
            os.startfile(path)
        elif platform.system() == 'Darwin':
            subprocess.run(['open', path])
        else:
            subprocess.run(['xdg-open', path])
    except Exception as e:
        print(f"Failed to open image: {e}")


def is_diagram_caption(text):
    if not text:
        return True
    text = text.strip()
    if len(text) < 25:
        return True
    if re.match(r'^(figure|fig\.?|diagram)\s*[\d\.]+', text, re.IGNORECASE):
        return True
    return False


class ThreadedCamera:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if getattr(config, 'ROTATE_180', False):
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                if getattr(config, 'FLIP_HORIZONTAL', False):
                    frame = cv2.flip(frame, 1)
                self.ret = ret
                self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()


def setup_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)
    if not os.path.exists(config.ACTIVITY_FILE):
        df = pd.DataFrame(columns=['timestamp', 'person_id', 'movement', 'status'])
        df.to_csv(config.ACTIVITY_FILE, index=False)


def flush_log_buffer(buffer):
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    df.to_csv(config.ACTIVITY_FILE, mode='a', header=False, index=False)
    buffer.clear()


def log_notes_file(block):
    if not block: return
    with open(config.NOTES_FILE, 'a', encoding='utf-8') as f:
        f.write(block)


class SharedState:
    def __init__(self):
        self.lock = threading.RLock()
        self.metrics = []
        self.scene_movement = 0.0
        self.activity_buffer = []
        
        self.debug_ocr_frame = None
        self.candidate_text = ""
        self.status_msg = "Initializing..."
        self.is_stable = False
        self.is_blocked = False
        self.ocr_busy = False
        
        self.yolo_time = 0.0
        self.ocr_time = 0.0
        
        self.ocr_request_frame = None
        self.ocr_request_metrics = []
        self.ocr_request_trigger = threading.Event()
        self.manual_snapshot_trigger = threading.Event()
        
        # New Atomic State
        self.note_entries = []
        self.latest_snapshot_path = None
        self.active_ocr_job_id = None
        
    def add_note_entry(self, text, color_path, bw_path, timestamp, source):
        with self.lock:
            final_text = text if text else ""
            if is_diagram_caption(final_text):
                final_text += "\n[Diagram/sketch captured - see snapshot]" if final_text else "[Diagram/sketch captured - see snapshot]"
                final_text = final_text.strip()
                
            entry = {
                "timestamp": timestamp,
                "text": final_text,
                "color_snapshot_path": color_path,
                "cleaned_snapshot_path": bw_path,
                "source": source
            }
            self.note_entries.append(entry)
            if color_path:
                self.latest_snapshot_path = color_path
                
            # Log pairing
            print(f"[{timestamp}] Note Accepted ({source}):")
            print(f"Text:\n{final_text}")
            print(f"Color: {color_path}")
            print(f"Clean: {bw_path}")
            print("-" * 40)
            
            # Save to plain text file as backup
            block = f"[{timestamp}]\n{final_text}\n"
            if color_path: block += f"Color: {color_path}\n"
            if bw_path: block += f"Clean: {bw_path}\n"
            block += "\n"
            log_notes_file(block)


def draw_video_overlay(frame, metrics, scene_movement, is_stable, is_blocked, ocr_busy, fps, yolo_t, ocr_t):
    display_frame = cv2.resize(frame, config.RESIZE_DIM)
    if is_blocked:
        status_text = "Status: Teacher Blocking Board"
        status_color = (0, 0, 255)
    elif ocr_busy:
        status_text = "Status: OCR PROCESSING"
        status_color = (255, 128, 0)
    elif not is_stable:
        status_text = "Status: Writing / Unstable"
        status_color = (0, 255, 255)
    else:
        status_text = "Status: Stable"
        status_color = (0, 255, 0)
        
    cv2.putText(display_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    for m in metrics:
        x1, y1, x2, y2 = map(int, m['bbox'])
        color = (0, 255, 0) if m['status'] == "Active" else (0, 0, 255)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, f"ID:{m['person_id']} {m['status']}", (x1, max(10, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
    rx1, ry1, rx2, ry2 = config.WHITEBOARD_ROI
    if all(0.0 <= val <= 1.0 for val in config.WHITEBOARD_ROI):
        rx1 = int(rx1 * config.RESIZE_DIM[0])
        ry1 = int(ry1 * config.RESIZE_DIM[1])
        rx2 = int(rx2 * config.RESIZE_DIM[0])
        ry2 = int(ry2 * config.RESIZE_DIM[1])
    else:
        scale_x = config.RESIZE_DIM[0] / 640.0
        scale_y = config.RESIZE_DIM[1] / 480.0
        rx1, ry1, rx2, ry2 = int(rx1*scale_x), int(ry1*scale_y), int(rx2*scale_x), int(ry2*scale_y)
        
    cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
    cv2.putText(display_frame, "ROI", (rx1, max(10, ry1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display_frame, f"YOLO: {yolo_t*1000:.0f}ms", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display_frame, f"OCR: {ocr_t*1000:.0f}ms", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
    return display_frame


class YOLOTrackerThread:
    def __init__(self, cap, shared_state):
        self.cap = cap
        self.shared_state = shared_state
        print("Initializing YOLO tracker...")
        self.tracker = ActivityTracker()
        self.running = True
        self.last_ocr_time = 0.0
        self.stable_start_time = None
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
                
            frame_count += 1
            if frame_count % config.FRAME_SKIP != 0:
                time.sleep(0.01)
                continue
                
            start_time = time.time()
            process_frame = cv2.resize(frame, config.RESIZE_DIM)
            _, metrics, scene_movement = self.tracker.track_frame(process_frame)
            if not metrics:
                scene_movement = 0.0

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_logs = [{'timestamp': timestamp, 'person_id': m['person_id'], 'movement': m['movement'], 'status': m['status']} for m in metrics]
            
            is_blocked = any(does_person_block_board(m['bbox'], config.WHITEBOARD_ROI) for m in metrics)
            is_stable = (scene_movement < config.STABILITY_THRESHOLD) and (not is_blocked)
            
            yolo_duration = time.time() - start_time

            with self.shared_state.lock:
                self.shared_state.metrics = metrics
                self.shared_state.scene_movement = scene_movement
                self.shared_state.is_stable = is_stable
                self.shared_state.is_blocked = is_blocked
                self.shared_state.yolo_time = yolo_duration
                self.shared_state.activity_buffer.extend(new_logs)
                if len(self.shared_state.activity_buffer) >= config.LOG_BUFFER_SIZE:
                    flush_log_buffer(self.shared_state.activity_buffer)
                ocr_busy = self.shared_state.ocr_busy
            
            current_time = time.time()
            if not is_stable:
                self.stable_start_time = None
                msg = "Teacher blocking" if is_blocked else "Scene unstable"
                with self.shared_state.lock:
                    self.shared_state.status_msg = msg
            else:
                if self.stable_start_time is None:
                    self.stable_start_time = current_time
                
                stable_duration = current_time - self.stable_start_time
                time_since_ocr = current_time - self.last_ocr_time
                
                if ocr_busy:
                    with self.shared_state.lock:
                        self.shared_state.status_msg = "OCR Working..."
                elif time_since_ocr < config.OCR_COOLDOWN:
                    with self.shared_state.lock:
                        self.shared_state.status_msg = f"Cooldown ({config.OCR_COOLDOWN - time_since_ocr:.1f}s)"
                elif stable_duration < config.BOARD_STABLE_SECONDS:
                    with self.shared_state.lock:
                        self.shared_state.status_msg = f"Stabilizing ({stable_duration:.1f}/{config.BOARD_STABLE_SECONDS}s)"
                else:
                    with self.shared_state.lock:
                        self.shared_state.status_msg = "Triggering OCR..."
                        self.shared_state.ocr_request_frame = frame.copy()
                        self.shared_state.ocr_request_metrics = metrics.copy()
                        self.shared_state.active_ocr_job_id = int(time.time() * 1000)
                        self.shared_state.ocr_request_trigger.set()
                    self.last_ocr_time = current_time

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()


class OCRWorkerThread:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        print("Initializing OCR Engine...")
        self.ocr = OCREngine()
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        
    def run(self):
        while self.running:
            triggered = self.shared_state.ocr_request_trigger.wait(timeout=0.5)
            manual = self.shared_state.manual_snapshot_trigger.is_set()
            
            if not triggered and not manual:
                continue
                
            with self.shared_state.lock:
                self.shared_state.ocr_busy = True
                frame = self.shared_state.ocr_request_frame
                metrics = self.shared_state.ocr_request_metrics
                job_id = self.shared_state.active_ocr_job_id
                
                self.shared_state.ocr_request_trigger.clear()
                self.shared_state.manual_snapshot_trigger.clear()
                
            if frame is None:
                with self.shared_state.lock:
                    self.shared_state.ocr_busy = False
                continue
                
            start_time = time.time()
            orig_h, orig_w = frame.shape[:2]
            high_res_roi = get_absolute_roi(config.WHITEBOARD_ROI, orig_w, orig_h)
            
            proc_w, proc_h = config.RESIZE_DIM
            scale_x = orig_w / proc_w
            scale_y = orig_h / proc_h
            high_res_bboxes = []
            for m in metrics:
                b = m['bbox']
                high_res_bboxes.append([int(b[0]*scale_x), int(b[1]*scale_y), int(b[2]*scale_x), int(b[3]*scale_y)])

            # process_frame now returns a strict OCRFrameResult dataclass
            res: OCRFrameResult = self.ocr.process_frame(frame, high_res_roi, high_res_bboxes)
            
            if manual:
                if not res.color_snapshot_path:
                    color_path, bw_path = self.ocr.capture_snapshot(frame, high_res_roi)
                    res.color_snapshot_path = color_path
                    res.cleaned_snapshot_path = bw_path
                res.status_msg = "Manual snapshot captured."
                
            ocr_duration = time.time() - start_time
            
            with self.shared_state.lock:
                # Discard stale OCR results
                if not manual and self.shared_state.active_ocr_job_id != job_id:
                    self.shared_state.ocr_busy = False
                    continue

                if res.text or (manual and res.color_snapshot_path):
                    self.shared_state.add_note_entry(
                        text=res.text,
                        color_path=res.color_snapshot_path,
                        bw_path=res.cleaned_snapshot_path,
                        timestamp=res.timestamp,
                        source="MANUAL" if manual else "OCR"
                    )

                if res.roi_image is not None:
                    self.shared_state.debug_ocr_frame = res.roi_image
                self.shared_state.candidate_text = res.candidate_text
                self.shared_state.status_msg = res.status_msg
                self.shared_state.ocr_time = ocr_duration
                self.shared_state.ocr_busy = False
                
    def clear_buffers(self):
        self.ocr.clear_confirm_buffer()
        self.ocr.clear_saved_lines()
        
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()


class WhiteboardDigitizerUI:
    def __init__(self, cap, shared_state, yolo_thread, ocr_thread):
        self.cap = cap
        self.shared_state = shared_state
        self.yolo_thread = yolo_thread
        self.ocr_thread = ocr_thread
        
        self.root = tk.Tk()
        self.root.title("Whiteboard Content Digitization & Activity Monitoring")
        self.root.configure(bg='#121212')
        self.root.geometry("1280x820")
        
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)
        
        self.note_images = []
        self.latest_snapshot_photo = None
        self.current_snapshot_path = None
        self.rendered_note_count = 0
        
        self.create_widgets()
        
        self.last_ui_time = time.time()
        self.ui_fps = 0.0
        
        self.update_loop()
        
    def create_widgets(self):
        # LEFT PANEL
        self.left_panel = tk.Frame(self.root, bg='#121212', padx=15, pady=15)
        self.left_panel.grid(row=0, column=0, sticky='nsew')
        self.left_panel.rowconfigure(1, weight=3)
        self.left_panel.rowconfigure(3, weight=2)
        self.left_panel.columnconfigure(0, weight=1)
        
        tk.Label(self.left_panel, text="LIVE SYSTEM FEED", font=('Segoe UI', 14, 'bold'), bg='#121212', fg='#1a73e8').grid(row=0, column=0, sticky='w', pady=(0, 10))
        self.video_canvas = tk.Canvas(self.left_panel, bg='#1e1e1e', highlightthickness=1, highlightbackground='#2c2c2c')
        self.video_canvas.grid(row=1, column=0, sticky='nsew')
        
        debug_ctrl_frame = tk.Frame(self.left_panel, bg='#121212', pady=10)
        debug_ctrl_frame.grid(row=2, column=0, sticky='ew')
        
        self.show_debug_var = tk.BooleanVar(value=True)
        tk.Checkbutton(debug_ctrl_frame, text="Show Preprocessed OCR Debug", variable=self.show_debug_var, bg='#121212', fg='#e8eaed', selectcolor='#121212', font=('Segoe UI', 9)).pack(side='left')
        self.ocr_mode_var = tk.StringVar(value=config.OCR_PREPROCESSING_MODE)
        tk.Label(debug_ctrl_frame, text=" | OCR Mode:", bg='#121212', fg='#9aa0a6').pack(side='left', padx=(10, 5))
        tk.Radiobutton(debug_ctrl_frame, text="Fast", variable=self.ocr_mode_var, value="fast", command=self.update_ocr_mode, bg='#121212', fg='#e8eaed', selectcolor='#121212').pack(side='left')
        tk.Radiobutton(debug_ctrl_frame, text="Accurate (Multi)", variable=self.ocr_mode_var, value="multi", command=self.update_ocr_mode, bg='#121212', fg='#e8eaed', selectcolor='#121212').pack(side='left')
        
        self.debug_frame = tk.Frame(self.left_panel, bg='#1e1e1e', bd=1, relief='solid')
        self.debug_frame.grid(row=3, column=0, sticky='nsew', pady=(5, 0))
        self.debug_frame.columnconfigure(0, weight=1)
        self.debug_frame.rowconfigure(1, weight=1)
        tk.Label(self.debug_frame, text="OCR Preprocessing Preview", font=('Segoe UI', 10, 'bold'), bg='#1e1e1e', fg='#9aa0a6').grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.debug_canvas = tk.Canvas(self.debug_frame, bg='#121212', highlightthickness=0)
        self.debug_canvas.grid(row=1, column=0, sticky='nsew', padx=10, pady=(0, 10))
        
        # RIGHT PANEL
        self.right_panel = tk.Frame(self.root, bg='#1e1e1e', padx=20, pady=20)
        self.right_panel.grid(row=0, column=1, sticky='nsew')
        self.right_panel.rowconfigure(2, weight=3)
        self.right_panel.rowconfigure(3, weight=2)
        self.right_panel.columnconfigure(0, weight=1)
        
        status_card = tk.Frame(self.right_panel, bg='#121212', padx=15, pady=15, bd=1, relief='solid')
        status_card.grid(row=0, column=0, sticky='ew', pady=(0, 15))
        status_card.columnconfigure(1, weight=1)
        tk.Label(status_card, text="Board Status:", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#9aa0a6').grid(row=0, column=0, sticky='w')
        self.status_lbl = tk.Label(status_card, text="INITIALIZING", font=('Segoe UI', 12, 'bold'), bg='#121212', fg='#ffffff')
        self.status_lbl.grid(row=0, column=1, sticky='e')
        tk.Label(status_card, text="OCR Activity:", font=('Segoe UI', 11, 'bold'), bg='#121212', fg='#9aa0a6').grid(row=1, column=0, sticky='w', pady=(8, 0))
        self.ocr_msg_lbl = tk.Label(status_card, text="Idle", font=('Segoe UI', 10, 'italic'), bg='#121212', fg='#1a73e8')
        self.ocr_msg_lbl.grid(row=1, column=1, sticky='e', pady=(8, 0))
        
        cand_card = tk.Frame(self.right_panel, bg='#121212', padx=15, pady=12, bd=1, relief='solid')
        cand_card.grid(row=1, column=0, sticky='ew', pady=(0, 15))
        tk.Label(cand_card, text="Latest Candidate (Unconfirmed):", font=('Segoe UI', 10, 'bold'), bg='#121212', fg='#1a73e8').pack(anchor='w')
        self.cand_txt_lbl = tk.Label(cand_card, text="No candidate text yet.", font=('Consolas', 10), bg='#121212', fg='#e8eaed', justify='left', anchor='w')
        self.cand_txt_lbl.pack(fill='x', pady=(6, 0))
        
        notes_frame = tk.Frame(self.right_panel, bg='#1e1e1e')
        notes_frame.grid(row=2, column=0, sticky='nsew')
        notes_frame.rowconfigure(1, weight=1)
        notes_frame.columnconfigure(0, weight=1)
        tk.Label(notes_frame, text="Confirmed Notes:", font=('Segoe UI', 11, 'bold'), bg='#1e1e1e', fg='#e8eaed').grid(row=0, column=0, sticky='w', pady=(0, 5))
        self.notes_box = ScrolledText(notes_frame, wrap='word', bg='#121212', fg='#e8eaed', insertbackground='#ffffff', font=('Consolas', 11), highlightthickness=1)
        self.notes_box.grid(row=1, column=0, sticky='nsew')
                
        snapshot_frame = tk.Frame(self.right_panel, bg='#1e1e1e')
        snapshot_frame.grid(row=3, column=0, sticky='nsew', pady=(15, 0))
        snapshot_frame.rowconfigure(1, weight=1)
        snapshot_frame.columnconfigure(0, weight=1)
        tk.Label(snapshot_frame, text="Latest Board Snapshot (Diagrams):", font=('Segoe UI', 11, 'bold'), bg='#1e1e1e', fg='#e8eaed').grid(row=0, column=0, sticky='w')
        self.snap_canvas = tk.Canvas(snapshot_frame, bg='#121212', highlightthickness=1, highlightbackground='#2c2c2c')
        self.snap_canvas.grid(row=1, column=0, sticky='nsew', pady=(5, 0))
        
        btn_frame = tk.Frame(self.right_panel, bg='#1e1e1e', pady=10)
        btn_frame.grid(row=4, column=0, sticky='ew')
        btn_frame.columnconfigure((0,1,2), weight=1)
        tk.Button(btn_frame, text="Manual Snapshot", command=self.manual_snapshot, bg='#f39c12', fg='white', relief='flat', font=('Segoe UI', 10, 'bold')).grid(row=0, column=0, sticky='ew', padx=(0,5))
        tk.Button(btn_frame, text="Export HTML", command=self.save_notes, bg='#1a73e8', fg='white', relief='flat', font=('Segoe UI', 10, 'bold')).grid(row=0, column=1, sticky='ew', padx=5)
        tk.Button(btn_frame, text="Clear", command=self.clear_notes, bg='#c5221f', fg='white', relief='flat', font=('Segoe UI', 10, 'bold')).grid(row=0, column=2, sticky='ew', padx=(5,0))
        
        roi_btn = tk.Button(self.right_panel, text="Adjust ROI", command=self.open_roi_dialog, bg='#34495e', fg='white', font=('Segoe UI', 9, 'bold'), relief='flat')
        roi_btn.grid(row=5, column=0, sticky='ew', pady=(10, 0))
        
    def update_ocr_mode(self):
        config.OCR_PREPROCESSING_MODE = self.ocr_mode_var.get()
        save_local_settings()
        
    def manual_snapshot(self):
        ret, frame = self.cap.read()
        if ret:
            with self.shared_state.lock:
                self.shared_state.ocr_request_frame = frame.copy()
                self.shared_state.manual_snapshot_trigger.set()
        messagebox.showinfo("Snapshot", "Snapshot capture triggered.")

    def open_roi_dialog(self):
        top = tk.Toplevel(self.root)
        top.title("Adjust ROI")
        top.geometry("300x200")
        top.configure(bg='#121212')
        tk.Label(top, text="Enter percentages (0.0 to 1.0)", bg='#121212', fg='white').grid(row=0, column=0, columnspan=2, pady=10)
        
        vars = []
        labels = ["X1", "Y1", "X2", "Y2"]
        for i, (l, v) in enumerate(zip(labels, config.WHITEBOARD_ROI)):
            tk.Label(top, text=l, bg='#121212', fg='white').grid(row=i+1, column=0)
            ent = tk.Entry(top)
            ent.insert(0, str(v))
            ent.grid(row=i+1, column=1, pady=2)
            vars.append(ent)
            
        def save():
            try:
                vals = [float(e.get()) for e in vars]
                if not (0.0 <= vals[0] < vals[2] <= 1.0 and 0.0 <= vals[1] < vals[3] <= 1.0):
                    raise ValueError("Ensure 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0")
                config.WHITEBOARD_ROI = vals
                save_local_settings()
                self.ocr_thread.clear_buffers()
                messagebox.showinfo("Success", "ROI Updated. OCR Buffer Cleared.")
                top.destroy()
            except Exception as e:
                messagebox.showerror("Error", str(e))
                
        tk.Button(top, text="Save", command=save, bg='#34a853', fg='white').grid(row=5, column=0, columnspan=2, pady=15)

    def save_notes(self):
        path = filedialog.asksaveasfilename(defaultextension=".html", initialfile="student_notes.html",
                                            filetypes=[("HTML Files", "*.html"), ("All Files", "*.*")])
        if not path:
            return
            
        html_content = [
            "<html><head><style>",
            "body{font-family:sans-serif; background:#f4f4f4; color:#333; padding:20px; max-width:800px; margin:auto;}",
            ".note{background:white; padding:15px; margin-bottom:15px; border-radius:5px; box-shadow:0 1px 3px rgba(0,0,0,0.1);}",
            "img{max-width:100%; border:1px solid #ddd; margin-top:10px; border-radius:4px;}",
            "</style></head><body>",
            "<h1>Whiteboard Notes</h1>"
        ]
        
        with self.shared_state.lock:
            entries = list(self.shared_state.note_entries)
            
        for note in entries:
            html_content.append('<div class="note">')
            html_content.append(f"<h3>[{note['timestamp']}]</h3>")
            
            if note['text']:
                html_content.append(f"<pre style='white-space: pre-wrap; font-family: monospace;'>{note['text']}</pre>")
                
            if note['color_snapshot_path']:
                rel_path = os.path.relpath(note['color_snapshot_path'], os.path.dirname(path))
                rel_path = rel_path.replace("\\", "/")
                html_content.append(f'<img src="{rel_path}" alt="Board Snapshot">')
                
                if note['cleaned_snapshot_path']:
                    c_rel = os.path.relpath(note['cleaned_snapshot_path'], os.path.dirname(path)).replace("\\", "/")
                    html_content.append(f'<p><a href="{c_rel}" target="_blank">View Cleaned (B/W) Snapshot</a></p>')
                
            html_content.append('</div>')
            
        html_content.append("</body></html>")
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("\n".join(html_content))
            messagebox.showinfo("Export Successful", f"Notes successfully exported to HTML:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save HTML: {e}")

    def clear_notes(self):
        if messagebox.askyesno("Clear", "Clear notes, snapshots, and OCR buffers?"):
            self.notes_box.delete("1.0", tk.END)
            self.cand_txt_lbl.configure(text="No candidate text yet.")
            
            with self.shared_state.lock:
                self.shared_state.note_entries.clear()
                self.shared_state.latest_snapshot_path = None
                
            self.note_images.clear()
            self.current_snapshot_path = None
            self.rendered_note_count = 0
            
            self.snap_canvas.delete("all")
            cw, ch = max(100, self.snap_canvas.winfo_width()), max(100, self.snap_canvas.winfo_height())
            self.snap_canvas.create_text(cw//2, ch//2, text="No snapshot captured yet", fill="gray")
            
            self.ocr_thread.clear_buffers()

    def _render_single_note(self, note):
        ts = note['timestamp']
        text = note['text']
        c_path = note['color_snapshot_path']
        
        self.notes_box.insert('end', f"[{ts}]\n")
        if text:
            self.notes_box.insert('end', f"{text}\n")
            
        if c_path and os.path.exists(c_path):
            try:
                pil_img = Image.open(c_path)
                # Larger thumbnails
                pil_img.thumbnail((500, 500), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil_img)
                self.note_images.append(photo)
                
                self.notes_box.image_create('end', image=photo)
                self.notes_box.insert('end', "\n")
                
                btn = tk.Button(self.notes_box, text="Open Full Image", bg='#34495e', fg='white', relief='flat', font=('Segoe UI', 9),
                                command=lambda p=c_path: open_file_safe(p))
                self.notes_box.window_create('end', window=btn)
                self.notes_box.insert('end', "\n")
            except Exception as e:
                self.notes_box.insert('end', f"[Error loading thumbnail: {e}]\n")
        
        self.notes_box.insert('end', "\n---\n")
        self.notes_box.see('end')

    def update_loop(self):
        current_time = time.time()
        self.ui_fps = 1.0 / (current_time - self.last_ui_time + 1e-6)
        self.last_ui_time = current_time
        
        ret, live_frame = self.cap.read()
        frame = live_frame.copy() if (ret and live_frame is not None) else None
        
        with self.shared_state.lock:
            debug_frame = self.shared_state.debug_ocr_frame.copy() if self.shared_state.debug_ocr_frame is not None else None
            metrics = self.shared_state.metrics.copy()
            scene_movement = self.shared_state.scene_movement
            is_stable = self.shared_state.is_stable
            is_blocked = self.shared_state.is_blocked
            ocr_msg = self.shared_state.status_msg
            candidate_text = self.shared_state.candidate_text
            ocr_busy = self.shared_state.ocr_busy
            yolo_t = self.shared_state.yolo_time
            ocr_t = self.shared_state.ocr_time
            
            # Read from exact data model
            entries_copy = list(self.shared_state.note_entries)
            latest_snap_path = self.shared_state.latest_snapshot_path

        # Render new UI notes iteratively
        while self.rendered_note_count < len(entries_copy):
            self._render_single_note(entries_copy[self.rendered_note_count])
            self.rendered_note_count += 1
            
        # Draw Latest Board Snapshot (Bottom Panel)
        if latest_snap_path != self.current_snapshot_path:
            self.current_snapshot_path = latest_snap_path
            cw, ch = max(100, self.snap_canvas.winfo_width()), max(100, self.snap_canvas.winfo_height())
            if self.current_snapshot_path:
                abs_path = os.path.abspath(self.current_snapshot_path)
                if os.path.exists(abs_path):
                    try:
                        pil_snap = Image.open(abs_path)
                        pil_snap.thumbnail((cw, ch), Image.Resampling.LANCZOS)
                        self.latest_snapshot_photo = ImageTk.PhotoImage(pil_snap)
                        
                        self.snap_canvas.delete("all")
                        self.snap_canvas.create_image(cw//2, ch//2, anchor='center', image=self.latest_snapshot_photo)
                    except Exception as e:
                        print(f"Failed to load snapshot {abs_path}: {e}")
                        self.snap_canvas.delete("all")
                        self.snap_canvas.create_text(cw//2, ch//2, text=f"Could not load snapshot\n{self.current_snapshot_path}", fill="red")
                else:
                    self.snap_canvas.delete("all")
                    self.snap_canvas.create_text(cw//2, ch//2, text="Snapshot file not found", fill="red")
            else:
                self.snap_canvas.delete("all")
                self.snap_canvas.create_text(cw//2, ch//2, text="No snapshot captured yet", fill="gray")

        # Draw Video Frame
        if frame is not None:
            overlay_frame = draw_video_overlay(frame, metrics, scene_movement, is_stable, is_blocked, ocr_busy, self.ui_fps, yolo_t, ocr_t)
            vw, vh = max(100, self.video_canvas.winfo_width()), max(100, self.video_canvas.winfo_height())
            overlay_frame = resize_keep_aspect_ratio(overlay_frame, max_width=vw, max_height=vh)
            rgb_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
            self.photo_img = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
            self.video_canvas.delete("all")
            self.video_canvas.create_image(vw//2, vh//2, anchor='center', image=self.photo_img)

        # Draw Debug Frame
        if self.show_debug_var.get() and debug_frame is not None:
            dcw, dch = max(50, self.debug_canvas.winfo_width()), max(50, self.debug_canvas.winfo_height())
            scaled_debug = resize_keep_aspect_ratio(debug_frame, max_width=dcw, max_height=dch)
            rgb_debug = cv2.cvtColor(scaled_debug, cv2.COLOR_GRAY2RGB) if len(scaled_debug.shape) == 2 else cv2.cvtColor(scaled_debug, cv2.COLOR_BGR2RGB)
            self.debug_photo_img = ImageTk.PhotoImage(image=Image.fromarray(rgb_debug))
            self.debug_canvas.delete("all")
            self.debug_canvas.create_image(dcw//2, dch//2, anchor='center', image=self.debug_photo_img)
        else:
            self.debug_canvas.delete("all")
            
        status_color = "#e74c3c" if is_blocked else "#f1c40f" if not is_stable else "#2ecc71"
        self.status_lbl.configure(text=("BLOCKED" if is_blocked else "UNSTABLE" if not is_stable else "STABLE"), fg=status_color)
        self.ocr_msg_lbl.configure(text=ocr_msg)
        self.cand_txt_lbl.configure(text=(candidate_text.strip() or "No candidate text yet."))
        
        self.root.after(33, self.update_loop)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=config.DEFAULT_VIDEO_SOURCE)
    args = parser.parse_args()

    setup_output_dir()
    
    source_val = int(args.source) if str(args.source).isdigit() else args.source
    cap = ThreadedCamera(source_val)
    if not cap.isOpened():
        print(f"[Error] Could not open video source {args.source}")
        return

    shared_state = SharedState()
    yolo_t = YOLOTrackerThread(cap, shared_state)
    ocr_t = OCRWorkerThread(shared_state)

    ui = WhiteboardDigitizerUI(cap, shared_state, yolo_t, ocr_t)
    ui.run()

    print("Cleaning up...")
    yolo_t.stop()
    ocr_t.stop()
    cap.release()
    if shared_state.activity_buffer:
        flush_log_buffer(shared_state.activity_buffer)

if __name__ == "__main__":
    main()

import os
import time
import shutil
import base64
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageGrab
import tkinter as tk

os.makedirs("report_assets", exist_ok=True)

print("Starting asset generation...")

# --- 1. Find snapshots for 06 and 08 ---
snap_dir = os.path.join("output", "board_snapshots")
snaps = []
if os.path.exists(snap_dir):
    snaps = [f for f in os.listdir(snap_dir) if f.endswith('.png') and not f.endswith('_cleaned.png')]
    snaps.sort()

if len(snaps) >= 1:
    shutil.copy(os.path.join(snap_dir, snaps[-1]), "report_assets/06_board_snapshot_diagram.png")
    print("Created 06_board_snapshot_diagram.png")

if len(snaps) >= 2:
    shutil.copy(os.path.join(snap_dir, snaps[-2]), "report_assets/08_board_snapshot_flowchart.png")
    print("Created 08_board_snapshot_flowchart.png")
elif len(snaps) == 1:
    shutil.copy(os.path.join(snap_dir, snaps[-1]), "report_assets/08_board_snapshot_flowchart.png")

# --- 2. 07_before_after_ocr.png ---
if len(snaps) >= 1:
    orig_img = Image.open(os.path.join(snap_dir, snaps[-1]))
    orig_img.thumbnail((600, 600))
    
    text_img = Image.new('RGB', (400, orig_img.height), color='#f0f0f0')
    d = ImageDraw.Draw(text_img)
    # Draw simple title and text
    d.text((20, 20), "BEFORE: Camera View", fill='#333333')
    d.text((orig_img.width + 20, 20), "AFTER: Extracted Notes", fill='#333333')
    
    # We will just write the text
    d.text((20, 60), "[OCR RESULT]\n\nFigure 6.7\n[Diagram/sketch captured - see snapshot]", fill='blue')
    
    combo = Image.new('RGB', (orig_img.width + 400 + 20, orig_img.height), color='white')
    combo.paste(orig_img, (0, 0))
    combo.paste(text_img, (orig_img.width + 20, 0))
    
    # Draw a line between
    d_combo = ImageDraw.Draw(combo)
    d_combo.line([(orig_img.width+10, 0), (orig_img.width+10, orig_img.height)], fill='gray', width=2)
    
    combo.save("report_assets/07_before_after_ocr.png")
    print("Created 07_before_after_ocr.png")

# --- 3. Mermaid API for 02 and 03 ---
def generate_mermaid_image(mermaid_text, filename):
    encoded = base64.urlsafe_b64encode(mermaid_text.encode('utf-8')).decode('utf-8')
    url = f"https://mermaid.ink/img/{encoded}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"report_assets/{filename}", "wb") as f:
                f.write(response.content)
            print(f"Created {filename}")
        else:
            print(f"Failed to generate {filename}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching mermaid {filename}: {e}")

arch_mmd = """
graph TD
    A[IP Camera/Webcam] --> B[Threaded Camera]
    B --> C[YOLOv8 Tracking tracker.py]
    C --> D[Stability Detection main.py]
    D --> E[OCR Worker main.py]
    E --> F[EasyOCR Text Extraction ocr_engine.py]
    F --> G[Snapshot Saving ocr_engine.py / utils.py]
    G --> H[Tkinter GUI main.py]
    H --> I[HTML Export main.py]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#dfd,stroke:#333,stroke-width:2px
"""
generate_mermaid_image(arch_mmd, "02_system_architecture.png")

flow_mmd = """
flowchart LR
    A[Camera Input] --> B[Frame Processing]
    B --> C[YOLO Tracking]
    C --> D{Is Stable?}
    D -- Yes --> E[OCR Trigger]
    D -- No --> C
    E --> F[Text Extraction]
    F --> G[Snapshot Saving]
    G --> H[GUI Display]
    H --> I[HTML Export]
"""
generate_mermaid_image(flow_mmd, "03_system_flowchart.png")

# --- 4. Mock HTML Export (04) ---
# Create an image that looks like an HTML page
html_img = Image.new('RGB', (800, 600), color='#f4f4f4')
hd = ImageDraw.Draw(html_img)
hd.rectangle([(20, 20), (780, 580)], fill='white', outline='#dddddd')
hd.text((40, 40), "Whiteboard Notes", fill='#333333')
hd.text((40, 80), "[2026-05-21 21:23:24]", fill='#666666')
hd.text((40, 110), "Figure 6.7\n[Diagram/sketch captured - see snapshot]", fill='black')
hd.rectangle([(40, 160), (440, 460)], fill='#eeeeee', outline='#cccccc')
hd.text((180, 300), "[ Board Snapshot Image Here ]", fill='#999999')
hd.text((40, 480), "View Cleaned (B/W) Snapshot", fill='blue')
html_img.save("report_assets/04_html_export_output.png")
print("Created 04_html_export_output.png")

# --- 5. Mock GUI for 01 and 05 ---
import main
import config

class DummyCap:
    def __init__(self, img_path):
        if img_path and os.path.exists(img_path):
            self.frame = cv2.imread(img_path)
            self.ret = True
        else:
            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.frame, "No camera feed", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            self.ret = True
    def read(self):
        return self.ret, self.frame
    def release(self):
        pass

class DummyThread:
    def stop(self):
        pass

state = main.SharedState()
state.status_msg = "STABLE"
state.is_stable = True
state.candidate_text = "Figure 6.7"
if len(snaps) >= 1:
    p = os.path.join(snap_dir, snaps[-1])
    state.add_note_entry("Figure 6.7\n[Diagram/sketch captured - see snapshot]", p, p.replace(".png", "_cleaned.png"), "2026-05-21 21:23:24", "OCR")
    
cap = DummyCap(os.path.join(snap_dir, snaps[-1]) if len(snaps) >= 1 else None)
ui = main.WhiteboardDigitizerUI(cap, state, DummyThread(), DummyThread())

def capture_ui():
    print("Rendering GUI...")
    ui.root.update()
    time.sleep(2)
    ui.root.update()
    
    x = ui.root.winfo_rootx()
    y = ui.root.winfo_rooty()
    w = ui.root.winfo_width()
    h = ui.root.winfo_height()
    
    img = ImageGrab.grab(bbox=(x, y, x+w, y+h))
    img.save("report_assets/01_live_gui_overview.png")
    print("Created 01_live_gui_overview.png")
    
    rx = ui.right_panel.winfo_rootx()
    ry = ui.right_panel.winfo_rooty()
    rw = ui.right_panel.winfo_width()
    rh = ui.right_panel.winfo_height()
    img_right = ImageGrab.grab(bbox=(rx, ry, rx+rw, ry+rh))
    img_right.save("report_assets/05_ocr_output_panel.png")
    print("Created 05_ocr_output_panel.png")
    
    ui.root.destroy()

ui.root.after(2000, capture_ui)
ui.root.mainloop()

print("All tasks complete.")

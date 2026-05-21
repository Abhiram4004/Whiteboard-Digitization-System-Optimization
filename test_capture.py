import tkinter as tk
from PIL import ImageGrab
import time
import os

root = tk.Tk()
root.title("Test Capture")
root.geometry("400x300")
lbl = tk.Label(root, text="Hello World!")
lbl.pack(expand=True)

def capture():
    root.update()
    x = root.winfo_rootx()
    y = root.winfo_rooty()
    w = root.winfo_width()
    h = root.winfo_height()
    print(f"Bbox: {x}, {y}, {w}, {h}")
    try:
        img = ImageGrab.grab(bbox=(x, y, x+w, y+h))
        os.makedirs("report_assets", exist_ok=True)
        img.save("report_assets/test_capture.png")
        print("Capture successful!")
    except Exception as e:
        print(f"Capture failed: {e}")
    root.destroy()

root.after(1000, capture)
root.mainloop()

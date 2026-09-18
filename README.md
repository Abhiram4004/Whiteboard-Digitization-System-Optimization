# Whiteboard-Digitization-System-Optimization
# Whiteboard Content Digitization & Activity Monitoring System

A real-time computer vision system that digitizes whiteboard content and monitors presenter activity using YOLOv8, OpenCV, PyTorch, and EasyOCR.

## Overview

The **Whiteboard Content Digitization & Activity Monitoring System** is a computer vision application designed to automatically capture and digitize content written on a physical whiteboard while monitoring presenter activity.

The system combines object detection, image processing, OCR, and real-time video analysis to identify the presenter, process relevant whiteboard regions, and extract written content as digital text.

The project supports image, snapshot, and video-based analysis through an interactive Streamlit interface, while the original desktop pipeline can be used for local camera/video workflows.

## Key Features

- Real-time whiteboard content processing
- Presenter detection using YOLOv8
- Presenter activity monitoring
- Whiteboard text extraction using EasyOCR
- OpenCV-based image and video processing
- Image preprocessing for improved OCR
- Multi-threaded processing for real-time workflows
- Image and video analysis
- OCR result visualization
- Streamlit web interface
- Processed result visualization and downloads

## System Architecture

```text
                    Image / Video / Camera
                              |
                              v
                     Frame Acquisition
                              |
                              v
                     OpenCV Processing
                              |
                              v
                       YOLOv8 Model
                              |
                    +---------+---------+
                    |                   |
                    v                   v
              Presenter             Whiteboard
              Detection               Region
                    |                   |
                    |                   v
                    |            Image Preprocessing
                    |                   |
                    |                   v
                    |               EasyOCR
                    |                   |
                    +---------+---------+
                              |
                              v
                    Activity & Content
                       Processing
                              |
                              v
                     Digitized Output

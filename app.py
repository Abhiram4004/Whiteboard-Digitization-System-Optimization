import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from tracker import ActivityTracker
from ocr_engine import OCREngine
from utils import get_absolute_roi


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Whiteboard Digitization System",
    page_icon="📝",
    layout="wide"
)


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

@st.cache_resource
def load_tracker():
    return ActivityTracker("yolov8n.pt")


@st.cache_resource
def load_ocr():
    return OCREngine(["en"])


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def prepare_frame(uploaded_image):
    """
    Convert uploaded Streamlit image to OpenCV BGR format.
    """
    image = Image.open(uploaded_image).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def process_image(frame, tracker, ocr_engine, roi):
    """
    Process a single image through YOLO + OCR.
    """

    # Resize to the same dimensions used by the original project
    processed_frame = cv2.resize(frame, (640, 480))

    # YOLO presenter detection
    annotated_frame, metrics, scene_movement = tracker.track_frame(
        processed_frame
    )

    # Convert percentage ROI to pixel coordinates
    roi_px = get_absolute_roi(
        roi,
        processed_frame.shape[1],
        processed_frame.shape[0]
    )

    # Get presenter bounding boxes
    person_bboxes = [
        metric["bbox"]
        for metric in metrics
    ]

    # OCR
    ocr_result = ocr_engine.process_frame(
        processed_frame,
        roi_px,
        person_bboxes
    )

    return (
        annotated_frame,
        roi_px,
        metrics,
        scene_movement,
        ocr_result
    )


def draw_roi(frame, roi):
    """
    Draw whiteboard ROI on an image.
    """
    output = frame.copy()

    x1, y1, x2, y2 = roi

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        output,
        "Whiteboard ROI",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    return output


def save_uploaded_video(uploaded_file):
    """
    Save uploaded Streamlit video to a temporary file.
    """
    suffix = os.path.splitext(uploaded_file.name)[1]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_file.write(uploaded_file.read())
    temp_file.close()

    return temp_file.name


def process_video(video_path, tracker, ocr_engine, roi, frame_skip=10):
    """
    Process an uploaded video.

    YOLO is run on selected frames and OCR is performed
    periodically to keep cloud processing practical.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, [], "Unable to open video."

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    processed_frames = []
    ocr_results = []
    activity_records = []

    frame_number = 0
    last_ocr_frame = -frame_skip

    progress = st.progress(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Process only selected frames
        if frame_number % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 480))

        # YOLO tracking
        annotated_frame, metrics, scene_movement = tracker.track_frame(
            frame
        )

        # Whiteboard ROI
        roi_px = get_absolute_roi(
            roi,
            frame.shape[1],
            frame.shape[0]
        )

        # Draw ROI
        annotated_frame = draw_roi(
            annotated_frame,
            roi_px
        )

        # Activity records
        for metric in metrics:

            activity_records.append({
                "frame": frame_number,
                "person_id": metric["person_id"],
                "movement": metric["movement"],
                "status": metric["status"]
            })

        # OCR periodically
        if frame_number - last_ocr_frame >= frame_skip * 3:

            person_bboxes = [
                metric["bbox"]
                for metric in metrics
            ]

            try:
                ocr_result = ocr_engine.process_frame(
                    frame,
                    roi_px,
                    person_bboxes
                )

                if ocr_result.text:

                    ocr_results.append({
                        "frame": frame_number,
                        "timestamp": frame_number / fps,
                        "text": ocr_result.text
                    })

                last_ocr_frame = frame_number

            except Exception as e:

                ocr_results.append({
                    "frame": frame_number,
                    "timestamp": frame_number / fps,
                    "text": f"OCR error: {e}"
                })

        processed_frames.append(
            annotated_frame
        )

        if total_frames > 0:

            progress.progress(
                min(
                    frame_number / total_frames,
                    1.0
                )
            )

    cap.release()
    progress.empty()

    return (
        processed_frames,
        ocr_results,
        activity_records
    )


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("📝 Whiteboard Content Digitization & Activity Monitoring")

st.markdown(
    """
    A computer vision system for whiteboard digitization and presenter
    activity monitoring using **YOLOv8, OpenCV, PyTorch and EasyOCR**.
    """
)

st.divider()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙️ Configuration")

mode = st.sidebar.radio(
    "Select Analysis Mode",
    [
        "Image / Snapshot",
        "Video Analysis"
    ]
)

st.sidebar.subheader("Whiteboard Region")

roi_x1 = st.sidebar.slider(
    "Left",
    0.0,
    1.0,
    0.05,
    0.01
)

roi_y1 = st.sidebar.slider(
    "Top",
    0.0,
    1.0,
    0.08,
    0.01
)

roi_x2 = st.sidebar.slider(
    "Right",
    0.0,
    1.0,
    0.95,
    0.01
)

roi_y2 = st.sidebar.slider(
    "Bottom",
    0.0,
    1.0,
    0.88,
    0.01
)

WHITEBOARD_ROI = [
    roi_x1,
    roi_y1,
    roi_x2,
    roi_y2
]


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

with st.spinner("Loading YOLOv8 model..."):
    tracker = load_tracker()

with st.spinner("Loading EasyOCR model..."):
    ocr_engine = load_ocr()


# ---------------------------------------------------------
# IMAGE / SNAPSHOT MODE
# ---------------------------------------------------------

if mode == "Image / Snapshot":

    st.header("Image / Snapshot Analysis")

    input_type = st.radio(
        "Choose Input",
        [
            "Upload Image",
            "Camera Snapshot"
        ],
        horizontal=True
    )

    uploaded_image = None

    if input_type == "Upload Image":

        uploaded_image = st.file_uploader(
            "Upload a whiteboard image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

    else:

        uploaded_image = st.camera_input(
            "Capture Whiteboard Image"
        )

    if uploaded_image is not None:

        frame = prepare_frame(
            uploaded_image
        )

        st.subheader("Input Image")

        input_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            input_rgb,
            use_container_width=True
        )

        if st.button(
            "🔍 Analyze Whiteboard",
            type="primary"
        ):

            with st.spinner(
                "Running YOLOv8 detection and OCR..."
            ):

                (
                    annotated_frame,
                    roi_px,
                    metrics,
                    scene_movement,
                    ocr_result
                ) = process_image(
                    frame,
                    tracker,
                    ocr_engine,
                    WHITEBOARD_ROI
                )

            st.divider()

            # -------------------------------------------------
            # DETECTION RESULTS
            # -------------------------------------------------

            st.subheader("Detection Result")

            annotated_rgb = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            annotated_rgb = cv2.rectangle(
                annotated_rgb.copy(),
                (roi_px[0], roi_px[1]),
                (roi_px[2], roi_px[3]),
                (255, 0, 0),
                2
            )

            st.image(
                annotated_rgb,
                caption="YOLOv8 Presenter Detection",
                use_container_width=True
            )

            # -------------------------------------------------
            # METRICS
            # -------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Presenters Detected",
                    len(metrics)
                )

            with col2:
                st.metric(
                    "Scene Movement",
                    f"{scene_movement:.2f}"
                )

            with col3:

                if metrics:
                    active_count = sum(
                        1
                        for metric in metrics
                        if metric["status"] == "Active"
                    )
                else:
                    active_count = 0

                st.metric(
                    "Active Presenters",
                    active_count
                )

            # -------------------------------------------------
            # PRESENTER ACTIVITY
            # -------------------------------------------------

            if metrics:

                st.subheader(
                    "Presenter Activity"
                )

                for metric in metrics:

                    st.write(
                        f"**Person ID:** {metric['person_id']}  |  "
                        f"**Status:** {metric['status']}  |  "
                        f"**Movement:** {metric['movement']}"
                    )

            else:

                st.info(
                    "No presenter was detected in this frame."
                )

            # -------------------------------------------------
            # OCR RESULT
            # -------------------------------------------------

            st.subheader(
                "OCR Result"
            )

            if ocr_result.text:

                st.success(
                    "Text extracted successfully."
                )

                st.text_area(
                    "Extracted Whiteboard Text",
                    ocr_result.text,
                    height=250
                )

            else:

                st.info(
                    ocr_result.status_msg
                )

            # -------------------------------------------------
            # OCR PREVIEW
            # -------------------------------------------------

            if ocr_result.roi_image is not None:

                st.subheader(
                    "Processed Whiteboard Region"
                )

                processed_ocr = ocr_result.roi_image

                if len(processed_ocr.shape) == 2:

                    st.image(
                        processed_ocr,
                        caption="OCR Preprocessed Region",
                        use_container_width=True
                    )

                else:

                    processed_ocr_rgb = cv2.cvtColor(
                        processed_ocr,
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        processed_ocr_rgb,
                        caption="OCR Region",
                        use_container_width=True
                    )

            # -------------------------------------------------
            # DOWNLOAD OCR TEXT
            # -------------------------------------------------

            if ocr_result.text:

                st.download_button(
                    label="⬇️ Download Extracted Text",
                    data=ocr_result.text,
                    file_name="whiteboard_text.txt",
                    mime="text/plain"
                )


# ---------------------------------------------------------
# VIDEO MODE
# ---------------------------------------------------------

else:

    st.header("Video Analysis")

    uploaded_video = st.file_uploader(
        "Upload a whiteboard video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )

    frame_skip = st.slider(
        "Process every Nth frame",
        min_value=5,
        max_value=30,
        value=10
    )

    if uploaded_video is not None:

        st.video(
            uploaded_video
        )

        if st.button(
            "▶️ Analyze Video",
            type="primary"
        ):

            video_path = save_uploaded_video(
                uploaded_video
            )

            try:

                with st.spinner(
                    "Processing video..."
                ):

                    (
                        processed_frames,
                        ocr_results,
                        activity_records
                    ) = process_video(
                        video_path,
                        tracker,
                        ocr_engine,
                        WHITEBOARD_ROI,
                        frame_skip
                    )

                st.success(
                    "Video processing completed."
                )

                # -------------------------------------------------
                # VIDEO SUMMARY
                # -------------------------------------------------

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Processed Frames",
                        len(processed_frames)
                    )

                with col2:

                    st.metric(
                        "OCR Results",
                        len(ocr_results)
                    )

                with col3:

                    st.metric(
                        "Activity Records",
                        len(activity_records)
                    )

                # -------------------------------------------------
                # SAMPLE PROCESSED FRAMES
                # -------------------------------------------------

                if processed_frames:

                    st.subheader(
                        "Processed Frames"
                    )

                    sample_count = min(
                        6,
                        len(processed_frames)
                    )

                    indexes = np.linspace(
                        0,
                        len(processed_frames) - 1,
                        sample_count,
                        dtype=int
                    )

                    for index in indexes:

                        frame = processed_frames[index]

                        frame_rgb = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )

                        st.image(
                            frame_rgb,
                            caption=f"Processed Frame {index + 1}",
                            use_container_width=True
                        )

                # -------------------------------------------------
                # OCR RESULTS
                # -------------------------------------------------

                st.subheader(
                    "Extracted Whiteboard Content"
                )

                if ocr_results:

                    combined_text = ""

                    for result in ocr_results:

                        combined_text += (
                            f"[{result['timestamp']:.2f}s]\n"
                            f"{result['text']}\n\n"
                        )

                    st.text_area(
                        "OCR Output",
                        combined_text,
                        height=300
                    )

                    st.download_button(
                        label="⬇️ Download OCR Results",
                        data=combined_text,
                        file_name="whiteboard_ocr_results.txt",
                        mime="text/plain"
                    )

                else:

                    st.info(
                        "No confirmed OCR text was extracted from the processed frames."
                    )

                # -------------------------------------------------
                # ACTIVITY DATA
                # -------------------------------------------------

                if activity_records:

                    st.subheader(
                        "Presenter Activity"
                    )

                    import pandas as pd

                    activity_df = pd.DataFrame(
                        activity_records
                    )

                    st.dataframe(
                        activity_df,
                        use_container_width=True
                    )

                    csv_data = activity_df.to_csv(
                        index=False
                    )

                    st.download_button(
                        label="⬇️ Download Activity CSV",
                        data=csv_data,
                        file_name="activity.csv",
                        mime="text/csv"
                    )

            finally:

                if os.path.exists(video_path):

                    os.remove(
                        video_path
                    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Whiteboard Content Digitization & Activity Monitoring System | "
    "YOLOv8 • OpenCV • PyTorch • EasyOCR"
)

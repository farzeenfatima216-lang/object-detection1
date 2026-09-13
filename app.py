import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import torch
import csv
from pathlib import Path
from typing import Any
from ultralytics import YOLO
import torchvision
from torchvision.transforms import functional as F


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM DARK + MAROON UI
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #120708 0%, #1c0b0e 45%, #2a0d12 100%);
    color: #ffffff;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #160709 0%, #260b10 100%);
    border-right: 1px solid #64202c;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown strong {
    color: #FFFFFF !important;
}

.main-title {
    font-size: 46px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    color: #d9aeb5;
    font-size: 17px;
    margin-bottom: 35px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 20px;
    margin-bottom: 15px;
}

.metric-card {
    background: linear-gradient(145deg, #351219, #1d090d);
    border: 1px solid #672532;
    border-radius: 15px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(0,0,0,0.30);
}

.metric-title {
    color: #d8a8b0;
    font-size: 14px;
}

.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 800;
}

.info-box {
    background: #260d12;
    border-left: 5px solid #a52d43;
    border-radius: 10px;
    padding: 15px;
    margin-top: 15px;
}

div.stButton > button {
    background: linear-gradient(90deg, #7d2034, #a52d43);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATHS
# ============================================================

YOLOV8_PATH = "runs/detect/train/weights/best.pt"
SSD_PATH = "ssd_car_detection.pth"

PROJECT_DIR = Path(__file__).resolve().parent
YOLO_RESULTS_DIR = PROJECT_DIR / "runs" / "detect"
SSD_RESULTS_DIR = PROJECT_DIR / "ssd_results"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD YOLO MODELS
# ============================================================

@st.cache_resource
def load_yolov8():

    model = YOLO(YOLOV8_PATH)

    return model


def find_yolo_results_dir():

    result_dirs = [
        path
        for path in YOLO_RESULTS_DIR.glob("*")
        if path.is_dir() and (path / "results.csv").is_file()
    ]

    if not result_dirs:
        return None

    return max(
        result_dirs,
        key=lambda path: (path / "results.csv").stat().st_mtime
    )


def find_ssd_graph_dir():

    graph_files = list(SSD_RESULTS_DIR.rglob("*.png"))

    if not graph_files:
        return None

    return graph_files[0].parent


def load_yolo_metrics():

    results_dir = find_yolo_results_dir()

    if results_dir is None:
        return {}

    with (results_dir / "results.csv").open(newline="") as metrics_file:
        rows = list(csv.DictReader(metrics_file))

    if not rows:
        return {}

    last_row = rows[-1]
    metric_names = {
        "Precision": "metrics/precision(B)",
        "Recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)"
    }

    return {
        name: float(last_row[column])
        for name, column in metric_names.items()
        if last_row.get(column, "")
    }


def load_ssd_metrics():

    statistics_path = PROJECT_DIR / "ssd_detection_statistics.csv"

    if not statistics_path.is_file():
        return {}

    with statistics_path.open(newline="") as statistics_file:
        rows = list(csv.DictReader(statistics_file))

    if not rows:
        return {}

    detection_counts = [float(row["Detections"]) for row in rows]
    confidence_scores = [float(row["Average_Confidence"]) for row in rows]

    return {
        "Average detections": np.mean(detection_counts),
        "Average confidence": np.mean(confidence_scores)
    }


def show_saved_graphs(title, results_dir, include_all_images=False):

    st.markdown(f"### {title}")

    if results_dir is None:
        st.info("Graph not available.")
        return

    graph_paths = sorted(
        path
        for path in results_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and (include_all_images or (
            "curve" in path.name.lower()
            or "confusion" in path.name.lower()
            or path.name.lower() in {"results.png", "labels.jpg"}
        ))
    )

    if not graph_paths:
        st.info("Graph not available.")
        return

    graph_columns = st.columns(2)

    for index, graph_path in enumerate(graph_paths):

        with graph_columns[index % 2]:

            st.image(
                str(graph_path),
                caption=graph_path.name,
                use_container_width=True
            )


def show_detection_details(model_label, class_names, scores):

    st.markdown(f"**{model_label} detected classes:** " + (
        ", ".join(class_names) if class_names else "None"
    ))
    st.markdown(f"**{model_label} confidence scores:** " + (
        ", ".join(f"{score:.2f}" for score in scores)
        if len(scores) > 0 else "None"
    ))
    st.markdown(f"**{model_label} detections:** {len(class_names)}")


def show_model_images(original_image, detected_image):

    image_columns = st.columns(2)

    with image_columns[0]:

        st.markdown("**Original image**")
        st.image(original_image, use_container_width=True)

    with image_columns[1]:

        st.markdown("**Detected image**")
        st.image(detected_image, use_container_width=True)


# ============================================================
# LOAD SSD MODEL
# ============================================================

@st.cache_resource
def load_ssd():

    checkpoint = torch.load(
        SSD_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # If complete model was saved
    if isinstance(checkpoint, torch.nn.Module):

        model = checkpoint

    # If state_dict was saved
    else:

        model = torchvision.models.detection.ssd300_vgg16(
            weights=None,
            weights_backbone=None,
            num_classes=91
        )

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                checkpoint = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:

                checkpoint = checkpoint["state_dict"]

            model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# SSD DETECTION
# ============================================================

def run_ssd(model, image):

    image_tensor = F.to_tensor(image).to(DEVICE)

    with torch.no_grad():

        prediction = model([image_tensor])[0]

    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()

    return boxes, scores, labels


# ============================================================
# DRAW SSD BOXES
# ============================================================

def draw_ssd_boxes(image, boxes, scores, labels, threshold):

    result = image.copy()

    draw = ImageDraw.Draw(result)

    detected = 0

    for box, score, label in zip(boxes, scores, labels):

        if score < threshold:
            continue

        x1, y1, x2, y2 = box

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="#ff4d6d",
            width=4
        )

        text = f"Car {score:.2f}"

        draw.rectangle(
            [x1, max(0, y1 - 25), x1 + 110, y1],
            fill="#7d2034"
        )

        draw.text(
            (x1 + 5, max(0, y1 - 22)),
            text,
            fill="white"
        )

        detected += 1

    return result, detected


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚗 AI Object Detection Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Compare YOLOv8 and SSD on your trained object detection models'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ⚙️ Model Settings")

model_name = st.sidebar.selectbox(
    "Select Detection Model",
    [
        "YOLOv8",
        "SSD",
        "Both"
    ]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"""
    **Device:** `{DEVICE}`

    **Selected Model:** `{model_name}`

    **Classes:** `Car`
    """
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

if model_name == "YOLOv8":
    performance_metrics = load_yolo_metrics()
elif model_name == "SSD":
    performance_metrics = load_ssd_metrics()
else:
    performance_metrics = {
        f"YOLOv8 {name}": value
        for name, value in load_yolo_metrics().items()
    }
    performance_metrics.update({
        f"SSD {name}": value
        for name, value in load_ssd_metrics().items()
    })

if performance_metrics:

    metric_columns = st.columns(len(performance_metrics))

    for metric_column, (metric_name, metric_value) in zip(
        metric_columns,
        performance_metrics.items()
    ):

        with metric_column:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{metric_name}</div>
                    <div class="metric-value">{
                        f"{metric_value:.2%}"
                        if metric_name.endswith((
                            "Precision", "Recall", "mAP50", "mAP50-95",
                            "Average confidence"
                        ))
                        else f"{metric_value:.2f}"
                    }</div>
                </div>
                """,
                unsafe_allow_html=True
            )

else:

    st.info("Metric not available.")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image for object detection",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# DETECTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
        '<div class="section-title">🔍 Complete Model Results</div>',
        unsafe_allow_html=True
    )

    show_yolo = model_name in {"YOLOv8", "Both"}
    show_ssd = model_name in {"SSD", "Both"}
    yolo_image: Any = None
    yolo_classes: Any = []
    yolo_scores: Any = np.array([])
    ssd_image: Any = None
    ssd_classes: Any = []
    ssd_scores: Any = np.array([])

    with st.spinner("Running selected model detection..."):

        if show_yolo:

            yolo_model = load_yolov8()
            yolo_result: Any = list(yolo_model.predict(
                image,
                conf=confidence,
                verbose=False
            ))[0]
            yolo_boxes: Any = yolo_result.boxes
            if yolo_boxes is not None and len(yolo_boxes) > 0:
                yolo_scores = yolo_boxes.conf.cpu().numpy()
                yolo_classes = [
                    yolo_result.names[int(class_id)]
                    for class_id in yolo_boxes.cls.cpu().numpy()
                ]
            else:
                yolo_scores = np.array([])
                yolo_classes = []
            yolo_image = Image.fromarray(yolo_result.plot())

        if show_ssd:

            ssd_model = load_ssd()
            ssd_boxes, ssd_scores, ssd_labels = run_ssd(ssd_model, image)
            ssd_image, _ = draw_ssd_boxes(
                image,
                ssd_boxes,
                ssd_scores,
                ssd_labels,
                confidence
            )
            ssd_mask = ssd_scores >= confidence
            ssd_scores = ssd_scores[ssd_mask]
            ssd_classes = ["Car"] * len(ssd_scores)

    result_columns = (
        st.columns(2)
        if model_name == "Both"
        else [st.container()]
    )

    if show_yolo:

        with result_columns[0]:

            st.markdown("### YOLOv8 Model Results")
            show_model_images(image, yolo_image)
            show_detection_details("YOLOv8", yolo_classes, yolo_scores)

            st.markdown("### YOLOv8 Training and Evaluation Graphs")
            show_saved_graphs("Saved YOLOv8 Results", find_yolo_results_dir())

    if show_ssd:

        column_index = 1 if model_name == "Both" else 0

        with result_columns[column_index]:

            st.markdown("### SSD Model Results")
            show_model_images(image, ssd_image)
            show_detection_details("SSD", ssd_classes, ssd_scores)

            st.markdown("### SSD Training and Evaluation Graphs")
            show_saved_graphs(
                "Saved SSD Results",
                find_ssd_graph_dir(),
                include_all_images=True
            )

    if model_name == "Both":

        st.markdown(
            '<div class="section-title">📊 YOLOv8 vs SSD Comparison</div>',
            unsafe_allow_html=True
        )

        comparison_columns = st.columns(2)

        with comparison_columns[0]:

            st.markdown("### YOLOv8")
            yolo_metrics = load_yolo_metrics()

            if yolo_metrics:
                for metric_name, metric_value in yolo_metrics.items():
                    st.write(f"{metric_name}: {metric_value:.4f}")
            else:
                st.info("Metric not available.")

        with comparison_columns[1]:

            st.markdown("### SSD")
            ssd_metrics = load_ssd_metrics()

            if ssd_metrics:
                for metric_name, metric_value in ssd_metrics.items():
                    st.write(f"{metric_name}: {metric_value:.4f}")
            else:
                st.info("Metric not available.")


else:

    st.markdown(
        """
        <div class="info-box">
        📌 Upload a car image above to start object detection.
        <br><br>
        The selected trained model will detect cars and display
        bounding boxes with confidence scores.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#b98f97;">
        AI Object Detection System • YOLOv8 • SSD
        <br>
        Trained models are used for inference
    </div>
    """,
    unsafe_allow_html=True
)

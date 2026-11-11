"""
Facial Emotion Recognition — Streamlit demo
Pretrained mini-Xception CNN (FER-2013, 7 emotions) + OpenCV face detection.
Model: Octavio Arriaga, "face_classification" (MIT License), as packaged in the `fer` library (MIT License).
Run locally:  streamlit run app.py
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "emotion_model.hdf5"
CASCADE_FILE = BASE_DIR / "haarcascade_frontalface_default.xml"

IMG_SIZE = 64
# Output order of this pretrained model (do not reorder)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOJI = {
    "Angry": "😠", "Disgust": "🤢", "Fear": "😨", "Happy": "😄",
    "Sad": "😢", "Surprise": "😲", "Neutral": "😐",
}

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🙂", layout="wide")


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import keras
    return keras.models.load_model(str(MODEL_FILE), compile=False)


@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(str(CASCADE_FILE))


# ---------------------------------------------------------------- inference
def preprocess(gray_face):
    """Resize to 64x64 and scale pixels to [-1, 1] (as the model was trained)."""
    x = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    return (x - 0.5) * 2.0


def predict_faces(model, faces):
    """Test-time augmentation: average prediction of each face and its mirror."""
    batch = np.stack(faces)[..., np.newaxis]
    p = model.predict(batch, verbose=0)
    p_flip = model.predict(batch[:, :, ::-1, :], verbose=0)
    return (p + p_flip) / 2


def analyze(image_rgb, model, detector):
    """Detect faces, predict emotion for each, return annotated image + results."""
    img = image_rgb.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(boxes) == 0:
        return img, []

    faces = [preprocess(gray[y:y + h, x:x + w]) for (x, y, w, h) in boxes]
    all_probs = predict_faces(model, faces)

    results = []
    thickness = max(2, img.shape[1] // 300)
    font_scale = max(0.6, img.shape[1] / 900)

    for i, ((x, y, w, h), probs) in enumerate(zip(boxes, all_probs), start=1):
        label = EMOTIONS[int(np.argmax(probs))]
        conf = float(np.max(probs)) * 100

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 75, 75), thickness)
        text = f"#{i} {label} {conf:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(y - 10, th + 10)
        cv2.rectangle(img, (x, ty - th - 8), (x + tw + 8, ty + 4), (255, 75, 75), -1)
        cv2.putText(img, text, (x + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        results.append({"face": i, "emotion": label, "confidence": conf, "probs": probs})

    return img, results


# ---------------------------------------------------------------- UI
st.title("🙂 Facial Emotion Recognition")
st.caption("Detects faces and classifies 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.")

with st.sidebar:
    st.header("About the model")
    st.markdown(
        "- **Dataset:** FER-2013 (~35k facial images)\n"
        "- **Model:** pretrained mini-Xception CNN (~66% accuracy on FER-2013)\n"
        "- **Face detection:** OpenCV Haar Cascade\n"
        "- **Inference:** test-time augmentation (image + mirror)"
    )
    st.caption("Pretrained model by Octavio Arriaga (face_classification, MIT License).")
    st.divider()
    st.markdown("🔒 Images are processed in memory and **not stored**.")
    st.markdown("Built by **Rimas Abdulrhman**  \n"
                "[GitHub](https://github.com/pro931)")

for f in (MODEL_FILE, CASCADE_FILE):
    if not f.exists():
        st.error(f"`{f.name}` was not found next to app.py.")
        st.stop()

model = load_model()
detector = load_face_detector()

tab_cam, tab_upload = st.tabs(["📷 Use camera", "🖼️ Upload a photo"])
with tab_cam:
    cam_file = st.camera_input("Take a photo of your face")
with tab_upload:
    up_file = st.file_uploader("Upload a photo with one or more faces", type=["jpg", "jpeg", "png"])

source = cam_file or up_file
if source is None:
    st.info("Take a photo or upload an image to get started.")
    st.stop()

image = np.array(Image.open(source).convert("RGB"))
annotated, results = analyze(image, model, detector)

col_img, col_res = st.columns([3, 2])
with col_img:
    st.image(annotated, use_container_width=True)

with col_res:
    if not results:
        st.warning("No face detected. Try a clearer, front-facing photo with good lighting.")
    for r in results:
        st.subheader(f"Face #{r['face']}: {EMOJI[r['emotion']]} {r['emotion']}")
        st.metric("Confidence", f"{r['confidence']:.1f}%")
        chart = pd.DataFrame({"Probability (%)": r["probs"] * 100}, index=EMOTIONS)
        st.bar_chart(chart, horizontal=True, height=260)

"""
EmotionAI — Facial Emotion Recognition (Streamlit)
Pretrained mini-Xception CNN (FER-2013, 7 emotions) + OpenCV face detection.
Model: Octavio Arriaga, "face_classification" (MIT License), as packaged in the `fer` library (MIT License).
Run locally:  streamlit run app.py
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "emotion_model.hdf5"
CASCADE_FILE = BASE_DIR / "haarcascade_frontalface_default.xml"

IMG_SIZE = 64
# Output order of this pretrained model (do not reorder)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOJI = {"Angry": "😠", "Disgust": "🤢", "Fear": "😨", "Happy": "😄",
         "Sad": "😢", "Surprise": "😲", "Neutral": "😐"}
COLOR = {"Angry": "#F87171", "Disgust": "#A3E635", "Fear": "#2DD4BF", "Happy": "#FBBF24",
         "Sad": "#60A5FA", "Surprise": "#C084FC", "Neutral": "#94A3B8"}

st.set_page_config(page_title="EmotionAI · Facial Emotion Recognition", page_icon="🙂",
                   layout="wide", initial_sidebar_state="collapsed")


# ================================================================ styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="st-"], .stMarkdown, button, input { font-family: 'Inter', sans-serif !important; }
.stApp {
  background:
    radial-gradient(900px 500px at 15% -10%, rgba(139,92,246,.18), transparent 60%),
    radial-gradient(700px 400px at 95% 0%, rgba(236,72,153,.10), transparent 60%),
    #0A0A0F;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
.block-container { max-width: 1120px; padding-top: 1.2rem; padding-bottom: 3rem; }

/* ---------- nav ---------- */
.nav { display:flex; align-items:center; justify-content:space-between; padding: 6px 0 28px; }
.brand { display:flex; align-items:center; gap:10px; font-weight:700; font-size:18px; color:#fff; letter-spacing:-.01em; }
.brand-mark { width:30px; height:30px; border-radius:9px; display:grid; place-items:center;
  background: linear-gradient(135deg,#8B5CF6,#EC4899); font-size:16px; box-shadow:0 6px 20px rgba(139,92,246,.35); }
.nav-links a { color:#A1A1B5 !important; text-decoration:none; font-size:14px; margin-left:22px; font-weight:500; }
.nav-links a:hover { color:#fff !important; }

/* ---------- hero ---------- */
.hero { text-align:center; padding: 18px 0 8px; }
.eyebrow { display:inline-flex; gap:8px; align-items:center; padding:6px 14px; border-radius:999px;
  border:1px solid rgba(255,255,255,.10); background:rgba(255,255,255,.04); color:#C4B5FD; font-size:13px; font-weight:500; }
.eyebrow .dot { width:7px; height:7px; border-radius:50%; background:#22C55E; box-shadow:0 0 10px #22C55E; }
.hero h1 { font-size: clamp(38px, 6vw, 64px); line-height:1.05; font-weight:800; letter-spacing:-.035em;
  margin: 22px 0 14px; color:#fff; }
.grad { background: linear-gradient(90deg,#A78BFA 0%,#F472B6 50%,#FBBF24 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p { color:#A1A1B5; font-size:18px; max-width:620px; margin:0 auto; line-height:1.6; }
.stats { display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin: 28px 0 36px; }
.stat { text-align:center; padding:12px 18px; border-radius:14px; background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.08); min-width:120px; }
.stat b { display:block; color:#fff; font-size:20px; font-weight:700; letter-spacing:-.02em; }
.stat span { color:#8B8BA3; font-size:12.5px; }

/* ---------- cards ---------- */
.card-title { color:#fff; font-weight:600; font-size:16px; margin: 2px 0 2px; }
.card-sub { color:#8B8BA3; font-size:13.5px; margin-bottom: 6px; }
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 20px !important; border-color: rgba(255,255,255,.08) !important;
  background: rgba(20,20,28,.65); backdrop-filter: blur(10px);
}
[data-testid="stFileUploaderDropzone"] { border-radius:16px; border:1.5px dashed rgba(167,139,250,.35); background:rgba(139,92,246,.05); padding:18px 16px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:#A78BFA; background:rgba(139,92,246,.09); }
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img { border-radius:16px; }
[data-testid="stImage"] img { border-radius:16px; }
div[data-testid="stSegmentedControl"] button, .stSegmentedControl button { border-radius:10px !important; }

/* ---------- results ---------- */
.result-head { display:flex; align-items:center; gap:16px; margin-bottom:14px; }
.result-emoji { font-size:46px; width:72px; height:72px; display:grid; place-items:center; border-radius:20px; }
.result-label { font-size:13px; color:#8B8BA3; font-weight:500; }
.result-emotion { font-size:28px; font-weight:800; color:#fff; letter-spacing:-.02em; line-height:1.1; }
.conf-pill { display:inline-block; margin-top:6px; padding:3px 10px; border-radius:999px; font-size:12.5px; font-weight:600; }
.bar-row { display:grid; grid-template-columns: 92px 1fr 48px; align-items:center; gap:10px; margin:9px 0; }
.bar-name { color:#C9C9D6; font-size:13.5px; }
.bar-track { height:8px; border-radius:99px; background:rgba(255,255,255,.07); overflow:hidden; }
.bar-fill { height:100%; border-radius:99px; transition: width .6s ease; }
.bar-val { color:#8B8BA3; font-size:12.5px; text-align:right; font-variant-numeric: tabular-nums; }
.face-tag { color:#8B8BA3; font-size:12px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; margin: 4px 0 6px; }
.divider { height:1px; background:rgba(255,255,255,.07); margin:18px 0; }

/* ---------- empty / how ---------- */
.empty { text-align:center; padding: 46px 20px; color:#8B8BA3; }
.empty .big { font-size:44px; margin-bottom:10px; opacity:.9; }
.empty b { color:#fff; display:block; font-size:16px; margin-bottom:4px; }
.section-title { text-align:center; color:#fff; font-size:28px; font-weight:700; letter-spacing:-.02em; margin:64px 0 6px; }
.section-sub { text-align:center; color:#8B8BA3; margin-bottom:26px; }
.how { display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; }
.how-card { padding:22px; border-radius:18px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }
.how-num { width:34px; height:34px; border-radius:10px; display:grid; place-items:center; font-weight:700; color:#fff;
  background: linear-gradient(135deg,#8B5CF6,#EC4899); margin-bottom:14px; }
.how-card h4 { color:#fff; margin:0 0 6px; font-size:16px; font-weight:600; }
.how-card p { color:#8B8BA3; font-size:14px; margin:0; line-height:1.55; }
.chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:22px; }
.chip { padding:6px 12px; border-radius:999px; font-size:13px; color:#C9C9D6; background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); }
.footer { text-align:center; color:#6B6B80; font-size:13px; margin-top:64px; padding-top:22px; border-top:1px solid rgba(255,255,255,.06); }
.footer a { color:#A78BFA !important; text-decoration:none; }
@media (max-width: 760px) { .how { grid-template-columns: 1fr; } .nav-links a:not(:last-child) { display:none; } }
</style>
""", unsafe_allow_html=True)


# ================================================================ model
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import keras
    return keras.models.load_model(str(MODEL_FILE), compile=False)


@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(str(CASCADE_FILE))


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


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def analyze(image_rgb, model, detector):
    """Detect faces, predict emotion for each, return annotated image + results."""
    img = image_rgb.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(boxes) == 0:
        return img, []

    boxes = sorted(boxes, key=lambda b: b[0])  # left to right
    faces = [preprocess(gray[y:y + h, x:x + w]) for (x, y, w, h) in boxes]
    all_probs = predict_faces(model, faces)

    results = []
    thick = max(2, img.shape[1] // 320)
    font_scale = max(0.55, img.shape[1] / 1100)

    for i, ((x, y, w, h), probs) in enumerate(zip(boxes, all_probs), start=1):
        label = EMOTIONS[int(np.argmax(probs))]
        conf = float(np.max(probs)) * 100
        c = hex_to_rgb(COLOR[label])

        cv2.rectangle(img, (x, y), (x + w, y + h), c, thick, lineType=cv2.LINE_AA)
        text = f"{i}  {label}  {conf:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
        ty = max(y - 8, th + 14)
        cv2.rectangle(img, (x, ty - th - 12), (x + tw + 14, ty + 2), c, -1)
        cv2.putText(img, text, (x + 7, ty - 5), cv2.FONT_HERSHEY_DUPLEX, font_scale, (15, 15, 20), 1, cv2.LINE_AA)

        results.append({"face": i, "emotion": label, "confidence": conf, "probs": probs})
    return img, results


def load_image(file):
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")  # respect phone photo rotation
    img.thumbnail((1600, 1600))
    return np.array(img)


# ================================================================ html helpers
def result_card_html(r):
    c = COLOR[r["emotion"]]
    order = np.argsort(r["probs"])[::-1]
    bars = "".join(
        f'<div class="bar-row"><div class="bar-name">{EMOJI[EMOTIONS[k]]} {EMOTIONS[k]}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{r["probs"][k]*100:.1f}%;background:{COLOR[EMOTIONS[k]]}"></div></div>'
        f'<div class="bar-val">{r["probs"][k]*100:.0f}%</div></div>'
        for k in order
    )
    return (
        f'<div class="face-tag">Face {r["face"]}</div>'
        f'<div class="result-head">'
        f'<div class="result-emoji" style="background:{c}22;border:1px solid {c}55">{EMOJI[r["emotion"]]}</div>'
        f'<div><div class="result-label">Detected emotion</div>'
        f'<div class="result-emotion">{r["emotion"]}</div>'
        f'<span class="conf-pill" style="background:{c}22;color:{c}">{r["confidence"]:.1f}% confidence</span></div></div>'
        f'{bars}'
    )


# ================================================================ page
st.markdown("""
<div class="nav">
  <div class="brand"><div class="brand-mark">🙂</div>EmotionAI</div>
  <div class="nav-links">
    <a href="#how-it-works">How it works</a>
    <a href="https://github.com/pro931/Facial-Emotion-Recognition-using-Machine-Learning-cnn" target="_blank">GitHub ↗</a>
  </div>
</div>

<div class="hero">
  <span class="eyebrow"><span class="dot"></span>Computer Vision · Deep Learning</span>
  <h1>Read the <span class="grad">emotion</span><br>behind every face.</h1>
  <p>Upload a photo or use your camera. The model finds each face and recognizes
     7 emotions in under a second. Nothing is saved.</p>
</div>

<div class="stats">
  <div class="stat"><b>7</b><span>emotions</span></div>
  <div class="stat"><b>~66%</b><span>accuracy on FER-2013</span></div>
  <div class="stat"><b>&lt;1s</b><span>per image</span></div>
  <div class="stat"><b>0</b><span>images stored</span></div>
</div>
""", unsafe_allow_html=True)

for f in (MODEL_FILE, CASCADE_FILE):
    if not f.exists():
        st.error(f"`{f.name}` was not found next to app.py.")
        st.stop()

model = load_model()
detector = load_face_detector()

left, right = st.columns([1.15, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="card-title">Your image</div>'
                    '<div class="card-sub">Clear, front-facing photos with good lighting work best.</div>',
                    unsafe_allow_html=True)
        mode = st.segmented_control("Input", ["🖼️  Upload photo", "📷  Use camera"],
                                    default="🖼️  Upload photo", label_visibility="collapsed")
        if mode == "📷  Use camera":
            source = st.camera_input("Take a photo", label_visibility="collapsed")
        else:
            source = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"],
                                      label_visibility="collapsed")

        annotated, results = None, []
        if source is not None:
            with st.spinner("Analyzing faces…"):
                annotated, results = analyze(load_image(source), model, detector)
            st.image(annotated, use_container_width=True)

with right:
    with st.container(border=True):
        if source is None:
            st.markdown("""<div class="empty"><div class="big">✨</div>
                <b>Results will appear here</b>
                Add a photo to see the detected emotion and the full probability breakdown.</div>""",
                        unsafe_allow_html=True)
        elif not results:
            st.markdown("""<div class="empty"><div class="big">🔍</div>
                <b>No face detected</b>
                Try a clearer, front-facing photo, closer to the camera and with better lighting.</div>""",
                        unsafe_allow_html=True)
        else:
            if len(results) > 1:
                st.markdown(f'<div class="card-sub">{len(results)} faces found · numbered left to right</div>',
                            unsafe_allow_html=True)
            html = '<div class="divider"></div>'.join(result_card_html(r) for r in results)
            st.markdown(html, unsafe_allow_html=True)

st.markdown("""
<div id="how-it-works" class="section-title">How it works</div>
<div class="section-sub">A three-step computer-vision pipeline</div>
<div class="how">
  <div class="how-card"><div class="how-num">1</div><h4>Detect</h4>
    <p>OpenCV's Haar Cascade scans the image and locates every face.</p></div>
  <div class="how-card"><div class="how-num">2</div><h4>Prepare</h4>
    <p>Each face is cropped, converted to grayscale, resized to 64×64 and normalized.</p></div>
  <div class="how-card"><div class="how-num">3</div><h4>Classify</h4>
    <p>A mini-Xception CNN scores 7 emotions; the face and its mirror image are averaged for a steadier result.</p></div>
</div>
<div class="chips">
  <span class="chip">Python</span><span class="chip">TensorFlow / Keras</span><span class="chip">OpenCV</span>
  <span class="chip">NumPy</span><span class="chip">Streamlit</span>
</div>

<div class="footer">
  Built by <b style="color:#C9C9D6">Rimas Abdulrhman</b>, Data Analyst ·
  <a href="https://github.com/pro931" target="_blank">GitHub</a><br>
  <span style="font-size:12px">Pretrained model by Octavio Arriaga (face_classification, MIT License).
  Images are processed in memory and never stored.</span>
</div>
""", unsafe_allow_html=True)

"""
EmotionAI — Facial Emotion Recognition (Streamlit)
Graduation project, B.Sc. Information Systems, King Khalid University (2023-24).
Real-time (WebRTC) + photo analysis with a pretrained mini-Xception CNN (FER-2013, 7 emotions).
Model: Octavio Arriaga, "face_classification" (MIT License), as packaged in the `fer` library (MIT License).
Run locally:  streamlit run app.py
"""

import threading
import time
from collections import deque
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "emotion_model.hdf5"
CASCADE_FILE = BASE_DIR / "haarcascade_frontalface_default.xml"
ASSETS = BASE_DIR / "assets"

IMG_SIZE = 64
# Output order of this pretrained model (do not reorder)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOJI = {"Angry": "😠", "Disgust": "🤢", "Fear": "😨", "Happy": "😄",
         "Sad": "😢", "Surprise": "😲", "Neutral": "😐"}
COLOR = {"Angry": "#F87171", "Disgust": "#A3E635", "Fear": "#2DD4BF", "Happy": "#FBBF24",
         "Sad": "#60A5FA", "Surprise": "#C084FC", "Neutral": "#94A3B8"}
GITHUB_REPO = "https://github.com/pro931/Facial-Emotion-Recognition-using-Machine-Learning-cnn"

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
header[data-testid="stHeader"] { background: rgba(10,10,15,.75); backdrop-filter: blur(12px); }
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
.block-container { max-width: 1120px; padding-top: 3.2rem; padding-bottom: 3rem; }

/* hero */
.hero { text-align:center; padding: 18px 0 8px; }
.eyebrow { display:inline-flex; gap:8px; align-items:center; padding:6px 14px; border-radius:999px;
  border:1px solid rgba(255,255,255,.10); background:rgba(255,255,255,.04); color:#C4B5FD; font-size:13px; font-weight:500; }
.eyebrow .dot { width:7px; height:7px; border-radius:50%; background:#22C55E; box-shadow:0 0 10px #22C55E; }
.hero h1 { font-size: clamp(36px, 6vw, 62px); line-height:1.05; font-weight:800; letter-spacing:-.035em; margin: 22px 0 14px; color:#fff; }
.grad { background: linear-gradient(90deg,#A78BFA 0%,#F472B6 50%,#FBBF24 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p { color:#A1A1B5; font-size:18px; max-width:640px; margin:0 auto; line-height:1.6; }
.stats { display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin: 28px 0 36px; }
.stat { text-align:center; padding:12px 18px; border-radius:14px; background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.08); min-width:120px; }
.stat b { display:block; color:#fff; font-size:20px; font-weight:700; letter-spacing:-.02em; }
.stat span { color:#8B8BA3; font-size:12.5px; }

/* cards */
.card-title { color:#fff; font-weight:600; font-size:16px; margin: 2px 0 2px; }
.card-sub { color:#8B8BA3; font-size:13.5px; margin-bottom: 6px; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 20px !important; border-color: rgba(255,255,255,.08) !important;
  background: rgba(20,20,28,.65); backdrop-filter: blur(10px); }
[data-testid="stFileUploaderDropzone"] { border-radius:16px; border:1.5px dashed rgba(167,139,250,.35); background:rgba(139,92,246,.05); padding:18px 16px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:#A78BFA; background:rgba(139,92,246,.09); }
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img, video { border-radius:16px; }
[data-testid="stImage"] img { border-radius:14px; }
[data-testid="stMetric"] { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:12px 14px; }
[data-testid="stMetricValue"] { font-size:24px; }

/* results */
.result-head { display:flex; align-items:center; gap:16px; margin-bottom:14px; }
.result-emoji { font-size:46px; width:72px; height:72px; display:grid; place-items:center; border-radius:20px; }
.result-label { font-size:13px; color:#8B8BA3; font-weight:500; }
.result-emotion { font-size:28px; font-weight:800; color:#fff; letter-spacing:-.02em; line-height:1.1; }
.conf-pill { display:inline-block; margin-top:6px; padding:3px 10px; border-radius:999px; font-size:12.5px; font-weight:600; }
.bar-row { display:grid; grid-template-columns: 92px 1fr 48px; align-items:center; gap:10px; margin:9px 0; }
.bar-name { color:#C9C9D6; font-size:13.5px; }
.bar-track { height:8px; border-radius:99px; background:rgba(255,255,255,.07); overflow:hidden; }
.bar-fill { height:100%; border-radius:99px; }
.bar-val { color:#8B8BA3; font-size:12.5px; text-align:right; font-variant-numeric: tabular-nums; }
.face-tag { color:#8B8BA3; font-size:12px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; margin: 4px 0 6px; }
.divider { height:1px; background:rgba(255,255,255,.07); margin:18px 0; }
.live-badge { display:inline-flex; align-items:center; gap:7px; padding:4px 11px; border-radius:999px; font-size:12.5px; font-weight:600;
  background:rgba(239,68,68,.12); color:#FCA5A5; border:1px solid rgba(239,68,68,.3); }
.live-badge i { width:7px; height:7px; border-radius:50%; background:#EF4444; box-shadow:0 0 8px #EF4444; animation: pulse 1.2s infinite; }
@keyframes pulse { 50% { opacity:.3; } }
.note { color:#8B8BA3; font-size:12.5px; margin-top:8px; line-height:1.5; }

/* empty / sections */
.empty { text-align:center; padding: 46px 20px; color:#8B8BA3; }
.empty .big { font-size:44px; margin-bottom:10px; opacity:.9; }
.empty b { color:#fff; display:block; font-size:16px; margin-bottom:4px; }
.section-title { text-align:center; color:#fff; font-size:28px; font-weight:700; letter-spacing:-.02em; margin:64px 0 6px; }
.section-sub { text-align:center; color:#8B8BA3; margin-bottom:26px; }
.grid3 { display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; }
.grid2 { display:grid; grid-template-columns: repeat(2, 1fr); gap:14px; }
.tile { padding:22px; border-radius:18px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }
.tile-num { width:34px; height:34px; border-radius:10px; display:grid; place-items:center; font-weight:700; color:#fff;
  background: linear-gradient(135deg,#8B5CF6,#EC4899); margin-bottom:14px; }
.tile h4 { color:#fff; margin:0 0 6px; font-size:16px; font-weight:600; }
.tile p, .tile li { color:#A1A1B5; font-size:14px; margin:0; line-height:1.6; }
.tile ul { padding-left:18px; margin:0; }
.chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:22px; }
.chip { padding:6px 12px; border-radius:999px; font-size:13px; color:#C9C9D6; background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); }
.kicker { color:#C4B5FD; font-size:13px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; margin:56px 0 8px; }
.h2 { color:#fff; font-size:30px; font-weight:800; letter-spacing:-.025em; margin:0 0 10px; line-height:1.15; }
.lead { color:#A1A1B5; font-size:16.5px; line-height:1.7; max-width:820px; margin-bottom:18px; }
.person { text-align:center; padding:20px; border-radius:18px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }
.avatar { width:54px; height:54px; border-radius:50%; margin:0 auto 10px; display:grid; place-items:center; font-weight:700; color:#fff; font-size:18px;
  background: linear-gradient(135deg,#8B5CF6,#EC4899); }
.person b { color:#fff; display:block; } .person span { color:#8B8BA3; font-size:13px; }
.footer { text-align:center; color:#6B6B80; font-size:13px; margin-top:64px; padding-top:22px; border-top:1px solid rgba(255,255,255,.06); }
.footer a { color:#A78BFA !important; text-decoration:none; }
@media (max-width: 760px) { .grid3, .grid2 { grid-template-columns: 1fr; } }
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


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def draw_label(img, box, label, conf, idx, bgr=False):
    x, y, w, h = box
    c = hex_to_rgb(COLOR[label])
    if bgr:
        c = c[::-1]
    thick = max(2, img.shape[1] // 320)
    font_scale = max(0.5, img.shape[1] / 1100)
    cv2.rectangle(img, (x, y), (x + w, y + h), c, thick, lineType=cv2.LINE_AA)
    text = f"{idx}  {label}  {conf:.0f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
    ty = max(y - 8, th + 14)
    cv2.rectangle(img, (x, ty - th - 12), (x + tw + 14, ty + 2), c, -1)
    cv2.putText(img, text, (x + 7, ty - 5), cv2.FONT_HERSHEY_DUPLEX, font_scale, (15, 15, 20), 1, cv2.LINE_AA)


def analyze_photo(image_rgb, model, detector):
    """Photo mode: detect faces, predict with test-time augmentation (face + mirror)."""
    img = image_rgb.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(boxes) == 0:
        return img, []
    boxes = sorted(boxes, key=lambda b: b[0])
    batch = np.stack([preprocess(gray[y:y + h, x:x + w]) for (x, y, w, h) in boxes])[..., None]
    probs_all = (model.predict(batch, verbose=0) + model.predict(batch[:, :, ::-1, :], verbose=0)) / 2
    results = []
    for i, (box, probs) in enumerate(zip(boxes, probs_all), start=1):
        label = EMOTIONS[int(np.argmax(probs))]
        conf = float(np.max(probs)) * 100
        draw_label(img, box, label, conf, i)
        results.append({"face": i, "emotion": label, "confidence": conf, "probs": probs})
    return img, results


def load_image(file):
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    img.thumbnail((1600, 1600))
    return np.array(img)


def result_card_html(r):
    c = COLOR[r["emotion"]]
    order = np.argsort(r["probs"])[::-1]
    bars = "".join(
        f'<div class="bar-row"><div class="bar-name">{EMOJI[EMOTIONS[k]]} {EMOTIONS[k]}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{r["probs"][k]*100:.1f}%;background:{COLOR[EMOTIONS[k]]}"></div></div>'
        f'<div class="bar-val">{r["probs"][k]*100:.0f}%</div></div>'
        for k in order)
    return (f'<div class="face-tag">Face {r["face"]}</div><div class="result-head">'
            f'<div class="result-emoji" style="background:{c}22;border:1px solid {c}55">{EMOJI[r["emotion"]]}</div>'
            f'<div><div class="result-label">Detected emotion</div><div class="result-emotion">{r["emotion"]}</div>'
            f'<span class="conf-pill" style="background:{c}22;color:{c}">{r["confidence"]:.1f}% confidence</span></div></div>{bars}')


# ================================================================ live (real-time)
class LiveStats:
    """Thread-safe store shared between the video thread and the page."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.counts = {e: 0 for e in EMOTIONS}
            self.timeline = deque(maxlen=600)  # (seconds since start, emotion, confidence)
            self.current = None                # (emotion, confidence, probs)
            self.faces_now = 0
            self.frames = 0
            self.start = time.time()
            self.smooth = None

    def update(self, primary_probs, faces_now):
        with self.lock:
            self.frames += 1
            self.faces_now = faces_now
            if primary_probs is None:
                return
            self.smooth = primary_probs if self.smooth is None else 0.6 * self.smooth + 0.4 * primary_probs
            label = EMOTIONS[int(np.argmax(self.smooth))]
            conf = float(np.max(self.smooth)) * 100
            self.current = (label, conf, self.smooth.copy())
            self.counts[label] += 1
            self.timeline.append((time.time() - self.start, label, conf))

    def snapshot(self):
        with self.lock:
            return (dict(self.counts), list(self.timeline), self.current,
                    self.faces_now, self.frames, time.time() - self.start)


def make_frame_callback(model, detector, stats):
    def callback(frame):
        import av
        img = frame.to_ndarray(format="bgr24")
        h0, w0 = img.shape[:2]
        scale = 480 / w0 if w0 > 480 else 1.0
        small = cv2.resize(img, (int(w0 * scale), int(h0 * scale))) if scale != 1.0 else img
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        boxes = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
        primary = None
        if len(boxes):
            boxes = sorted(boxes, key=lambda b: b[0])
            batch = np.stack([preprocess(gray[y:y + h, x:x + w]) for (x, y, w, h) in boxes])[..., None]
            probs_all = np.asarray(model(batch, training=False))
            areas = [w * h for (_, _, w, h) in boxes]
            primary = probs_all[int(np.argmax(areas))]
            for i, ((x, y, w, h), probs) in enumerate(zip(boxes, probs_all), start=1):
                box = tuple(int(v / scale) for v in (x, y, w, h))
                label = EMOTIONS[int(np.argmax(probs))]
                draw_label(img, box, label, float(np.max(probs)) * 100, i, bgr=True)
        stats.update(primary, len(boxes))
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    return callback


def live_dashboard(stats, ph_metrics, ph_card, ph_charts):
    counts, timeline, current, faces_now, frames, elapsed = stats.snapshot()
    with ph_metrics.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Faces now", faces_now)
        c2.metric("Frames analyzed", f"{frames:,}")
        c3.metric("Session", f"{int(elapsed // 60)}:{int(elapsed % 60):02d}")
    with ph_card.container():
        if current is None:
            st.markdown('<div class="empty" style="padding:26px"><div class="big">👀</div><b>Looking for a face…</b>'
                        'Face the camera with good lighting.</div>', unsafe_allow_html=True)
        else:
            label, conf, probs = current
            st.markdown(result_card_html({"face": "live · smoothed", "emotion": label,
                                          "confidence": conf, "probs": probs}), unsafe_allow_html=True)
    with ph_charts.container():
        if sum(counts.values()):
            scale = alt.Scale(domain=EMOTIONS, range=[COLOR[e] for e in EMOTIONS])
            df = pd.DataFrame({"Emotion": list(counts), "Frames": list(counts.values())})
            donut = alt.Chart(df[df.Frames > 0]).mark_arc(innerRadius=55, cornerRadius=4).encode(
                theta="Frames:Q",
                color=alt.Color("Emotion:N", scale=scale,
                                legend=alt.Legend(orient="bottom", columns=4, labelColor="#C9C9D6", title=None)),
                tooltip=["Emotion", "Frames"]).properties(height=260, title="Emotion share this session")
            tl = pd.DataFrame(timeline, columns=["t", "Emotion", "Confidence"])
            dots = alt.Chart(tl).mark_circle(size=28, opacity=.85).encode(
                x=alt.X("t:Q", title="seconds"), y=alt.Y("Emotion:N", sort=EMOTIONS, title=None),
                color=alt.Color("Emotion:N", scale=scale, legend=None),
                tooltip=["Emotion", alt.Tooltip("Confidence:Q", format=".0f"), alt.Tooltip("t:Q", format=".1f")]
            ).properties(height=260, title="Emotion timeline")
            a, b = st.columns(2)
            a.altair_chart(donut.configure(background="transparent").configure_view(stroke=None)
                           .configure_title(color="#EDEDF2"), use_container_width=True)
            b.altair_chart(dots.configure(background="transparent").configure_view(stroke=None)
                           .configure_title(color="#EDEDF2")
                           .configure_axis(labelColor="#A1A1B5", titleColor="#8B8BA3", gridColor="#ffffff14"),
                           use_container_width=True)


# ================================================================ pages
def footer():
    st.markdown(f"""
<div class="footer">
  Built by <b style="color:#C9C9D6">Rimas Abdulrhman</b>, Data Analyst ·
  <a href="https://github.com/pro931" target="_blank">GitHub</a> · <a href="{GITHUB_REPO}" target="_blank">Source code</a><br>
  <span style="font-size:12px">Graduation project, King Khalid University. Pretrained model by Octavio Arriaga
  (face_classification, MIT License). Video and images are processed in memory and never stored.</span>
</div>""", unsafe_allow_html=True)


def page_demo():
    st.markdown("""
<div class="hero">
  <span class="eyebrow"><span class="dot"></span>Graduation Project · Computer Vision</span>
  <h1>Read the <span class="grad">emotion</span><br>behind every face, live.</h1>
  <p>Turn on your camera for real-time emotion recognition, or analyze a single photo.
     Each face is detected and classified into 7 emotions. Nothing is saved.</p>
</div>
<div class="stats">
  <div class="stat"><b>Real-time</b><span>live video</span></div>
  <div class="stat"><b>7</b><span>emotions</span></div>
  <div class="stat"><b>~66%</b><span>accuracy on FER-2013</span></div>
  <div class="stat"><b>0</b><span>images stored</span></div>
</div>""", unsafe_allow_html=True)

    for f in (MODEL_FILE, CASCADE_FILE):
        if not f.exists():
            st.error(f"`{f.name}` was not found next to app.py.")
            st.stop()
    model = load_model()
    detector = load_face_detector()

    mode = st.segmented_control("Mode", ["🔴  Live (real-time)", "🖼️  Upload photo", "📷  Snapshot"],
                                default="🔴  Live (real-time)", label_visibility="collapsed")

    if mode == "🔴  Live (real-time)":
        from streamlit_webrtc import WebRtcMode, webrtc_streamer

        if "live_stats" not in st.session_state:
            st.session_state.live_stats = LiveStats()
        stats = st.session_state.live_stats

        left, right = st.columns([1.15, 1], gap="large")
        with left:
            with st.container(border=True):
                st.markdown('<div class="card-title">Live camera <span class="live-badge" style="margin-left:6px">'
                            '<i></i>REAL-TIME</span></div><div class="card-sub">Press <b>START</b> and allow camera '
                            'access. Emotions are tracked frame by frame.</div>', unsafe_allow_html=True)
                ctx = webrtc_streamer(
                    key="emotion-live",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=make_frame_callback(model, detector, stats),
                    media_stream_constraints={"video": {"width": {"ideal": 640}}, "audio": False},
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302",
                                                                "stun:stun1.l.google.com:19302"]}]},
                    async_processing=True,
                )
                c1, c2 = st.columns([1, 2])
                if c1.button("Reset session", use_container_width=True):
                    stats.reset()
                c2.markdown('<div class="note">If the video doesn\'t connect (some office or mobile networks block it), '
                            'use <b>Snapshot</b> or <b>Upload photo</b> above.</div>', unsafe_allow_html=True)
        with right:
            with st.container(border=True):
                ph_card = st.empty()
        ph_metrics = st.empty()
        ph_charts = st.empty()

        if ctx.state.playing:
            while ctx.state.playing:
                live_dashboard(stats, ph_metrics, ph_card, ph_charts)
                time.sleep(1.0)
        else:
            live_dashboard(stats, ph_metrics, ph_card, ph_charts)
            if stats.frames == 0:
                ph_card.markdown('<div class="empty"><div class="big">🎥</div><b>Live results appear here</b>'
                                 'Start the camera to see the current emotion, confidence and a live session '
                                 'dashboard.</div>', unsafe_allow_html=True)
    else:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            with st.container(border=True):
                st.markdown('<div class="card-title">Your image</div>'
                            '<div class="card-sub">Clear, front-facing photos with good lighting work best.</div>',
                            unsafe_allow_html=True)
                if mode == "📷  Snapshot":
                    source = st.camera_input("Take a photo", label_visibility="collapsed")
                else:
                    source = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"],
                                              label_visibility="collapsed")
                results = []
                if source is not None:
                    with st.spinner("Analyzing faces…"):
                        annotated, results = analyze_photo(load_image(source), model, detector)
                    st.image(annotated, use_container_width=True)
        with right:
            with st.container(border=True):
                if source is None:
                    st.markdown('<div class="empty"><div class="big">✨</div><b>Results will appear here</b>'
                                'Add a photo to see the detected emotion and the full probability breakdown.</div>',
                                unsafe_allow_html=True)
                elif not results:
                    st.markdown('<div class="empty"><div class="big">🔍</div><b>No face detected</b>'
                                'Try a clearer, front-facing photo, closer to the camera and with better lighting.'
                                '</div>', unsafe_allow_html=True)
                else:
                    if len(results) > 1:
                        st.markdown(f'<div class="card-sub">{len(results)} faces found · numbered left to right</div>',
                                    unsafe_allow_html=True)
                    st.markdown('<div class="divider"></div>'.join(result_card_html(r) for r in results),
                                unsafe_allow_html=True)

    st.markdown("""
<div class="section-title">How it works</div>
<div class="section-sub">A three-step computer-vision pipeline</div>
<div class="grid3">
  <div class="tile"><div class="tile-num">1</div><h4>Detect</h4><p>OpenCV's Haar Cascade locates every face in each video frame or photo.</p></div>
  <div class="tile"><div class="tile-num">2</div><h4>Prepare</h4><p>Each face is cropped, converted to grayscale, resized to 64×64 and normalized.</p></div>
  <div class="tile"><div class="tile-num">3</div><h4>Classify</h4><p>A CNN scores 7 emotions. Live results are smoothed over frames to avoid flicker.</p></div>
</div>
<div class="chips"><span class="chip">Python</span><span class="chip">TensorFlow / Keras</span><span class="chip">OpenCV</span>
<span class="chip">WebRTC</span><span class="chip">Altair</span><span class="chip">Streamlit</span></div>""", unsafe_allow_html=True)
    footer()


def img(name, caption=None):
    p = ASSETS / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)


def page_project():
    st.markdown("""
<div class="hero">
  <span class="eyebrow">🎓 B.Sc. Information Systems · King Khalid University · 2023–24</span>
  <h1>Facial Emotion Recognition<br><span class="grad">using Machine Learning</span></h1>
  <p>Graduation project: a system that reads customers' emotions from their facial expressions,
     helping retailers understand satisfaction and which products customers respond to.</p>
</div>
<div class="stats">
  <div class="stat"><b>7</b><span>emotion classes</span></div>
  <div class="stat"><b>28k</b><span>training images</span></div>
  <div class="stat"><b>~63%</b><span>validation accuracy (our CNN)</span></div>
  <div class="stat"><b>3</b><span>apps: web, mobile, dashboard</span></div>
</div>

<div class="kicker">01 · The problem</div>
<div class="h2">Every purchase hides a feeling.</div>
<div class="lead">Retailers want customers to come back, but surveys are slow and suffer from survey fatigue.
We set out to read customer sentiment directly from facial expressions, in real time through in-store cameras or from photos
customers share, so decision-makers can see which products are welcomed and how satisfaction changes over time.</div>

<div class="kicker">02 · Objectives</div>
<div class="grid2">
  <div class="tile"><h4>🎯 Aim</h4><p>Design an application model that accurately predicts customer emotions across seven categories:
  anger, disgust, fear, happiness, neutral, sadness and surprise.</p></div>
  <div class="tile"><h4>✅ Objectives</h4><ul><li>Train and evaluate an ML model on a labelled dataset</li>
  <li>Integrate the model into a user-facing application</li><li>Build an emotions dashboard for decision-makers</li>
  <li>Make emotion identification easy for non-technical users</li></ul></div>
</div>

<div class="kicker">03 · Methodology</div>
<div class="h2">Agile, end-to-end ML lifecycle.</div>
<div class="lead">We followed Agile for a small team with incremental delivery, and the machine-learning cycle:
manage data → train models → evaluate → deploy. The FER dataset from Kaggle (~28k labelled faces) was split 75% / 25% for training and testing,
and a 4-block Convolutional Neural Network was trained in TensorFlow/Keras on 48×48 grayscale faces.</div>
""", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        img("ml_cycle.png", "Machine-learning modeling cycle")
    with c2:
        img("training_curves.png", "Training vs validation accuracy and loss over 15 epochs")

    st.markdown('<div class="kicker">04 · System design</div><div class="h2">Designed before it was built.</div>'
                '<div class="lead">Full systems-analysis package: use cases, class model, activity flow and database '
                'design.</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Use case", "Architecture flow", "Class diagram", "ER diagram"])
    with t1:
        img("use_case.png", "UML use case diagram")
    with t2:
        img("architecture.png", "Activity flow: read frame → detect face → recognize → display → store → dashboard")
    with t3:
        img("class_diagram.png", "UML class diagram")
    with t4:
        img("er_diagram.png", "Entity-relationship diagram (SQLite)")

    st.markdown("""
<div class="kicker">05 · Implementation</div>
<div class="h2">From model to product.</div>
<div class="grid3" style="margin-bottom:18px">
  <div class="tile"><h4>🧠 Model</h4><p>CNN trained in TensorFlow/Keras; faces detected with OpenCV Haar Cascade.</p></div>
  <div class="tile"><h4>🌐 Flask API + SQLite</h4><p>Endpoints for live video feed and image upload; every prediction stored with date, emotion and confidence.</p></div>
  <div class="tile"><h4>📱 Mobile app + dashboard</h4><p>Take photo, select image, real-time mode, and dashboards by ratio, time series and emotion.</p></div>
</div>""", unsafe_allow_html=True)
    a, b, c = st.columns(3, gap="medium")
    with a:
        img("app_home.png", "Mobile app: home")
    with b:
        img("app_dashboard_ratio.png", "Dashboard: emotion ratio")
    with c:
        img("app_dashboard_emotion.png", "Dashboard: counts by emotion")

    st.markdown("""
<div class="kicker">06 · Results & next steps</div>
<div class="grid2">
  <div class="tile"><h4>📈 Outcome</h4><p>A working pipeline from camera to dashboard. Our CNN reached about 63% validation accuracy
  on FER, a hard dataset where human agreement is around 65%. This live demo runs a pretrained mini-Xception (~66%)
  rebuilt as a web app with real-time video.</p></div>
  <div class="tile"><h4>🔭 Future work</h4><ul><li>Recognize veiled faces using the eye region</li>
  <li>Add more emotion categories and a larger dataset</li><li>Improve model accuracy (e.g. transfer learning)</li>
  <li>Better UI/UX ✓ done in this version</li></ul></div>
</div>

<div class="kicker">07 · Team</div>
<div class="grid3">
  <div class="person"><div class="avatar">RA</div><b>Rimas Abdulrhman</b><span>Team member · this web version</span></div>
  <div class="person"><div class="avatar">GN</div><b>Ghadi Nawaf</b><span>Team member</span></div>
  <div class="person"><div class="avatar">MM</div><b>Manar Mohamad</b><span>Team member</span></div>
</div>
<p class="note" style="text-align:center;margin-top:14px">Supervised by Dr. Suhail Qadir GH Qadir · College of Computer Science, King Khalid University</p>
""", unsafe_allow_html=True)
    footer()


pg = st.navigation([st.Page(page_demo, title="Live Demo", icon="🎥", url_path="demo", default=True),
                    st.Page(page_project, title="The Project", icon="🎓", url_path="project")],
                   position="top")
pg.run()

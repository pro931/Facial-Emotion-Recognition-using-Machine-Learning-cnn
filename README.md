# 🙂 Facial Emotion Recognition

A web app that detects faces in a photo and classifies each one into **7 emotions**:
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

**🔗 Live demo:** _add your Streamlit link here_

## How it works
1. **Face detection:** OpenCV Haar Cascade finds faces in the image.
2. **Preprocessing:** each face is cropped, converted to grayscale, resized to 64×64 and scaled to [-1, 1].
3. **Classification:** a pretrained mini-Xception CNN predicts the emotion; the prediction is averaged with the mirrored face (test-time augmentation).

## Model
| | |
|---|---|
| Dataset | FER-2013 (~35k facial images, 7 classes) |
| Model | mini-Xception CNN, pretrained (~66% accuracy on FER-2013; human accuracy on this dataset is ~65%) |
| Inference | Test-time augmentation (image + horizontal flip) |

## Tech stack
Python · TensorFlow/Keras · OpenCV · NumPy · Pandas · Streamlit

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app (camera + photo upload) |
| `emotion_model.hdf5` | Pretrained emotion classifier |
| `haarcascade_frontalface_default.xml` | Face detector |
| `face_emotion.py`, `EmotionRecognition.ipynb` | My first version: CNN trained from scratch (~63%) |
| `main.py`, `camera.py`, `database.py` | My first version: Flask app with live webcam and SQLite logging |

## Credits
Pretrained model by Octavio Arriaga ([face_classification](https://github.com/oarriaga/face_classification), MIT License),
as packaged in the [fer](https://github.com/justinshenk/fer) library (MIT License).

## Privacy
Images are processed in memory and are not stored by the web app.

---
Built by **Rimas Abdulrhman**, Data Analyst · [GitHub](https://github.com/pro931)

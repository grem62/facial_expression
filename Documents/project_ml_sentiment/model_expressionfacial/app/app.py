import streamlit as st
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS style
st.markdown("""
    <style>
        .main {
            background-color: #1e1e1e;
        }
        .stButton>button {
            color: white;
            background-color: #444444;
            border-radius: 8px;
            font-size: 18px;
        }
        h1 {
            color: #ffffff;
        }
        h2 {
            color: #cccccc;
        }
        .stMarkdown {
            color: #dddddd;
        }
        .stImage {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("Facial Expression Recognition AI 😊")
st.markdown("""
Welcome to our facial expression recognition app!  
Upload an image or use the real-time camera to detect faces and their expressions.  
""")

# Tabs for options
tab1, tab2 = st.tabs(["📂 Upload an Image", "📸 Use Live Camera"])

# Loading the face cascade model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# List of emotions
EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]

# Loading the custom model
model_path = "/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/models/fer2013/test4.h5"
model = load_model(model_path)

def detect_faces_and_predict(image):
    """
    Face detection and emotion prediction.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Emotion prediction
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # Predict with the model
        preds = model.predict(face)
        emotion = EMOTIONS[preds.argmax()]
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image

with tab1:
    st.header("📤 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image (JPG or PNG format)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Analyze button
        if st.button("Analyze Expression in Image"):
            processed_image = detect_faces_and_predict(image_np)
            st.image(processed_image, caption="Detection Result", use_column_width=True)

with tab2:
    st.header("📸 Live Camera")
    run_camera = st.checkbox("Enable Camera")

    if run_camera:
        # Camera configuration
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access the camera.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = detect_faces_and_predict(frame)
            stframe.image(processed_frame, channels="RGB", use_column_width=True)

        cap.release()
        cv2.destroyAllWindows()

# Footer
st.markdown("---")
st.markdown("""
👨‍💻 **Developed by Matheo Gremont**  
Contact me for any questions or improvements!  
""")
st.markdown("""
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/grem62/facial_expression)
[![Slack](https://img.shields.io/badge/Slack-Channel-green?logo=slack)](https://app.slack.com/client/T0423U1MW21/D08835SQCAU)
""")

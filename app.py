import streamlit as st
import cv2
from PIL import Image
import numpy as np
from transformers.models.auto import AutoModelForImageClassification
from transformers import AutoFeatureExtractor
from tensorflow.keras.models import load_model



# Ajouter une vidéo de fond
def add_background_video(video_path):
    video_html = f"""
    <video autoplay muted loop id="bg-video">
        <source src="{video_path}" type="video/mp4">
    </video>
    <style>
    #bg-video {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        z-index: -1;
        filter: brightness(50%);
    }}
    </style>
    """
    st.markdown(video_html, unsafe_allow_html=True)

# Appel de la fonction pour ajouter la vidéo de fond
add_background_video("/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/image_test/7020022_Brain_Science_3840x2160.mp4")


# Titre principal
st.title("Facial Expression Recognition AI 😊")
st.markdown("""

""")

# Onglets pour les options
tab1, tab2 = st.tabs(["📂 Télécharger une image", "📸 Utiliser la caméra en direct"])

# Chargement du modèle de cascade de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Liste des émotions
EMOTIONS = ["Colère", "Dégoût", "Peur", "Joie", "Tristesse", "Surprise", "Neutre"]

# Sélection du modèle
model_choice = st.sidebar.selectbox("Choisissez un modèle", ["motheecreator/vit-Facial-Expression-Recognition", "improve model"])

if model_choice == "improve model":
    model = load_model("/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/models/fer2013/test4.h5")
    feature_extractor = None  # Not needed for the Keras model
else:
    model = AutoModelForImageClassification.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
    feature_extractor = AutoFeatureExtractor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")

def detect_faces_and_predict(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Prédiction de l'émotion
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = (face * 255).astype("uint8")
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = np.expand_dims(face, axis=0)
        face = np.transpose(face, (0, 3, 1, 2))  # Convert to channel-first format

        # Extract features and predict with the model
        inputs = feature_extractor(images=face, return_tensors="pt")
        outputs = model(**inputs)
        preds = outputs.logits.softmax(dim=1).detach().numpy()
        emotion = EMOTIONS[preds.argmax()]
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return image

with tab1:
    st.header("📤 Téléchargez une image")
    uploaded_file = st.file_uploader("Choisissez une image (format JPG ou PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image, caption="Image téléchargée", use_column_width=True)

        # Bouton d'analyse
        if st.button("Analyser l'expression dans l'image"):
            processed_image = detect_faces_and_predict(image_np)
            st.image(processed_image, caption="Résultat de la détection", use_column_width=True)

with tab2:
    st.header("📸 Caméra en direct")
    run_camera = st.checkbox("Activer la caméra")

    if run_camera:
        # Configuration de la caméra
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Impossible d'accéder à la caméra.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = detect_faces_and_predict(frame)
            stframe.image(processed_frame, channels="RGB", use_column_width=True)

        cap.release()
        cv2.destroyAllWindows()

# Footer
st.markdown("---")
st.markdown("""
👨‍💻 **Développé par Matheo Gremont**  
Contactez-moi pour toute question ou amélioration !  
""")

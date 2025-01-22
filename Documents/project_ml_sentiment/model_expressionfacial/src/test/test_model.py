import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/models/test4.h5")

# Load an image for testing
image_path = "/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/image_test/IMG_5343.jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale for the model
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use the Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# If at least one face is detected in the image
if len(faces) > 0:
    (x, y, w, h) = faces[0]
    face_image = gray_image[y:y+h, x:x+w]

    # Resize the face to the required size for the model
    face_resized = cv2.resize(face_image, (48, 48))

    # Normalize the pixel values
    face_normalized = face_resized / 255.0

    # Add a dimension to simulate a batch
    face_input = np.expand_dims(face_normalized, axis=0)
    face_input = np.expand_dims(face_input, axis=-1)

    # Make a prediction
    predictions = model.predict(face_input)

    # Display the results
    class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]

    print(f"Predicted class: {predicted_class} (probability: {predictions[0][predicted_class_index]:.2f})")
else:
    print("No face detected in the image.")

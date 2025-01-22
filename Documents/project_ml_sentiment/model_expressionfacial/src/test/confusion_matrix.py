import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/models/fer2013/test4_improved.h5')

# Prepare the test data
# Replace this with the actual loading of your test data and their labels
X_test = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/test/X_test.npy')
y_test = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/test/y_test.npy')

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Ensure y_test is in the same format as y_pred_classes
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
	y_test = np.argmax(y_test, axis=1)

# Define the labels for the confusion matrix
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predictions')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

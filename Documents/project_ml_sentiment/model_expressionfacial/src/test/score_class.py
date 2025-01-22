import numpy as np
from sklearn.metrics import classification_report
from keras.models import load_model

# Load test data
X_test = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/test/X_test.npy')
y_test = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/test/y_test.npy')

# Load the pre-trained model
model = load_model('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/models/fer2013/test4.h5')

# Predict the class probabilities for the test data
y_pred = model.predict(X_test)

# Convert predicted probabilities to class indices
y_pred = np.argmax(y_pred, axis=1)

# Convert y_test to class indices if it is one-hot encoded
y_test = np.argmax(y_test, axis=1)

# Print the shapes of y_pred and y_test for verification
print("Shape of y_pred:", y_pred.shape)
print("Shape of y_test:", y_test.shape)

# Generate classification report (precision, recall, F1-score, support)
report = classification_report(y_test, y_pred, output_dict=True)
for label, metrics in report.items():
    if isinstance(metrics, dict):
        print(f"Class {label}:")
        print(f"  Precision: {metrics['precision']:.2f}")
        print(f"  Recall: {metrics['recall']:.2f}")
        print(f"  F1-score: {metrics['f1-score']:.2f}")
        print(f"  Support: {metrics['support']}")

import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split

# Data path
train_dir = '/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/raw/dataset_fusion/train'

# Define the labels
label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

# Function to load images and labels
def load_data(data_dir, label_map):
    images = []
    labels = []
    for label_folder, label_id in label_map.items():
        folder_path = os.path.join(data_dir, label_folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist.")
            continue
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
                img = image.img_to_array(img) / 255
                images.append(img)
                labels.append(label_id)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Load training and test data
X, y = load_data(train_dir, label_map)

# One-hot encode the labels
y = to_categorical(y, num_classes=len(label_map))

# Split data into training and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print the shapes of the datasets
print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)

# Save paths
base_save_dir = '/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed'
train_save_dir = os.path.join(base_save_dir, 'train')
test_save_dir = os.path.join(base_save_dir, 'test')
val_save_dir = os.path.join(base_save_dir, 'val')

# Save the datasets
np.save(os.path.join(train_save_dir, 'X_train.npy'), X_train)
np.save(os.path.join(train_save_dir, 'y_train.npy'), y_train)
np.save(os.path.join(test_save_dir, 'X_test.npy'), X_test)
np.save(os.path.join(test_save_dir, 'y_test.npy'), y_test)
np.save(os.path.join(val_save_dir, 'X_val.npy'), X_val)
np.save(os.path.join(val_save_dir, 'y_val.npy'), y_val)

# Verify if the datasets are saved successfully
if all([os.path.exists(os.path.join(train_save_dir, f)) for f in ['X_train.npy', 'y_train.npy']]):
    print("Training data saved successfully.")
if all([os.path.exists(os.path.join(test_save_dir, f)) for f in ['X_test.npy', 'y_test.npy']]):
    print("Test data saved successfully.")
if all([os.path.exists(os.path.join(val_save_dir, f)) for f in ['X_val.npy', 'y_val.npy']]):
    print("Validation data saved successfully.")

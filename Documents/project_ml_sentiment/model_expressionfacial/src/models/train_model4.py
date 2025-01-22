import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Load data
X_train = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/train/X_train.npy')
y_train = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/train/y_train.npy')
X_val = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/val/X_val.npy')
y_val = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/val/y_val.npy')
X_test = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/test/X_test.npy')
y_test = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/test/y_test.npy')

# Display data dimensions
print(f"Training data dimensions: {X_train.shape}")
print(f"Validation data dimensions: {X_val.shape}")
print(f"Training labels dimensions: {y_train.shape}")
print(f"Validation labels dimensions: {y_val.shape}")

# Calculate class weights
class_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_labels),
    y=class_labels
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

print("Calculated class weights: ", class_weights_dict)

# Count class distribution
class_counts = Counter(class_labels)
print("Initial class distribution:", class_counts)

# Data augmentation for underrepresented classes
augment_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

X_augmented = []
y_augmented = []

for label, count in class_counts.items():
    if count < max(class_counts.values()):
        # Generate additional samples
        samples_needed = max(class_counts.values()) - count
        idx = np.where(class_labels == label)[0]
        X_class = X_train[idx]
        y_class = y_train[idx]

        # Augment data
        augmented = augment_datagen.flow(X_class, y_class, batch_size=32)
        for i in range(samples_needed):
            augmented_batch = next(augmented)
            X_augmented.append(augmented_batch[0][0])
            y_augmented.append(augmented_batch[1][0])

# Convert lists to numpy arrays
X_augmented = np.array(X_augmented)
y_augmented = np.array(y_augmented)

# Combine original data with augmented data
X_train_balanced = np.concatenate((X_train, X_augmented), axis=0)
y_train_balanced = np.concatenate((y_train, y_augmented), axis=0)

print(f"Augmented data: {X_augmented.shape}, {y_augmented.shape}")
print(f"New training data dimensions: {X_train_balanced.shape}")

# Create data generators for training and validation
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train_balanced, y_train_balanced, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# Define the improved model structure with additional block and regularization
model = Sequential([
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    Conv2D(512, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    GlobalAveragePooling2D(),
    Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer=RMSprop(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Save the model
save_model_dir = '/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/models'
model_name = 'test4_improved.h5'
save_model_path = os.path.join(save_model_dir, model_name)

model.save(save_model_path)
print(f"Model saved at: {save_model_path}")

# Plot training curves
plt.figure(figsize=(12, 6))

# Accuracy curves
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy during Training and Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss curves
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss during Training and Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# Save the plot
output_path = "/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/reports/figures/"
plt.savefig(output_path + 'training_curves_improved.png')
print(f"Plot saved at: {output_path}")
plt.show()

# Predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(y_pred_classes)

# Display confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class ' + str(i) for i in range(8)], yticklabels=['Class ' + str(i) for i in range(8)])
plt.xlabel('Predictions')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

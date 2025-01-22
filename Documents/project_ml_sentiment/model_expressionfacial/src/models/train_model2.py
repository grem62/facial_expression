import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt

# data path
X_train = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/train/X_train.npy')
y_train = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/train/y_train.npy')
X_val = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/val/X_val.npy')
y_val = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/val/y_val.npy')
X_test = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/test/X_test.npy')
y_test = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/test/y_test.npy')

# dimenssions des données
print(X_train.shape, X_val.shape)

# dimenssions labbels
print(y_train.shape, y_val.shape)

# data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

# ajout de la data augmentation
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# strucutre du modele complexifié
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# compilation du modèle
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# arret automatique callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# entraînement du modèle
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# evaluation du model a partirr du test
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# sauvegarde du modèle
save_model_dir = '/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/models'
model_name = 'test2.h5'
save_model_path = os.path.join(save_model_dir, model_name)

model.save(save_model_path)
print(f"Model saved to {save_model_path}")

# plot graphique 
plt.figure(figsize=(12, 6))

# courbes de précision
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision - Entraînement')
plt.plot(history.history['val_accuracy'], label='Précision - Validation')
plt.title('Précision pendant l\'entraînement et la validation')
plt.xlabel('Epoques')
plt.ylabel('Précision')
plt.legend()

# courbes de perte
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte - Entraînement')
plt.plot(history.history['val_loss'], label='Perte - Validation')
plt.title('Perte pendant l\'entraînement et la validation')
plt.xlabel('Epoques')
plt.ylabel('Perte')
plt.legend()

plt.tight_layout()

# sauvegarde graphique 
output_path = "/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/reports/figures/"
plt.savefig(output_path)
print(f"Graphique sauvegardé dans {output_path}")
plt.show()

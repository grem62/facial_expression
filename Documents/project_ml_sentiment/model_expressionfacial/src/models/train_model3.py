import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Activation
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt

# data path 

X_train = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/train/X_train.npy')
y_train = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/train/y_train.npy')
X_val = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/val/X_val.npy')
y_val = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/val/y_val.npy')
X_test = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/test/X_test.npy')
y_test = np.load('/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/data/processed/test/y_test.npy')

# dimenssion des données(train, val, test)
print(f"Dimensions des données d'entraînement : {X_train.shape}")
print(f"Dimensions des données de validation : {X_val.shape}")
print(f"Dimensions des étiquettes d'entraînement : {y_train.shape}")
print(f"Dimensions des étiquettes de validation : {y_val.shape}")

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

# structure du modèle amélioré avec l'ajout de BatchNormalisation, dropout, MaxPooling
model = Sequential([
    # Bloc 1
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    # Bloc 2
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    
    # Bloc 3
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    
    # Global Pooling et Dense
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# compilation du modèle
model.compile(optimizer=RMSprop(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# arret automatique callback et reduction learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# entraînement du modèle
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping, reduce_lr]
)

# évalutation du model a partir du jeu de test
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# sauvegarde du modèle
save_model_dir = '/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/models'
model_name = 'test3_improved.h5'
save_model_path = os.path.join(save_model_dir, model_name)

model.save(save_model_path)
print(f"Modèle sauvegardé à l'emplacement : {save_model_path}")

# plot graphique
plt.figure(figsize=(12, 6))

# courbes de précision
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision - Entraînement')
plt.plot(history.history['val_accuracy'], label='Précision - Validation')
plt.title('Précision pendant l\'entraînement et la validation')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.legend()

# courbes de perte
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte - Entraînement')
plt.plot(history.history['val_loss'], label='Perte - Validation')
plt.title('Perte pendant l\'entraînement et la validation')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()

plt.tight_layout()

# sauvegarder le graphique
output_path = "/Users/grem/Documents/project_ml_sentiment/model_expressionfacial/reports/figures/"
plt.savefig(output_path + 'training_curves_improved.png')
print(f"Graphique sauvegardé dans : {output_path}")
plt.show()

import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import random

# Définition des chemins des dossiers contenant les images d'entraînement et de test
train_folder = "./Images_Processed_Threshold"

# Fonction pour charger les images
def load_images(folder_path, nb_digit, nb_train_version, nb_test_version):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for digit in range(nb_digit):
        all_indices = list(range(nb_train_version + nb_test_version))  # Liste de tous les indices des images
        test_indices = random.sample(all_indices, nb_test_version)  # Sélection aléatoire des indices pour les images de test
        for version in range(nb_train_version + nb_test_version):
            image_path = os.path.join(folder_path, f"{digit}_{version}.bmp")
            with Image.open(image_path) as img:
                img = img.convert("L")
                img = img.resize((28, 28))
                image = np.array(img)
            image = image / 255.0  # Normalisation des pixels
            if version in test_indices:  # Si l'index fait partie des indices de test
                test_images.append(image)
                test_labels.append(digit)
            else:  # Sinon, l'ajouter à l'ensemble d'entraînement
                train_images.append(image)
                train_labels.append(digit)
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

# Chargement des images d'entraînement et de test
x_train, y_train, x_test, y_test = load_images(train_folder, 10, 9, 1)

# Création du modèle
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(1.5*(28*28), activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(x_train, y_train, epochs=25, batch_size=1)

# Évaluation du modèle sur les données de test final
eval_loss, eval_acc = model.evaluate(x_test, y_test, batch_size=1)
print('Final Test Accuracy: ', eval_acc*100)

# Prédiction sur les données de test final
predictions = model.predict(x_test)

# Affichage des images avec leurs étiquettes prédites
plt.figure(figsize=(10, 10))
for i in range(len(x_test)):
    plt.subplot(5, 2, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Label réel: {y_test[i]}, Prédiction: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.show()

import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Définition des chemins des dossiers contenant les images d'entraînement et d'évaluation
train_folders = ["./Images_Processed_Threshold",
                 "./enzoThreshold",
                 "./colinThreshold",
                 "./jordanThreshold",
                 "./nathanThreshold",
                 "./boliniThreshold"]

test_folder = "./tomaThreshold"

# Fonction pour charger les images
def load_images(folders):
    images = []
    labels = []
    for folder_path in folders:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".bmp"):
                image_path = os.path.join(folder_path, file_name)
                digit = int(file_name.split("_")[0])
                with Image.open(image_path) as img:
                    img = img.convert("L")
                    img = img.resize((28, 28))
                    image = np.array(img)
                image = image / 255.0  # Normalisation des pixels
                images.append(image)
                labels.append(digit)
    if not images:
        print("Aucune image n'a été chargée.")
    else:
        print(f"{len(images)} images ont été chargées.")
    return np.array(images), np.array(labels)

# Définition des valeurs à tester pour la couche Dense
dense_values = [1.1*(28*28), 1.2*(28*28), 1.3*(28*28), 1.4*(28*28), 1.5*(28*28)]

# Chargement des images d'entraînement
x_train, y_train = load_images(train_folders)

# Chargement des images pour évaluation
x_eval, y_eval = load_images([test_folder])

# Boucle sur les différentes valeurs de la couche Dense

for dense_value in dense_values:

    # Création du modèle avec la valeur de dense_value
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(dense_value, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compilation du modèle
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Entraînement du modèle
    print(f"\nEntraînement avec dense_value = {dense_value}")
    history = model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0)

    # Évaluation du modèle sur les données d'évaluation
    eval_loss, eval_acc = model.evaluate(x_eval, y_eval, batch_size=1, verbose=0)
    print(f'Evaluation Accuracy: {eval_acc*100:.2f}%')

    # Prédiction sur les données d'évaluation
    predictions = model.predict(x_eval)

    # Affichage des étiquettes prédites pour les données d'évaluation
    for i in range(len(x_eval)):
        print(f"Image {i+1}: Label réel: {y_eval[i]}, Prédiction: {np.argmax(predictions[i])}")

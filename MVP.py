import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Définition des chemins des dossiers contenant les images d'entraînement et d'évaluation
train_folder = "./Images_Processed_Threshold"
eval_folder = "./enzoThreshold"

# Fonction pour charger les images
def load_images(folder_path):
    images = []
    labels = []
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

# Chargement des images d'entraînement
x_train, y_train = load_images(train_folder)

# Chargement des images pour évaluation
x_eval, y_eval = load_images(eval_folder)

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
history = model.fit(x_train, y_train, epochs=100, batch_size=1)

# Évaluation du modèle sur les données d'évaluation
eval_loss, eval_acc = model.evaluate(x_eval, y_eval, batch_size=1)
print('Evaluation Accuracy: ', eval_acc*100)

# Prédiction sur les données d'évaluation
predictions = model.predict(x_eval)

# Affichage des étiquettes prédites pour les données d'évaluation
for i in range(len(x_eval)):
    print(f"Image {i+1}: Label réel: {y_eval[i]}, Prédiction: {np.argmax(predictions[i])}")

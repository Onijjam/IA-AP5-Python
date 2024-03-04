import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Chargement des données MNIST
(x_train_mnist, y_train_mnist), _ = mnist.load_data()

# Normalisation des pixels
x_train_mnist = x_train_mnist / 255.0

# Fonction pour charger les images de test
def load_test_images(folders):
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

# Définition des dossiers de test
test_folders = ["./enzoThreshold",
                 "./colinThreshold",
                 "./jordanThreshold",
                 "./nathanThreshold",
                 "./boliniThreshold",
                 "./Images_Processed_Threshold",
                 "./tomaThreshold"]

# Chargement des images de test
x_test, y_test = load_test_images(test_folders)

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

# Entraînement du modèle avec les données MNIST
history = model.fit(x_train_mnist, y_train_mnist, epochs=5, batch_size=32, validation_split=0.2)

# Évaluation du modèle sur les données de test
eval_loss, eval_acc = model.evaluate(x_test, y_test, batch_size=32)
print('Evaluation Accuracy: ', eval_acc*100)

# Prédiction sur les données de test
predictions = model.predict(x_test)

# Affichage d'une donnée d'entraînement avec un plot et sauvegarde
plt.imshow(x_train_mnist[0], cmap='gray')
plt.title(f"Label réel: {y_train_mnist[0]}")
plt.axis('off')
plt.savefig('train_example.png')

# Affichage d'une donnée de test avec un plot
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Label réel: {y_test[0]}, Prédiction: {np.argmax(predictions[0])}")
plt.axis('off')
plt.savefig('test_example.png')
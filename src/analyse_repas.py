
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

# Charger le modèle MobileNetV3
print("Chargement de MobileNetV3...")

model = MobileNetV3Large(
    weights="imagenet",
    include_top=False
)

print("Modèle chargé avec succès !")

# Charger une image
image_path = "images/repas.jpg"

image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# Préparer l'image
image = preprocess_input(image)

# Extraire les caractéristiques
features = model.predict(image)

print("\nAnalyse terminée !")
print("Dimensions des caractéristiques :", features.shape)
print("Nombre de valeurs extraites :", features.size)

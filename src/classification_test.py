
import numpy as np
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.utils import load_img, img_to_array

# Charger MobileNetV3 avec sa couche de classification
print("Chargement du modèle...")

model = MobileNetV3Large(
    weights="imagenet",
    include_top=True,
    input_shape=(224, 224, 3)
)

# Charger la photo
image = load_img(
    "images/repas.jpg",
    target_size=(224, 224)
)

image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# Prétraiter l'image
image = preprocess_input(image)

# Effectuer la prédiction
predictions = model.predict(image)

# Afficher les 5 meilleures prédictions
results = decode_predictions(predictions, top=5)[0]

print("\nRésultats de la classification :")

for _, label, score in results:
    print(f"{label} : {score * 100:.2f} %")

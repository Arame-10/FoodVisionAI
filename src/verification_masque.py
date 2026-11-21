
from PIL import Image
import numpy as np

mask_path = r"C:\Users\diopa\Downloads\FoodVisionAI\dataset\UECFoodPixComplete\UECFOODPIXCOMPLETE\data\UECFoodPIXCOMPLETE\train\mask\336.png"

mask = np.array(Image.open(mask_path))
annotations = mask[:, :, 0]

# Création d'une image colorée
couleurs = {
    0: [0, 0, 0],
    4: [255, 0, 0],
    101: [0, 255, 0]
}

resultat = np.zeros((*annotations.shape, 3), dtype=np.uint8)

for classe, couleur in couleurs.items():
    resultat[annotations == classe] = couleur

Image.fromarray(resultat).save("results/masque_336_couleurs.png")

print("Masque coloré enregistré dans results/")

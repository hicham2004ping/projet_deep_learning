import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Les émotions, dans le même ordre que pendant l'entraînement (ordre alphabétique des dossiers)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 1. Charger le modèle sauvegardé
model = tf.keras.models.load_model("emotion_model.keras")

# 2. Charger une photo (mets ici le chemin d'une vraie image de ton dossier test)
chemin = "test/happy/PrivateTest_2260082.jpg"
img = tf.keras.utils.load_img(chemin, target_size=(96, 96))   # même taille qu'à l'entraînement
x = tf.keras.utils.img_to_array(img)       # l'image devient un tableau de nombres
x = np.expand_dims(x, axis=0)              # le modèle attend un "paquet" d'images, ici un paquet de 1

# 3. Demander au modèle
probas = model.predict(x)[0]               # 7 pourcentages, un par émotion
emotion = class_names[np.argmax(probas)]   # la case avec le plus grand pourcentage

# 4. Afficher le résultat
print("Émotion prédite :", emotion)
for nom, p in zip(class_names, probas):
    print(f"  {nom:10s} {p*100:5.1f} %")

plt.imshow(img)
plt.title(f"{emotion} ({probas.max()*100:.0f} %)")
plt.axis("off")
plt.show()
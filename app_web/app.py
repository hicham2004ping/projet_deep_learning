import os
import uuid

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory

# ---------- Configuration ----------
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_FR = {
    'angry': 'Colère', 'disgust': 'Dégoût', 'fear': 'Peur', 'happy': 'Joie',
    'neutral': 'Neutre', 'sad': 'Tristesse', 'surprise': 'Surprise',
}
IMG_SIZE = 96                   # même taille qu'à l'entraînement

# Tous les chemins sont calculés à partir du dossier où se trouve app.py,
# pour que le serveur marche quel que soit le dossier de lancement.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")    # dossier où les images sont stockées
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR)
app.json.sort_keys = False      # garder l'ordre des émotions dans la réponse


# ---------- Lecture / écriture d'images ----------
# Sur Windows, cv2.imread / cv2.imwrite échouent si le chemin contient un accent
# (ex. "Ingénieur"). On lit et on écrit donc les fichiers avec numpy, et OpenCV
# ne fait que décoder / encoder les octets en mémoire.
def lire_image(chemin):
    return cv2.imdecode(np.fromfile(chemin, dtype=np.uint8), cv2.IMREAD_COLOR)


def ecrire_image(chemin, image):
    ok, octets = cv2.imencode(".jpg", image)
    if ok:
        octets.tofile(chemin)


# ---------- Chargement du modèle ----------
# On cherche emotion_model.keras dans le dossier de app.py, puis dans le dossier parent.
def trouver_modele():
    for dossier in (BASE_DIR, os.path.dirname(BASE_DIR)):
        chemin = os.path.join(dossier, "emotion_model.keras")
        if os.path.exists(chemin):
            return chemin
    raise FileNotFoundError(
        "emotion_model.keras introuvable : mets-le à côté de app.py "
        "ou dans le dossier juste au-dessus.")


# Le modèle est chargé UNE seule fois au démarrage du serveur
model = tf.keras.models.load_model(trouver_modele())


# ---------- Détecteur de visages ----------
# Même problème d'accents : on lit le fichier XML avec Python, puis on le donne
# à OpenCV depuis la mémoire au lieu de lui donner le chemin.
def charger_detecteur():
    chemin = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(chemin):
        raise RuntimeError(
            "Détecteur de visages introuvable dans OpenCV. "
            "Installe OpenCV 4 avec : pip install \"opencv-python<5\"")
    with open(chemin, encoding="utf-8") as f:
        contenu = f.read()
    stockage = cv2.FileStorage(contenu, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    det = cv2.CascadeClassifier()
    if not det.read(stockage.getFirstTopLevelNode()):
        raise RuntimeError("Impossible de charger le détecteur de visages.")
    return det


detector = charger_detecteur()


def predire_visage(visage_gris):
    """Prépare un visage (en gris) comme à l'entraînement et renvoie les probabilités."""
    visage = cv2.resize(visage_gris, (IMG_SIZE, IMG_SIZE))
    visage = cv2.cvtColor(visage, cv2.COLOR_GRAY2RGB)
    entree = np.expand_dims(visage.astype("float32"), axis=0)
    # model(...) est beaucoup plus rapide que model.predict(...) pour une seule image,
    # ce qui compte pour le temps réel
    return model(entree, training=False).numpy()[0]


def resultat(probas):
    """Transforme les 7 probabilités en un dictionnaire lisible."""
    meilleur = CLASS_NAMES[int(np.argmax(probas))]
    return {
        "emotion": EMOTIONS_FR[meilleur],
        "confiance": round(float(probas.max()) * 100, 1),
        "probabilites": {EMOTIONS_FR[n]: round(float(p) * 100, 1) for n, p in zip(CLASS_NAMES, probas)},
    }


# ---------- Pages ----------
@app.route("/")
def accueil():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/uploads/<nom>")
def image_stockee(nom):
    return send_from_directory(UPLOAD_DIR, nom)


# ---------- API appelée par le JavaScript ----------
@app.route("/predict", methods=["POST"])
def predict():
    fichier = request.files.get("image")
    if fichier is None or fichier.filename == "":
        return jsonify({"erreur": "Aucune image reçue"}), 400

    # 1. Stocker l'image avec un nom unique
    extension = os.path.splitext(fichier.filename)[1].lower() or ".jpg"
    nom = f"{uuid.uuid4().hex}{extension}"
    chemin = os.path.join(UPLOAD_DIR, nom)
    fichier.save(chemin)

    photo = lire_image(chemin)
    if photo is None:
        return jsonify({"erreur": "Fichier illisible, envoie une image JPG ou PNG"}), 400
    gris = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

    # 2. Chercher les visages
    visages = detector.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    resultats = []
    if len(visages) == 0:
        # Aucun visage détecté : on suppose que l'image est déjà un visage recadré
        # (c'est le cas des images du dataset FER2013)
        resultats.append(resultat(predire_visage(gris)))
    else:
        for (x, y, w, h) in visages:
            r = resultat(predire_visage(gris[y:y + h, x:x + w]))
            resultats.append(r)
            # Dessiner le cadre et l'émotion sur l'image
            cv2.rectangle(photo, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(photo, f"{r['emotion']} {r['confiance']:.0f}%", (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    # 3. Sauvegarder l'image annotée
    nom_annote = f"annote_{os.path.splitext(nom)[0]}.jpg"
    ecrire_image(os.path.join(UPLOAD_DIR, nom_annote), photo)

    return jsonify({
        "image": f"/uploads/{nom_annote}",
        "nb_visages": int(len(visages)),
        "resultats": resultats,
    })


# ---------- Temps réel : une image de la webcam ----------
# Appelée plusieurs fois par seconde par la page. L'image n'est PAS stockée :
# elle est analysée en mémoire puis oubliée.
@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    fichier = request.files.get("frame")
    if fichier is None:
        return jsonify({"erreur": "Aucune image reçue"}), 400

    octets = np.frombuffer(fichier.read(), dtype=np.uint8)
    image = cv2.imdecode(octets, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"erreur": "Image illisible"}), 400
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    visages = detector.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    resultats = []
    for (x, y, w, h) in visages:
        r = resultat(predire_visage(gris[y:y + h, x:x + w]))
        r["cadre"] = [int(x), int(y), int(w), int(h)]   # position du visage dans l'image
        resultats.append(r)

    return jsonify({"largeur": int(image.shape[1]), "hauteur": int(image.shape[0]), "visages": resultats})


if __name__ == "__main__":
    # use_reloader=False : évite de charger TensorFlow deux fois au démarrage
    app.run(debug=True, use_reloader=False)
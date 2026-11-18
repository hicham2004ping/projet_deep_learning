# Reconnaissance automatique des émotions faciales
 
Projet de Deep Learning — **5IIR · DDSI — EMSI**
 
Application qui reconnaît l’émotion d’un visage à partir d’une **image importée** ou de la **caméra en temps réel**, grâce à un réseau de neurones convolutif (CNN) entraîné par **apprentissage par transfert** (MobileNetV2 pré-entraîné sur ImageNet).
 
Les 7 émotions reconnues : **colère, dégoût, peur, joie, neutre, tristesse, surprise**.
 
**Réalisé par :** BAKHALED Mohammed · MOUL RAGOUBA Hicham
 
---
 
## Sommaire
 
1. [Fonctionnement](#fonctionnement)
2. [Structure du projet](#structure-du-projet)
3. [Prérequis](#prérequis)
4. [Installation](#installation)
5. [Données](#données)
6. [Exécution](#exécution)
7. [Résultats](#résultats)
8. [Problèmes fréquents](#problèmes-fréquents)
9. [Éthique et vie privée](#éthique-et-vie-privée)
---
 
## Fonctionnement
 
```
Image ou caméra → Détection du visage (OpenCV) → Recadrage, gris → RGB, 96×96 → CNN → Émotion + %
```
 
- **Modèle** : MobileNetV2 gelé (poids ImageNet) + `GlobalAveragePooling2D` + `Dropout(0.3)` + `Dense(7, softmax)`.
- **Données** : FER2013 (visages 48×48 en niveaux de gris, 7 classes).
- **Application web** : serveur Flask (Python) + page HTML/JavaScript, avec deux modes :
  - **Importer une image** : l’image est analysée et renvoyée avec un cadre et l’émotion sur chaque visage ;
  - **Caméra en temps réel** : la webcam est analysée en continu, avec un lissage des prédictions et l’affichage des analyses par seconde et de la latence.
---
 
## Structure du projet
 
```
projet-deep-learning/
├── train/                     # données d'entraînement (à télécharger, non versionnées)
│   ├── angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
├── test/                      # données de test (mêmes 7 dossiers)
├── train_model.py             # entraînement du modèle
├── fiche_experimentale.py     # évaluation : métriques + matrice de confusion
├── emotion_model.keras        # modèle entraîné (créé par train_model.py)
├── history.json               # historique de l'entraînement (créé par train_model.py)
├── fiche/                     # résultats de l'évaluation (créé par fiche_experimentale.py)
├── app_web/
│   ├── app.py                 # serveur Flask
│   ├── static/
│   │   └── index.html         # interface web (HTML + CSS + JavaScript)
│   └── uploads/               # images importées (créé automatiquement, non versionné)
├── requirements.txt
├── .gitignore
└── README.md
```
 
---
 
## Prérequis
 
- **Python 3.10, 3.11 ou 3.12** (versions compatibles avec TensorFlow).
- Une **webcam** pour le mode temps réel.
- Un navigateur récent (Chrome, Edge ou Firefox).
> Sous Windows, TensorFlow utilise le processeur (CPU) : l’entraînement fonctionne, mais il est plus lent qu’avec une carte graphique. Le message `TensorFlow GPU support is not available on native Windows` est normal.
 
---
 
## Installation
 
1. **Cloner le projet** puis se placer dans son dossier :
```bash
git clone <url-du-depot>
cd projet-deep-learning
```
 
2. **Créer et activer un environnement virtuel** :
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```
 
3. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```
 
   Ou, au minimum :
 
```bash
pip install tensorflow flask scikit-learn matplotlib "opencv-python<5"
```
 
   > ⚠️ Il faut **OpenCV 4** (`opencv-python<5`) : les versions 5.x ne contiennent plus le détecteur de visages (Haar Cascade).
 
---
 
## Données
 
1. Télécharger le jeu de données **FER2013** depuis Kaggle (version en dossiers d’images).
2. Placer les dossiers `train/` et `test/` **à la racine du projet**, chacun avec les 7 sous-dossiers :
   `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
Il n’y a pas de fichier CSV : **le nom du dossier sert d’étiquette**. Keras lit ces dossiers dans l’ordre alphabétique et donne à chacun un numéro de 0 à 6.
 
| Dossier | Images | Utilisation |
|---|---|---|
| `train/` (85 %) | ≈ 24 400 | entraînement |
| `train/` (15 %) | ≈ 4 300 | validation pendant l’entraînement |
| `test/` | 7 178 | évaluation finale (images jamais vues) |
 
---
 
## Exécution
 
Toutes les commandes se lancent **depuis la racine du projet**, avec l’environnement virtuel activé. Dans PyCharm : clic droit sur le fichier → **Run**.
 
### 1. Entraîner le modèle
 
```bash
python train_model.py
```
 
Durée : de quelques minutes à plus d’une heure selon l’ordinateur (10 epochs).
Fichiers créés :
 
- `emotion_model.keras` : le modèle entraîné ;
- `history.json` : la perte et l’accuracy à chaque epoch.
### 2. Évaluer le modèle
 
```bash
pip install scikit-learn   # si ce n'est pas déjà fait
python fiche_experimentale.py
```
 
Fichiers créés dans `fiche/` :
 
| Fichier | Contenu |
|---|---|
| `rapport.txt` | loss, accuracy, précision / rappel / F1 par émotion, matrice de confusion |
| `tableau_metriques.csv` | tableau des métriques (s’ouvre dans Excel) |
| `matrice_confusion.png` | matrice de confusion |
| `fiche_experimentale.pdf` | fiche expérimentale complète |
 
### 3. Lancer l’application web
 
Le serveur cherche `emotion_model.keras` à côté de `app.py` **ou** dans le dossier juste au-dessus (la racine du projet) : il n’est donc pas nécessaire de le déplacer.
 
```bash
python app_web/app.py
```
 
Attendre le message :
 
```
Running on http://127.0.0.1:5000
```
 
puis ouvrir **http://127.0.0.1:5000** dans un navigateur.
 
- **Onglet « Importer une image »** : choisir une image, puis cliquer sur **Analyser l’image**.
- **Onglet « Caméra en temps réel »** : cliquer sur **Démarrer la caméra**, puis autoriser l’accès à la caméra dans le navigateur.
Pour arrêter le serveur : `Ctrl + C` dans le terminal (ou le bouton Stop de PyCharm).
 
#### Routes du serveur
 
| Route | Méthode | Rôle |
|---|---|---|
| `/` | GET | page web |
| `/predict` | POST | analyse une image importée (enregistrée dans `uploads/`) |
| `/predict_frame` | POST | analyse une image de la webcam (non enregistrée) |
| `/uploads/<nom>` | GET | renvoie une image annotée |
 
---
 
## Résultats
 
Évaluation sur les **7 178 images** du dossier `test` :
 
| Métrique | Valeur |
|---|---|
| Accuracy | **47,7 %** |
| F1-score macro | 40,5 % |
| Loss | 1,40 |
 
| Émotion | Précision | Rappel | F1-score | Images |
|---|---|---|---|---|
| Colère | 38,5 % | 31,9 % | 34,9 % | 958 |
| Dégoût | 38,9 % | 12,6 % | 19,0 % | 111 |
| Peur | 44,9 % | 8,6 % | 14,4 % | 1 024 |
| Joie | 60,0 % | 69,6 % | **64,4 %** | 1 774 |
| Neutre | 42,0 % | 48,4 % | 45,0 % | 1 233 |
| Tristesse | 35,6 % | 49,7 % | 41,5 % | 1 247 |
| Surprise | 60,7 % | 67,7 % | **64,0 %** | 831 |
 
La joie et la surprise sont bien reconnues. La peur est souvent confondue avec la tristesse, et le dégoût manque d’exemples d’entraînement.
 
**Pistes d’amélioration :** fine-tuning des dernières couches de MobileNetV2, augmentation de données, poids de classes pour les émotions rares.
 
---
 
## Problèmes fréquents
 
| Problème | Solution |
|---|---|
| `Détecteur de visages introuvable` | Installer OpenCV 4 : `pip install "opencv-python<5"` |
| Erreur d’ouverture de fichier sous Windows alors que le fichier existe | Chemin contenant un accent (ex. `Ingénieur`) : la version actuelle de `app.py` gère ce cas ; sinon, déplacer le projet dans un dossier sans accent |
| `emotion_model.keras introuvable` | Lancer d’abord `train_model.py`, ou placer le modèle à côté de `app.py` |
| `history.json introuvable` | Relancer `train_model.py` (il enregistre l’historique à la fin de l’entraînement) |
| « Aucun visage détecté » | Visage de face, bien éclairé et assez grand dans l’image |
| La caméra ne démarre pas | Autoriser la caméra dans le navigateur ; fermer les autres applications qui l’utilisent ; ouvrir la page via `http://127.0.0.1:5000` |
| Installation de TensorFlow impossible | Vérifier la version de Python (`python --version`) : utiliser 3.10, 3.11 ou 3.12 |
| Messages `oneDNN` / `cpu_feature_guard` au démarrage | Simples informations, à ignorer |
 
---
 
## Éthique et vie privée
 
- Les images de la **caméra ne sont jamais enregistrées** : elles sont analysées en mémoire puis oubliées.
- Les images importées sont stockées localement dans `app_web/uploads/`, **exclu de Git**.
- Le système ne cherche **pas à identifier** les personnes.
- Une expression affichée n’est pas forcément l’émotion ressentie : les résultats sont une **estimation**, à ne jamais utiliser pour prendre une décision concernant une personne.
- Toujours obtenir le **consentement** des personnes filmées.
---
 
## Technologies
 
Python · TensorFlow / Keras · OpenCV · Flask · scikit-learn · Matplotlib · HTML / CSS / JavaScript
 
## Références principales
 
- M. Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018.
- I. Goodfellow et al., *Challenges in Representation Learning: A report on three machine learning contests* (FER2013), ICONIP 2013.
- P. Viola et M. Jones, *Rapid Object Detection using a Boosted Cascade of Simple Features*, CVPR 2001.
 
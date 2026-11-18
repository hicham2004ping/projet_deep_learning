"""
Fiche expérimentale de MON modèle (emotion_model.keras).

À mettre à la racine du projet, à côté de :
    emotion_model.keras   → le modèle entraîné
    test/                 → le dossier test (7 sous-dossiers d'émotions)
    history.json          → l'historique de l'entraînement (pour les courbes)

Crée le dossier "fiche/" avec :
    fiche_experimentale.pdf   → la fiche complète (3 pages) à mettre dans le rapport
    tableau_metriques.csv     → le tableau des métriques (s'ouvre dans Excel / se copie dans Word)
    courbes_loss_accuracy.png → courbes training / validation
    matrice_confusion.png     → matrice de confusion
    rapport.txt               → tous les chiffres en texte
"""
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELE = os.path.join(BASE_DIR, "emotion_model.keras")
DOSSIER_TEST = os.path.join(BASE_DIR, "test")
HISTORIQUE = os.path.join(BASE_DIR, "history.json")
SORTIE = os.path.join(BASE_DIR, "fiche")
os.makedirs(SORTIE, exist_ok=True)

NOMS_FR = {'angry': 'Colère', 'disgust': 'Dégoût', 'fear': 'Peur', 'happy': 'Joie',
           'neutral': 'Neutre', 'sad': 'Tristesse', 'surprise': 'Surprise'}


# ---------------------------------------------------------------------------
# 1. Calcul des métriques
# ---------------------------------------------------------------------------
def evaluer():
    model = tf.keras.models.load_model(MODELE)
    # shuffle=False : les images restent dans l'ordre, pour comparer prédictions et vraies étiquettes
    test = tf.keras.utils.image_dataset_from_directory(
        DOSSIER_TEST, image_size=(96, 96), batch_size=64, label_mode="categorical", shuffle=False)
    noms = [NOMS_FR.get(n, n) for n in test.class_names]

    loss, accuracy = model.evaluate(test, verbose=1)
    probas = model.predict(test, verbose=1)
    y_pred = np.argmax(probas, axis=1)
    y_vrai = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test])

    precision, rappel, f1, support = precision_recall_fscore_support(y_vrai, y_pred, zero_division=0)
    macro = precision_recall_fscore_support(y_vrai, y_pred, average="macro", zero_division=0)
    pondere = precision_recall_fscore_support(y_vrai, y_pred, average="weighted", zero_division=0)

    return {
        "noms": noms,
        "nb_images": len(y_vrai),
        "loss": float(loss),
        "accuracy": float(accuracy),
        "par_classe": list(zip(noms, precision, rappel, f1, support)),
        "macro": macro[:3],
        "pondere": pondere[:3],
        "matrice": confusion_matrix(y_vrai, y_pred),
        "rapport_sklearn": classification_report(y_vrai, y_pred, target_names=noms, digits=3, zero_division=0),
        "nb_parametres": model.count_params(),
    }


# ---------------------------------------------------------------------------
# 2. Les figures
# ---------------------------------------------------------------------------
def figure_tableaux(r, historique):
    """Page 1 : tableau des métriques globales + tableau par émotion."""
    fig = plt.figure(figsize=(8.27, 11.69))     # format A4
    fig.suptitle("Fiche expérimentale — Reconnaissance des émotions faciales", fontsize=15, y=0.97)

    globales = [
        ["Modèle", "MobileNetV2 (ImageNet) + couche à 7 émotions"],
        ["Images de test", f"{r['nb_images']}"],
        ["Loss (test)", f"{r['loss']:.4f}"],
        ["Accuracy (test)", f"{r['accuracy'] * 100:.2f} %"],
        ["Précision (moyenne macro)", f"{r['macro'][0] * 100:.2f} %"],
        ["Rappel (moyenne macro)", f"{r['macro'][1] * 100:.2f} %"],
        ["F1-score (moyenne macro)", f"{r['macro'][2] * 100:.2f} %"],
        ["F1-score (moyenne pondérée)", f"{r['pondere'][2] * 100:.2f} %"],
        ["Nombre de paramètres", f"{r['nb_parametres']:,}".replace(",", " ")],
    ]
    if historique:
        globales += [
            ["Epochs d'entraînement", f"{len(historique['loss'])}"],
            ["Meilleure accuracy de validation", f"{max(historique['val_accuracy']) * 100:.2f} %"],
            ["Plus petite loss de validation", f"{min(historique['val_loss']):.4f}"],
        ]

    ax1 = fig.add_axes([0.08, 0.55, 0.84, 0.36])
    ax1.axis("off")
    ax1.set_title("Tableau 1 — Métriques globales", fontsize=12, loc="left")
    t1 = ax1.table(cellText=globales, colLabels=["Métrique", "Valeur"], loc="upper center",
                   cellLoc="left", colWidths=[0.45, 0.55])
    t1.auto_set_font_size(False)
    t1.set_fontsize(10)
    t1.scale(1, 1.6)

    lignes = [[n, f"{p * 100:.1f} %", f"{rc * 100:.1f} %", f"{f * 100:.1f} %", f"{int(s)}"]
              for n, p, rc, f, s in r["par_classe"]]
    lignes.append(["Moyenne macro", f"{r['macro'][0] * 100:.1f} %", f"{r['macro'][1] * 100:.1f} %",
                   f"{r['macro'][2] * 100:.1f} %", f"{r['nb_images']}"])

    ax2 = fig.add_axes([0.08, 0.08, 0.84, 0.40])
    ax2.axis("off")
    ax2.set_title("Tableau 2 — Métriques par émotion", fontsize=12, loc="left")
    t2 = ax2.table(cellText=lignes, colLabels=["Émotion", "Précision", "Rappel", "F1-score", "Images"],
                   loc="upper center", cellLoc="center")
    t2.auto_set_font_size(False)
    t2.set_fontsize(10)
    t2.scale(1, 1.6)

    for t in (t1, t2):                          # en-têtes en gras sur fond bleu clair
        for (ligne, _), cellule in t.get_celld().items():
            if ligne == 0:
                cellule.set_facecolor("#dbeafe")
                cellule.set_text_props(weight="bold")
    return fig


def figure_courbes(historique):
    """Page 2 : courbes loss et accuracy, entraînement vs validation."""
    epochs = range(1, len(historique["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))
    fig.suptitle("Courbes d'entraînement", fontsize=15)

    ax1.plot(epochs, historique["loss"], "o-", label="Training loss")
    ax1.plot(epochs, historique["val_loss"], "o-", label="Validation loss")
    ax1.set_title("Loss (erreur) — plus c'est bas, mieux c'est")
    ax1.set_ylabel("Loss")

    ax2.plot(epochs, [a * 100 for a in historique["accuracy"]], "o-", label="Training accuracy")
    ax2.plot(epochs, [a * 100 for a in historique["val_accuracy"]], "o-", label="Validation accuracy")
    ax2.set_title("Accuracy — plus c'est haut, mieux c'est")
    ax2.set_ylabel("Accuracy (%)")

    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def figure_matrice(r):
    """Page 3 : matrice de confusion en nombres et en pourcentages."""
    matrice, noms = r["matrice"], r["noms"]
    pourcent = matrice / np.maximum(matrice.sum(axis=1, keepdims=True), 1) * 100

    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    fig.suptitle("Matrice de confusion (lignes = vraie émotion, colonnes = prédiction)", fontsize=13)
    for ax, valeurs, titre, fmt in [(axes[0], matrice, "En nombre d'images", "{:.0f}"),
                                    (axes[1], pourcent, "En % de chaque vraie émotion", "{:.0f}%")]:
        im = ax.imshow(valeurs, cmap="Blues")
        ax.set_title(titre)
        ax.set_xticks(range(len(noms)), noms, rotation=45, ha="right")
        ax.set_yticks(range(len(noms)), noms)
        ax.set_xlabel("Émotion prédite")
        ax.set_ylabel("Vraie émotion")
        seuil = valeurs.max() / 2
        for i in range(len(noms)):
            for j in range(len(noms)):
                ax.text(j, i, fmt.format(valeurs[i, j]), ha="center", va="center",
                        color="white" if valeurs[i, j] > seuil else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# 3. Programme principal
# ---------------------------------------------------------------------------
def main():
    historique = None
    if os.path.exists(HISTORIQUE):
        with open(HISTORIQUE, encoding="utf-8") as f:
            historique = json.load(f)
    else:
        print("⚠ history.json introuvable : la fiche sera faite SANS les courbes d'entraînement.")

    r = evaluer()

    # Fichiers séparés (pour les coller dans le rapport)
    with open(os.path.join(SORTIE, "tableau_metriques.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Émotion", "Précision (%)", "Rappel (%)", "F1-score (%)", "Images"])
        for n, p, rc, f1, s in r["par_classe"]:
            w.writerow([n, round(p * 100, 1), round(rc * 100, 1), round(f1 * 100, 1), int(s)])
        w.writerow(["Moyenne macro", *[round(v * 100, 1) for v in r["macro"]], r["nb_images"]])
        w.writerow([])
        w.writerow(["Loss (test)", round(r["loss"], 4)])
        w.writerow(["Accuracy (test) (%)", round(r["accuracy"] * 100, 2)])

    texte = (f"Loss (test)     : {r['loss']:.4f}\n"
             f"Accuracy (test) : {r['accuracy'] * 100:.2f} %\n\n"
             f"{r['rapport_sklearn']}\n"
             f"Matrice de confusion (lignes = vraie émotion, colonnes = prédiction) :\n"
             f"{r['noms']}\n{r['matrice']}\n")
    with open(os.path.join(SORTIE, "rapport.txt"), "w", encoding="utf-8") as f:
        f.write(texte)
    print("\n" + texte)

    pages = [figure_tableaux(r, historique)]
    if historique:
        fig_c = figure_courbes(historique)
        fig_c.savefig(os.path.join(SORTIE, "courbes_loss_accuracy.png"), dpi=150)
        pages.append(fig_c)
    fig_m = figure_matrice(r)
    fig_m.savefig(os.path.join(SORTIE, "matrice_confusion.png"), dpi=150)
    pages.append(fig_m)

    # La fiche complète en PDF
    with PdfPages(os.path.join(SORTIE, "fiche_experimentale.pdf")) as pdf:
        for fig in pages:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Fiche enregistrée dans : {SORTIE}")


if __name__ == "__main__":
    main()
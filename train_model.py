import json

import tensorflow as tf
from tensorflow.keras import layers

# ---------- 1. Charger les données ----------
train = tf.keras.utils.image_dataset_from_directory(
    "train", image_size=(96, 96), batch_size=64, label_mode="categorical",
    validation_split=0.15, subset="training", seed=42)

val = tf.keras.utils.image_dataset_from_directory(
    "train", image_size=(96, 96), batch_size=64, label_mode="categorical",
    validation_split=0.15, subset="validation", seed=42)

print(train.class_names)

# ---------- 2. Construire le modèle ----------
base = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3), include_top=False, weights="imagenet")
base.trainable = False

model = tf.keras.Sequential([
    layers.Rescaling(1/127.5, offset=-1),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(7, activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ---------- 3. Entraîner ----------
history = model.fit(train, validation_data=val, epochs=10)

# ---------- 4. Sauvegarder le modèle ----------
model.save("emotion_model.keras")
print("Modèle sauvegardé : emotion_model.keras")

# ---------- 5. NOUVEAU : sauvegarder l'historique (pour les courbes) ----------
# history.history contient, pour chaque epoch : loss, accuracy, val_loss, val_accuracy
with open("history.json", "w") as f:
    json.dump(history.history, f, indent=2)
print("Historique sauvegardé : history.json")
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ---------- 1. Charger les données ----------
train = tf.keras.utils.image_dataset_from_directory(
    "train", image_size=(96, 96), batch_size=64, label_mode="categorical",
    validation_split=0.15, subset="training", seed=42)

val = tf.keras.utils.image_dataset_from_directory(
    "train", image_size=(96, 96), batch_size=64, label_mode="categorical",
    validation_split=0.15, subset="validation", seed=42)

test = tf.keras.utils.image_dataset_from_directory(
    "test", image_size=(96, 96), batch_size=64, label_mode="categorical",
    shuffle=False)

class_names = train.class_names
print(class_names)

# ---------- 2. Regarder quelques images ----------
images, labels = next(iter(train))
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i].numpy().argmax()])
    plt.axis("off")
plt.show()   # une fenêtre s'ouvre : ferme-la pour que le programme continue

# ---------- 3. Construire le modèle ----------
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

# ---------- 4. Entraîner ----------
history = model.fit(train, validation_data=val, epochs=3)

# ---------- 5. Sauvegarder le modèle ----------
model.save("emotion_model.keras")
print("Modèle sauvegardé !")
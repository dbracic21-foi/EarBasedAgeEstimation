import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score


MODEL_PATH = "./models/age_prediction_model.keras"
DATA_DIR = "data/processed"
TEST_CSV = "./data/test_labels.csv"
RESULTS_DIR = "result"
def age_category(age):
    if age < 20:
        return 0
    elif age < 40:
        return 1
    elif age < 60:
        return 2
    else:
        return 3

model = load_model(MODEL_PATH)
test_df = pd.read_csv("data/test_labels.csv")
test_df["age_category"] = test_df["age"].apply(age_category).astype(str)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=DATA_DIR,
    x_col="image_name",
    y_col="age_category",
    target_size=(128, 128),
    class_mode="sparse",
    batch_size=32,
    shuffle=False
)

# 📌 Učitavanje treniranog modela
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model učitan:", MODEL_PATH)

# 📌 Predviđanje rezultata
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# 📌 Stvarne oznake
true_classes = test_generator.classes

# 📌 Izračun metrike
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average="macro", zero_division=1)
recall = recall_score(true_classes, predicted_classes, average="macro", zero_division=1)
f1 = f1_score(true_classes, predicted_classes, average="macro")

# 📌 Ispis rezultata
print("\n🎯 **Evaluacija Modela:**")
print(f"✅ Accuracy: {accuracy:.2%}")
print(f"✅ Precision: {precision:.2%}")
print(f"✅ Recall: {recall:.2%}")
print(f"✅ F1 Score: {f1:.2%}")

# 📌 Spremanje rezultata u tekstualni fajl
with open(os.path.join(RESULTS_DIR, "evaluation.txt"), "w") as f:
    f.write("=======================================\n")
    f.write("🎯 EVALUACIJA MODELA\n")
    f.write("=======================================\n")
    f.write(f"✅ Accuracy: {accuracy:.2%}\n")
    f.write(f"✅ Precision: {precision:.2%}\n")
    f.write(f"✅ Recall: {recall:.2%}\n")
    f.write(f"✅ F1 Score: {f1:.2%}\n")
    f.write("=======================================\n")

# 📌 Kreiranje grafičkog prikaza predikcija
plt.figure(figsize=(8,6))
plt.scatter(true_classes, predicted_classes, color="blue", label="Predikcija")
plt.plot([0, 3], [0, 3], linestyle="dashed", color="red", label="Idealna predikcija")
plt.xlabel("Stvarna dobna skupina")
plt.ylabel("Predviđena dobna skupina")
plt.legend()
plt.title("Stvarna vs. Predviđena dobna skupina")
plt.savefig(os.path.join(RESULTS_DIR, "predictions_plot.png"))
plt.close()

print("📊 Rezultati spremljeni u 'result/' folder.")
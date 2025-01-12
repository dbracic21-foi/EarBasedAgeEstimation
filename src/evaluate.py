import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score


MODEL_PATH = "./models/age_prediction_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

test_df = pd.read_csv("data/test_labels.csv")
DATA_DIR = "data/processed"

y_true = []
y_pred = []

for index, row in test_df.iterrows():
    image_path = f"{DATA_DIR}/{row['image_name']}"
    if not os.path.exists(image_path):
        continue

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128,128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    predicted_age = model.predict(image)[0][0]
    y_true.append(row["age"])
    y_pred.append(predicted_age)
    
y_true = np.array(y_true, dtype=int)
y_pred = np.round(y_pred).astype(int)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

def age_category(age):
    if age < 20:
        return 0
    elif age < 40:
        return 1
    elif age < 60:
        return 2
    else:
        return 3

y_true_cat = np.array([age_category(age) for age in y_true])
y_pred_cat = np.array([age_category(age) for age in y_pred])

accuracy = accuracy_score(y_true_cat, y_pred_cat)
precision = precision_score(y_true_cat, y_pred_cat, average="weighted")
recall = recall_score(y_true_cat, y_pred_cat, average="weighted")
f1 = f1_score(y_true_cat, y_pred_cat, average="weighted")

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

with open("./result/evaluation.txt", "w") as f:
    f.write(f"Mean Absolute Error: {mae:.2f}\n")
    f.write(f"R2 Score: {r2:.2f}\n")
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall: {recall:.2f}\n")
    f.write(f"F1 Score: {f1:.2f}\n")
    
plt.figure(figsize=(10, 5))
plt.scatter(y_true, y_pred, alpha=0.5, color="blue", label="Predikcija")
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="--", label="Idealna predikcija")
plt.xlabel("Stvarna dob")
plt.ylabel("Predviđena dob")
plt.title("Stvarna vs. Predviđena dob")
plt.legend()
plt.savefig("result/predictions_plot.png")
plt.show()
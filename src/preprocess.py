import os
import cv2
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def preprocess(image_path,output_path):
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.resize(image, (128, 128))
    cv2.imwrite(output_path,image)

csv_path = "data/ear_age_labels.csv"
df = pd.read_csv(csv_path)

for index, row in df.iterrows():
    input_path = os.path.join(RAW_DIR,row["image_name"])
    output_path = os.path.join(PROCESSED_DIR,row["image_name"])
    if os.path.exists(input_path):
        preprocess(input_path,output_path)

print("Preprocessing done!")

import os
import pandas as pd

csv_path = "data/ear_age_labels.csv"
df = pd.read_csv(csv_path)

processed_dir = "data/processed/"

existing_images = set(os.listdir(processed_dir))
df_filtered = df[df["image_name"].isin(existing_images)]

df_filtered.to_csv("data/filtered_labels.csv", index=False)


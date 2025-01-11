import pandas as pd
from sklearn.model_selection import train_test_split


csv_path = "data/filtered_labels.csv"
df = pd.read_csv(csv_path)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_csv_path = "data/train_labels.csv"
test_csv_path = "data/test_labels.csv"

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"Data are shared: \nTrain: {len(train_df)} samples \nTest: {len(test_df)} samples")
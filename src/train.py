import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd 
import os

DATA_DIR = "data/processed"
TRAIN_CSV = "data/train_labels.csv"
TEST_CSV = "data/test_labels.csv"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=DATA_DIR,
    x_col="image_name",
    y_col="age",
    target_size=(128,128),
    class_mode="raw",
    batch_size=32
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=DATA_DIR,
    x_col="image_name",
    y_col="age",
    target_size=(128,128),
    class_mode="raw",
    batch_size=32
)

model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128,(3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1)
]) 
model.compile(optimizer="adam", loss="mean_squared_error",metrics=["mae"])

history = model.fit(train_generator, validation_data=test_generator, epochs=10)

MODEL_PATH = "models/age_prediction_model.keras"
model.save(MODEL_PATH)

print("Model saved at: ", MODEL_PATH)
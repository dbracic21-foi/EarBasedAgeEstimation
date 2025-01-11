import tensorflow as tf

MODEL_PATH = "models/age_prediction_model.keras"  # Ako si koristio .keras

model = tf.keras.models.load_model(MODEL_PATH)

model.summary()

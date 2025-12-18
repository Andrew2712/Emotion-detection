import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "saved_models/emotion_cnn.h5"
EMOTIONS = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

model = tf.keras.models.load_model(MODEL_PATH)

def predict_emotion(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48,48))
    img = img.reshape(1,48,48,1) / 255.0

    prediction = model.predict(img)
    emotion = EMOTIONS[np.argmax(prediction)]

    print("Predicted Emotion:", emotion)

predict_emotion("test.jpg")  # <-- change to your image

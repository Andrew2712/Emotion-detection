import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from data_preprocessing import load_data

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_PATH = "saved_models/emotion_cnn.h5"

def build_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax'),  # 7 emotion classes
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model():
    train_data, test_data = load_data(TRAIN_DIR, TEST_DIR)

    model = build_cnn()
    print(model.summary())

    model.fit(train_data, epochs=25, validation_data=test_data)
    model.save(MODEL_PATH)
    print("Model saved at:", MODEL_PATH)

if __name__ == "__main__":
    train_model()

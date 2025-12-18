import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ================== PATHS ==================
MODEL_PATH = "saved_models/emotion_cnn.h5"
TEST_DIR = "dataset/test"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ================== CONFIG ==================
IMG_SIZE = 48
BATCH_SIZE = 64

CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ================== LOAD MODEL ==================
print("🔄 Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ================== DATA GENERATOR ==================
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ================== EVALUATION ==================
print("\n📊 Evaluating model on test dataset...")
loss, accuracy = model.evaluate(test_generator, verbose=1)

print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"❌ Test Loss: {loss:.4f}")

# ================== PREDICTIONS ==================
print("\n🔍 Generating predictions...")
pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_generator.classes

# ================== CLASSIFICATION REPORT ==================
print("\n📄 Classification Report:\n")

report_text = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES
)

print(report_text)

# Save classification report as TXT
with open(f"{RESULTS_DIR}/classification_report.txt", "w") as f:
    f.write(report_text)

# Save classification report as CSV
report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True
)

df_report = pd.DataFrame(report_dict).transpose()
df_report.to_csv(f"{RESULTS_DIR}/classification_report.csv")

print("✅ Classification report saved")

# ================== CONFUSION MATRIX ==================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
plt.show()

print("✅ Confusion matrix saved")

# ================== NORMALIZED CONFUSION MATRIX ==================
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix_normalized.png")
plt.show()

print("✅ Normalized confusion matrix saved")

print("\n🎉 Model evaluation completed successfully!")

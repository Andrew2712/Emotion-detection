import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="centered"
)

# ---------------- CONSTANTS ----------------
MODEL_PATH = "saved_models/emotion_cnn.h5"

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOJI_MAP = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😄',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😲'
}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image: Image.Image):
    """
    Preprocess image exactly like FER-2013
    """
    image = image.convert("L")               # Grayscale
    image = image.resize((48, 48))            # FER-2013 size
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=(0, -1))  # (1,48,48,1)
    return image

# ---------------- PREDICTION ----------------
def predict_emotion(image: Image.Image):
    processed = preprocess_image(image)

    preds = model.predict(processed, verbose=0)[0]

    # Safety: ensure probabilities sum to 1
    preds = tf.nn.softmax(preds).numpy()

    emotion_index = int(np.argmax(preds))
    confidence = float(preds[emotion_index])

    return preds, EMOTIONS[emotion_index], confidence

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main { background-color: #f5f7fa; }
.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 24px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.title("😊 Emotion Detection App")
st.write("Upload a **clear, front-facing face image** for best results.")

st.warning(
    "⚠️ Model trained on **FER-2013 (48×48 grayscale)**. "
    "Side poses, sunglasses, group photos may give incorrect results."
)

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Predict Emotion"):
        with st.spinner("Analyzing facial expression..."):
            probabilities, emotion, confidence = predict_emotion(image)

        # ---------- MAIN RESULT ----------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Prediction Result")
        st.markdown(f"## {EMOJI_MAP[emotion]} **{emotion}**")
        st.progress(float(confidence))
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

        # ---------- DISTRIBUTION ----------
        df = pd.DataFrame({
            "Emotion": EMOTIONS,
            "Probability (%)": np.round(probabilities * 100, 2)
        }).sort_values("Probability (%)", ascending=False)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Emotion Probability Distribution")
        st.bar_chart(df.set_index("Emotion"))
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("🧠 CNN Emotion Recognition | FER-2013 | TensorFlow")
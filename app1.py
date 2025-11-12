import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

# -------------------------------
# ⚙️ App Configuration
# -------------------------------
st.set_page_config(page_title="🧠 Brain Tumor MRI Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload a brain MRI scan and choose which trained model to use for prediction.")

# -------------------------------
# 📂 Model Selector
# -------------------------------
model_options = {
    "EfficientNetB0": ["best_efficientnetb0.keras", "best_efficientnetb0.h5"],
    "ResNet50": ["best_resnet50.keras", "best_resnet50.h5"],
    "Custom CNN": ["best_custom_cnn.keras", "best_custom_cnn.h5"]
}

selected_model_name = st.selectbox("🧩 Choose a Model:", list(model_options.keys()))

# -------------------------------
# 🔍 Helper: Auto-load model and labels
# -------------------------------
@st.cache_resource
def load_model(model_name):
    # Locate the correct model file
    possible_files = model_options[model_name]
    model_path = next((p for p in possible_files if os.path.exists(p) or os.path.exists(f"exports/{p}")), None)

    if model_path is None:
        st.error(f"❌ Model file for {model_name} not found. Please place it in the same folder or in 'exports/'.")
        st.stop()

    # Try in root and exports/
    if not os.path.exists(model_path) and os.path.exists(f"exports/{model_path}"):
        model_path = f"exports/{model_path}"

    # Load model
    st.write(f"✅ Loading **{model_name}** model...")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Load labels (common for all)
    labels_path = "class_names.json" if os.path.exists("class_names.json") else "exports/class_names.json"
    if not os.path.exists(labels_path):
        st.error("❌ 'class_names.json' not found. Please ensure it’s in the same directory as the model.")
        st.stop()

    with open(labels_path, "r") as f:
        class_names = json.load(f)

    return model, class_names

model, CLASS_NAMES = load_model(selected_model_name)

# -------------------------------
# 🖼️ Image Upload
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🩺 Uploaded MRI Image", use_column_width=True)

    # -------------------------------
    # 🔄 Preprocessing
    # -------------------------------
    st.write("🧮 Preprocessing image...")
    img_size = (224, 224)  # same as training size
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------------
    # 🤖 Prediction
    # -------------------------------
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = np.max(preds) * 100

    # -------------------------------
    # 📊 Display Results
    # -------------------------------
    st.subheader("✅ Prediction Result")
    st.success(f"**{pred_class}**  ({confidence:.2f}% confidence)")

    st.write("### Probability Distribution")
    probs = {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(probs)

else:
    st.info("👆 Upload an MRI image to start classification.")

# -------------------------------
# 🧩 Footer
# -------------------------------
st.markdown("""
---  
Model Options: EfficientNetB0, ResNet50, Custom CNN  
Powered by TensorFlow & Streamlit
""")

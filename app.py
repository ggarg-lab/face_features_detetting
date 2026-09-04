import streamlit as st
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image

# -----------------------------
# Load model + config (cached so it only loads once)
# -----------------------------
@st.cache_resource
def load_model_and_config():
    project_dir = Path(__file__).resolve().parent
    model = tf.keras.models.load_model(project_dir / "face_model.h5")
    config_path = project_dir / "deployment_config.json"

    if config_path.exists():
        with config_path.open("r") as f:
            config = json.load(f)
        attribute_cols = config["attribute_cols"]
        thresholds = np.array(config["thresholds"])
    else:
        output_count = model.output_shape[-1]
        attribute_cols = [f"Attribute {index + 1}" for index in range(output_count)]
        thresholds = np.full(output_count, 0.5)

    return model, attribute_cols, thresholds

model, attribute_cols, thresholds = load_model_and_config()

TARGET_SIZE = (128, 128)

# -----------------------------
# Preprocessing (must match training)
# -----------------------------
def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize(TARGET_SIZE)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Prediction
# -----------------------------
def predict_attributes(pil_img):
    img_array = preprocess_image(pil_img)
    probs = model.predict(img_array)[0]

    results = []
    for attr, prob, thresh in zip(attribute_cols, probs, thresholds):
        results.append({
            "attribute": attr,
            "present": bool(prob >= thresh),
            "confidence": float(prob)
        })
    return results

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Face Attribute Detector", layout="centered")
st.title("Face Attribute Detector")
st.write("Upload an image or take a live photo to see which attributes the model detects.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Take a live photo")
image_file = camera_file or uploaded_file

if image_file is not None:
    pil_img = Image.open(image_file)
    image_caption = "Live Photo" if camera_file is not None else "Uploaded Image"

    col1, col2 = st.columns(2)

    with col1:
        st.image(pil_img, caption=image_caption, use_container_width=True)

    with st.spinner("Running model..."):
        results = predict_attributes(pil_img)

    detected = sorted(
        [r for r in results if r["present"]],
        key=lambda x: x["confidence"],
        reverse=True
    )
    not_detected = [r for r in results if not r["present"]]

    with col2:
        st.subheader("Detected Features")
        if detected:
            for r in detected:
                st.write(f"✅ **{r['attribute']}** — {r['confidence']*100:.1f}% confidence")
        else:
            st.write("No features detected above threshold.")

    with st.expander("Show full confidence breakdown (all attributes)"):
        all_sorted = sorted(results, key=lambda x: x["confidence"], reverse=True)
        for r in all_sorted:
            bar_color = "green" if r["present"] else "gray"
            st.write(f"{r['attribute']}: {r['confidence']:.3f}")
            st.progress(min(max(r["confidence"], 0.0), 1.0))

    with st.expander("Show attributes NOT detected"):
        if not_detected:
            for r in sorted(not_detected, key=lambda x: x["confidence"], reverse=True):
                st.write(f"❌ {r['attribute']} — {r['confidence']*100:.1f}% confidence")
        else:
            st.write("All attributes were detected.")
else:
    st.info("Upload an image or take a live photo to get started.")

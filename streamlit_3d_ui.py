# # streamlit_3d_ui.py
# from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# import streamlit as st
# import streamlit.components.v1 as components
# from PIL import Image
# import tensorflow as tf
# import numpy as np
# import tempfile

# st.set_page_config(page_title="E-Waste Classifier", layout="wide")

# st.markdown("""
#     <style>
#     .stApp {
#         background-color: transparent;
#     }
#     </style>
# """, unsafe_allow_html=True)


# # Vanta.GLOBE background
# components.html("""
# <div id="vanta-bg" style="position: fixed; width: 100vw; height: 100vh; z-index: -1; top: 0; left: 0;"></div>

# <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
# <script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.globe.min.js"></script>

# <script>
#   VANTA.GLOBE({
#     el: "#vanta-bg",
#     mouseControls: true,
#     touchControls: true,
#     gyroControls: false,
#     minHeight: 200.00,
#     minWidth: 200.00,
#     scale: 1.0,
#     scaleMobile: 1.0,
#     color: 0x10b981,
#     backgroundColor: 0x0f172a,
#     size: 1.0
#   });
# </script>
# """, height=0)
# # 1. Vanta.js Full-Screen Animated Background via components.html





# # 2. App Title
# st.markdown("""
#     <h1 style='text-align: center; color: #0f766e;'>E-Waste Image Classifier 🌿</h1>
#     <p style='text-align: center; color: #475569;'>Upload an image to classify it into one of 10 e-waste categories.</p>
# """, unsafe_allow_html=True)

# # 3. Load Model and Class Names
# @st.cache_resource
# def load_model():
#     import urllib.request
#     import os

#     MODEL_PATH = "Efficient_classify_prakriti_improved.keras"
#     MODEL_URL = "https://drive.google.com/uc?export=download&id=1zwy63UPyw2PKuocnnfUa0saJopjAoy_C"

#     if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:  # <1MB means failed download
#         with st.spinner("🔄 Downloading model from Google Drive..."):
#             try:
#                 urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
#                 st.success("✅ Model downloaded successfully.")
#                 st.write(f"📦 Model file size: {os.path.getsize(MODEL_PATH) / 1_000_000:.2f} MB")
#             except Exception as e:
#                 st.error(f"❌ Failed to download the model: {e}")
#                 raise e

#     try:
#         model = tf.keras.models.load_model(MODEL_PATH)
#     except Exception as e:
#         st.error("❌ Model file exists but failed to load. It might be corrupted.")
#         raise e

#     class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB',
#                    'Player', 'Printer', 'Television', 'Washing Machine']
#     return model, class_names


# # 4. Image Uploader
# uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess and Predict
#     img_resized = image.resize((224, 224))  # Resize to model input size
#     img_array = np.expand_dims(np.array(img_resized), axis=0)  # Add batch dimension
#     img_array = preprocess_input(img_array)  # Apply EfficientNet preprocessing

#     preds = model.predict(img_array)[0]
#     pred_index = np.argmax(preds)
#     pred_class = class_names[pred_index]
#     confidence = round(preds[pred_index] * 100, 2)

#     st.success(f"🔍 Predicted Class: **{pred_class}** with **{confidence}%** confidence.")

#     st.bar_chart({"Confidence (%)": preds})

# # 5. 3D Model Viewer (using model-viewer via iframe)
# st.markdown("### 🧩 3D E-Waste Item Viewer")
# model_viewer_code = """
# <model-viewer src="<model-viewer src="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/BoomBox/glTF-Binary/BoomBox.glb"
#               alt="Electronic component" auto-rotate camera-controls background-color='#ffffff'
#               style="width: 100%; height: 500px;">
# </model-viewer>
# <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
# """
# components.html(model_viewer_code, height=500)

# # Footer
# st.markdown("""
#     <hr>
#     <p style='text-align: center; font-size: 0.8em; color: #888;'>
#     Built with ❤️ using Streamlit, TensorFlow, and Three.js
#     </p>
# """, unsafe_allow_html=True)


from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import tensorflow as tf
import numpy as np
import base64
import os
import urllib.request
import tensorflow as tf


# MODEL_PATH = "Efficient_classify_prakriti_improved.keras"
# MODEL_URL = "https://drive.google.com/uc?export=download&id=1zwy63UPyw2PKuocnnfUa0saJopjAoy_C"


# Set page layout
st.set_page_config(page_title="E-Waste Classifier", layout="wide")

# ---------- Set full app background image ----------
def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
        css = f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.12)),
                        url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# Set your background image
set_bg_from_local("data-destruction.jpg")

# ---------- Style ----------
st.markdown("""
    <style>
    h1, h2, h3 {
        text-shadow: 1px 1px 5px rgba(0,0,0,0.4);
    }
    .upload-area {
        border: 2px dashed #10b981;
        padding: 1em;
        border-radius: 12px;
        background-color: rgba(255, 255, 255, 0.1);
        transition: background 0.3s ease;
    }
    .upload-area:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    .stButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='
            color: #22d3ee;
            font-size: 3em;
            font-weight: 800;
            text-shadow: 0 0 10px rgba(0,0,0,0.5);
            background: rgba(255, 255, 255, 0.1);
            padding: 10px 20px;
            border-radius: 12px;
            display: inline-block;
        '>
            ⚡ E-Waste Image Classifier
        </h1>
        <p style='
            color: #f1f5f9;
            font-size: 1.2em;
            margin-top: 10px;
            text-shadow: 0 0 5px rgba(0,0,0,0.6);
            background: rgba(0, 0, 0, 0.25);
            padding: 8px 16px;
            border-radius: 8px;
            display: inline-block;
        '>
            Drop your e-waste image below and watch the AI work its magic! 🌱
        </p>
    </div>
""", unsafe_allow_html=True)


# ---------- Load Model ----------
@st.cache_resource
def load_model():
    MODEL_PATH = "Efficient_classify_prakriti_improved.keras"
    MODEL_URL = "https://drive.google.com/uc?export=download&id=1zwy63UPyw2PKuocnnfUa0saJopjAoy_C"
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB',
                   'Player', 'Printer', 'Television', 'Washing Machine']
    return model, class_names

# ✅ FIXED: Now actually load the model
model, class_names = load_model()


# ---------- Upload Image ----------
with st.container():
    st.markdown("""
        <div class='upload-area' style='
            border: 2px dashed #10b981;
            padding: 1em;
            border-radius: 12px;
            background-color: rgba(255, 255, 255, 0.1);
            transition: background 0.3s ease;
        '>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])
    
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # Styled uploaded image header
        st.markdown(f"""
            <div style='
                background: rgba(0, 0, 0, 0.4);
                padding: 15px;
                border-radius: 12px;
                margin-top: 20px;
                text-align: center;
                box-shadow: 0 0 20px rgba(0,0,0,0.4);
            '>
                <h3 style='color: #f1f5f9; font-size: 1.5em;'>🖼️ Uploaded Image</h3>
            </div>
        """, unsafe_allow_html=True)
        st.image(image, use_column_width=True)

        # Preprocess and predict
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized), axis=0)
        img_array = preprocess_input(img_array)
        preds = model.predict(img_array)[0]

        pred_index = np.argmax(preds)
        pred_class = class_names[pred_index]
        confidence = round(preds[pred_index] * 100, 2)

        # Styled prediction block
        st.markdown(f"""
            <div style='
                background: rgba(255, 255, 255, 0.15);
                padding: 20px;
                margin-top: 20px;
                border-radius: 12px;
                text-align: center;
                backdrop-filter: blur(4px);
                box-shadow: 0 0 15px rgba(0,0,0,0.3);
            '>
                <h2 style='color: #22d3ee;'>🔍 Predicted Class: <strong>{pred_class}</strong></h2>
                <h4 style='color: #f1f5f9;'>Confidence: <strong>{confidence}%</strong></h4>
            </div>
        """, unsafe_allow_html=True)

        st.bar_chart({"Confidence (%)": preds})

# ---------- Optional: 3D Viewer ----------
st.markdown("""
    <h3 style='
        text-align: center;
        color: #f1f5f9;
        font-size: 1.8em;
        font-weight: 600;
        text-shadow: 0 0 6px rgba(0,0,0,0.5);
        background: rgba(0, 0, 0, 0.25);
        padding: 10px 20px;
        border-radius: 10px;
        display: inline-block;
    '>
        🧩 3D E-Waste Item Viewer
    </h3>
""", unsafe_allow_html=True)


model_viewer_code = """
<div style="position: relative; width: 100%; height: 500px; border-radius: 15px; overflow: hidden;
            background-color: rgba(255,255,255,0.05); box-shadow: 0 0 30px rgba(0,0,0,0.4);">
    <model-viewer src="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/BoomBox/glTF-Binary/BoomBox.glb"
                  alt="Electronic component" auto-rotate camera-controls
                  style="width: 100%; height: 100%; background-color: transparent;">
    </model-viewer>
</div>
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
"""
components.html(model_viewer_code, height=500)

# ---------- Footer ----------
st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 0.85em; color: #cbd5e1;'>
        🚀 Built with ❤️ using <strong>Streamlit</strong>, <strong>TensorFlow</strong>, and <strong>Three.js</strong><br>
        <em>© 2025 Prakriti Roy</em>
    </p>
""", unsafe_allow_html=True)

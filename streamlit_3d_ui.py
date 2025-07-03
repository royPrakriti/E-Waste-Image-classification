# from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# import streamlit as st
# import streamlit.components.v1 as components
# from PIL import Image
# import tensorflow as tf
# import numpy as np
# import base64
# import os
# import gdown

# # ------------------ PAGE SETUP ------------------
# st.set_page_config(page_title="E-Waste Classifier", layout="wide")

# # Set background image from local file
# def set_bg_from_local(image_file):
#     with open(image_file, "rb") as f:
#         data = f.read()
#         encoded = base64.b64encode(data).decode()
#         css = f"""
#         <style>
#         .stApp {{
#             background: linear-gradient(rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.12)),
#                         url("data:image/jpg;base64,{encoded}");
#             background-size: cover;
#             background-position: center;
#             background-attachment: fixed;
#         }}
#         </style>
#         """
#         st.markdown(css, unsafe_allow_html=True)

# # Background image
# set_bg_from_local("data-destruction.jpg")

# # ------------------ STYLES ------------------
# st.markdown("""
#     <style>
#     h1, h2, h3 {
#         text-shadow: 1px 1px 5px rgba(0,0,0,0.4);
#     }
#     .upload-area {
#         border: 2px dashed #10b981;
#         padding: 1em;
#         border-radius: 12px;
#         background-color: rgba(255, 255, 255, 0.1);
#     }
#     .stButton>button {
#         background-color: #10b981;
#         color: white;
#         border-radius: 10px;
#         padding: 10px 20px;
#         font-weight: bold;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ------------------ TITLE ------------------
# st.markdown("""
#     <div style='text-align: center; padding: 1rem 0;'>
#         <h1 style='
#             color: #22d3ee;
#             font-size: 3em;
#             font-weight: 800;
#             background: rgba(255, 255, 255, 0.1);
#             padding: 10px 20px;
#             border-radius: 12px;
#             display: inline-block;
#         '>
#             ⚡ E-Waste Image Classifier
#         </h1>
#         <p style='
#             color: #f1f5f9;
#             font-size: 1.2em;
#             margin-top: 10px;
#             background: rgba(0, 0, 0, 0.25);
#             padding: 8px 16px;
#             border-radius: 8px;
#             display: inline-block;
#         '>
#             Drop your e-waste image below and watch the AI work its magic! 🌱
#         </p>
#     </div>
# """, unsafe_allow_html=True)

# # ------------------ MODEL LOADER ------------------
# @st.cache_resource
# def load_model():
#     MODEL_PATH = "Efficient_classify_prakriti_improved.keras"
#     FILE_ID = "1zwy63UPyw2PKuocnnfUa0saJopjAoy_C"
#     MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

#     if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
#         with st.spinner("📥 Downloading model from Google Drive..."):
#             try:
#                 gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
#                 st.success(f"✅ Model downloaded. Size: {os.path.getsize(MODEL_PATH)/1_000_000:.2f} MB")
#             except Exception as e:
#                 st.error("❌ Failed to download model.")
#                 raise e

#     try:
#         model = tf.keras.models.load_model(MODEL_PATH)
#     except Exception as e:
#         st.error("❌ Model exists but failed to load.")
#         raise e

#     class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB',
#                    'Player', 'Printer', 'Television', 'Washing Machine']
#     return model, class_names

# model, class_names = load_model()

# # ------------------ UPLOAD IMAGE ------------------
# with st.container():
#     st.markdown("<div class='upload-area'>", unsafe_allow_html=True)
#     uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])
#     st.markdown("</div>", unsafe_allow_html=True)

#     if uploaded_file:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.markdown("<h3 style='color:#f1f5f9;'>🖼️ Uploaded Image</h3>", unsafe_allow_html=True)
#         st.image(image, use_column_width=True)

#         # Preprocess and predict
#         img_resized = image.resize((224, 224))
#         img_array = np.expand_dims(np.array(img_resized), axis=0)
#         img_array = preprocess_input(img_array)

#         preds = model.predict(img_array)[0]
#         pred_index = np.argmax(preds)
#         pred_class = class_names[pred_index]
#         confidence = round(preds[pred_index] * 100, 2)

#         # Prediction Output
#         st.markdown(f"""
#             <div style='
#                 background: rgba(255, 255, 255, 0.15);
#                 padding: 20px;
#                 margin-top: 20px;
#                 border-radius: 12px;
#                 text-align: center;
#                 backdrop-filter: blur(4px);
#                 box-shadow: 0 0 15px rgba(0,0,0,0.3);
#             '>
#                 <h2 style='color: #22d3ee;'>🔍 Predicted Class: <strong>{pred_class}</strong></h2>
#                 <h4 style='color: #f1f5f9;'>Confidence: <strong>{confidence}%</strong></h4>
#             </div>
#         """, unsafe_allow_html=True)

#         st.bar_chart({"Confidence (%)": preds})

# # ------------------ 3D VIEWER ------------------
# st.markdown("""
#     <h3 style='text-align: center; color: #f1f5f9; font-size: 1.8em;'>
#         🧩 3D E-Waste Item Viewer
#     </h3>
# """, unsafe_allow_html=True)

# model_viewer_code = """
# <div style="position: relative; width: 100%; height: 500px; border-radius: 15px; overflow: hidden;
#             background-color: rgba(255,255,255,0.05); box-shadow: 0 0 30px rgba(0,0,0,0.4);">
#     <model-viewer src="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/BoomBox/glTF-Binary/BoomBox.glb"
#                   alt="Electronic component" auto-rotate camera-controls
#                   style="width: 100%; height: 100%; background-color: transparent;">
#     </model-viewer>
# </div>
# <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
# """
# components.html(model_viewer_code, height=500)

# # ------------------ FOOTER ------------------
# st.markdown("""
#     <hr>
#     <p style='text-align: center; font-size: 0.85em; color: #cbd5e1;'>
#         🚀 Built with ❤️ using <strong>Streamlit</strong>, <strong>TensorFlow</strong>, and <strong>Three.js</strong><br>
#         <em>© 2025 Prakriti Roy</em>
#     </p>
# """, unsafe_allow_html=True)

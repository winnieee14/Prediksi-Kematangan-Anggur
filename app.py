import streamlit as st
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

# Setup Config Page
st.set_page_config(
    page_title="Grape Ripeness Model",
    layout="centered"
)

# Load All Models
@st.cache_resource
def load_models():
    print("Loading models...")
    try:
        # Load YOLO
        yolo = YOLO('best_k.pt') 
        # Load XGBoost
        xgb = joblib.load('xgboost_rgb_model.pkl') 
        return yolo, xgb
    except Exception as e:
        return None, str(e)

yolo_model, xgb_model = load_models()

if isinstance(xgb_model, str):
    st.error(f" Gagal memuat model: {xgb_model}")
    st.stop()

# Process Image And Prediction
def process_prediction(uploaded_file):

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    img = cv2.imdecode(file_bytes, 1) 
    
    if img is None: return None, None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    results = yolo_model.predict(img, verbose=False, retina_masks=True, conf=0.50)
    
    if not results or not results[0].masks:
        return None, None, None

    # Masking
    raw_mask = np.any(results[0].masks.data.cpu().numpy(), axis=0).astype(np.uint8)
    mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Segmentasi
    segmented_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    # Ekstraction Colour and Texture
    mean, std = cv2.meanStdDev(img_rgb, mask=mask)
    mean_r, mean_g, mean_b = mean.flatten()
    std_r, std_g, std_b    = std.flatten()
    
    pixel_count = mask.sum()
    relative_size = pixel_count / (h * w)

    features = np.array([[mean_r, mean_g, mean_b, std_r, std_g, std_b, relative_size]])
    
    # Prediction
    days_left = xgb_model.predict(features)[0]
    
    return days_left, segmented_img, (mask, img_rgb, features[0])

# User Interface

st.title("Deteksi Kematangan Anggur")
st.write("Upload foto anggur 'Cherny Cristal' untuk memprediksi sisa hari panen.")

uploaded_file = st.file_uploader("Pilih Gambar...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    
    # Display Original Image
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption='Gambar Asli', width=350)

    uploaded_file.seek(0)

    if st.button('Prediksi Sekarang'):
        with st.spinner('Sedang menganalisis warna & tekstur...'):
            days, seg_img, extra_data = process_prediction(uploaded_file)
        
        if days is not None:
            mask, img_rgb, feat_vals = extra_data
            
            st.success(f"### Estimasi Waktu Matang: {days:.1f} Hari Lagi")
            st.info(f"Rata-rata RGB: R={int(feat_vals[0])}, G={int(feat_vals[1])}, B={int(feat_vals[2])}")
            
            with col2:
                st.image(seg_img, caption='Area yang Dianalisis', width=350)
            
            st.markdown("---")
            st.subheader(" Analisis Histogram Warna")
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            colors = ('red', 'green', 'blue')
            titles = ['Distribusi Merah', 'Distribusi Hijau', 'Distribusi Biru']
            
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img_rgb], [i], mask, [256], [0, 256])
                ax[i].plot(hist, color=color, linewidth=2)
                ax[i].fill_between(range(256), hist.flatten(), color=color, alpha=0.3)
                ax[i].set_title(f"{titles[i]}")
                ax[i].set_xlim([0, 256])
                ax[i].grid(True, alpha=0.3, linestyle='--')
            
            st.pyplot(fig)
            
        else:
            st.error(" Anggur tidak terdeteksi! Pastikan objek terlihat jelas.")
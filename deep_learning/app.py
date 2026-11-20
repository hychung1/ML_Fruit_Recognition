import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Classical scripts import their siblings by bare module name.
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'classical_ml'))

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nhận diện Rau Củ Quả",
    page_icon="🥦",
    layout="centered"
)

# Danh sách 36 nhãn rau củ quả (Theo đúng thứ tự alphabet trong thư mục train)
CLASS_NAMES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# ---------------------------------------------------------
# 2. NẠP MÔ HÌNH DÃ LƯU (SỬ DỤNG CACHE ĐỂ TỐI ƯU TỐC ĐỘ)
# ---------------------------------------------------------
@st.cache_resource
def load_efficientnet_model():
    model_path = os.path.join(PROJECT_ROOT, 'model_dl', 'efficientnet_b0_fruit_veg.h5')
    model = load_model(model_path)
    return model

# ---------------------------------------------------------
# 3. GIAO DIỆN NGƯỜI DÙNG (UI)
# ---------------------------------------------------------
st.title("🥦 Phân Loại Rau Củ Quả")
st.write("Hỗ trợ: **EfficientNetB0**, **KNN**, **SVM**, **Random Forest**")
st.markdown("---")

MODEL_CHOICES = ("EfficientNetB0", "KNN", "SVM", "Random Forest")
model_choice = st.selectbox("Chọn mô hình dự đoán", MODEL_CHOICES)

# Bộ chọn file ảnh từ máy tính
uploaded_file = st.file_uploader(
    "Chọn một bức ảnh rau củ quả từ máy tính của bạn...",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # Hiển thị ảnh đã chọn
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)

    # Nút bấm dự đoán
    if st.button("🔍 Dự đoán loại Rau Củ Quả", type="primary"):
        with st.spinner("Đang phân tích hình ảnh..."):
            if model_choice == "EfficientNetB0":
                with st.spinner("Đang tải mô hình AI..."):
                    model = load_efficientnet_model()
                img_batch = np.expand_dims(np.array(image.resize((224, 224))), axis=0)
                predictions = model.predict(preprocess_input(img_batch))[0]
                top_3 = [(CLASS_NAMES[i], float(predictions[i]) * 100)
                         for i in np.argsort(predictions)[-3:][::-1]]
            else:
                if model_choice == "KNN":
                    from train_knn import predict_image_knn as predict
                elif model_choice == "SVM":
                    from train_svm import predict_image_svm as predict
                else:
                    from train_rf import predict_image_rf as predict
                _, _, raw_top_3 = predict(image)
                top_3 = [(item["class"], item["probability"]) for item in raw_top_3]

            predicted_name, confidence = top_3[0]

            # HIỂN THỊ KẾT QUẢ
            st.success(f"**Mô hình:** {model_choice} | **Kết quả:** {predicted_name.upper()}")
            st.info(f"**Độ tin cậy (Confidence):** {confidence:.2f}%")

            st.write("---")
            st.write("📊 **Top 3 khả năng cao nhất:**")
            for class_name, prob in top_3:
                st.write(f"- **{class_name.capitalize()}**: {prob:.2f}%")
                st.progress(int(prob))

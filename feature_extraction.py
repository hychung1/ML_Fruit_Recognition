"""
BẢNG PHÂN CÔNG ĐỒ ÁN HỌC MÁY - ĐỀ TÀI 07
Module: feature_extraction.py
Thực hiện: Người 1 & Thống nhất toàn nhóm

Nhiệm vụ:
1. Resize và tiền xử lý ảnh đầu vào.
2. Trích xuất đặc trưng kết hợp:
   - Màu sắc (Color Histogram trong không gian HSV): 8x8x8 = 512 chiều.
   - Hình dạng và kết cấu (HOG - Histogram of Oriented Gradients): 288 chiều.
   Tổng vector đặc trưng: 800 chiều.
3. Trích xuất đặc trưng đồng bộ cho cả 3 tập: Train, Validation, Test.
4. Chuẩn hóa đặc trưng bằng StandardScaler (CHỈ FIT TRÊN TẬP TRAIN - Tránh data leakage).
5. Lưu cache đặc trưng và scaler vào đĩa để tái sử dụng nhanh chóng.
"""

import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. HÀM TRÍCH XUẤT ĐẶC TRƯNG TỪ MỘT HÌNH ẢNH
# -------------------------------------------------------------
def extract_features_from_image(img_input):
    """
    Trích xuất vector đặc trưng 800 chiều từ một hình ảnh.
    img_input: có thể là đường dẫn file (str), ảnh numpy array (BGR hoặc RGB), hoặc PIL Image.
    """
    if isinstance(img_input, str):
        # Dùng PIL để tương thích hoàn toàn với đường dẫn tiếng Việt Unicode trên Windows
        pil_img = Image.open(img_input).convert('RGB')
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(img_input, Image.Image):
        img_rgb = np.array(img_input.convert('RGB'))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(img_input, np.ndarray):
        img_bgr = img_input
    else:
        raise ValueError("Định dạng ảnh đầu vào không hợp lệ!")

    # 1.1. Trích xuất đặc trưng Màu sắc (Color Histogram trong không gian HSV)
    # HSV phân tách rõ ràng Hue (màu sắc) và Value (độ sáng), rất tốt cho nhận diện trái cây
    img_resized = cv2.resize(img_bgr, (128, 128))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten() # 512 chiều

    # 1.2. Trích xuất đặc trưng Hình dạng/Biên cạnh (HOG)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_64 = cv2.resize(gray, (64, 64))
    hog_feat = hog(
        gray_64,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    ) # 288 chiều

    # 1.3. Kết hợp vector đặc trưng (800 chiều)
    combined_features = np.hstack([hist, hog_feat]).astype(np.float32)
    return combined_features

# -------------------------------------------------------------
# 2. HÀM QUÉT VÀ TRÍCH XUẤT CHO CẢ TẬP DỮ LIỆU
# -------------------------------------------------------------
def load_dataset_features(split_name, force_reload=False):
    """
    Trích xuất hoặc nạp đặc trưng của một tập dữ liệu (train/validation/test).
    Có lưu cache dạng .joblib để chạy siêu nhanh.
    """
    cache_file = os.path.join(FEATURES_DIR, f"{split_name}_features.joblib")
    if os.path.exists(cache_file) and not force_reload:
        print(f"[CACHE] Đang nạp đặc trưng tập '{split_name}' từ cache...")
        data = joblib.load(cache_file)
        return data["X"], data["y"], data["class_names"]

    print(f"\n[XỬ LÝ] Bắt đầu trích xuất đặc trưng cho tập '{split_name}'...")
    split_dir = os.path.join(DATASET_DIR, split_name)
    class_names = sorted(os.listdir(os.path.join(DATASET_DIR, "train")))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    X_list = []
    y_list = []

    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        
        file_list = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fname in file_list:
            fpath = os.path.join(cls_dir, fname)
            try:
                feat = extract_features_from_image(fpath)
                X_list.append(feat)
                y_list.append(class_to_idx[cls])
            except Exception as e:
                print(f"Lỗi khi đọc file {fpath}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Lưu cache
    joblib.dump({"X": X, "y": y, "class_names": class_names}, cache_file)
    print(f"[XONG] Đã trích xuất {len(X)} mẫu tập '{split_name}'. Kích thước ma trận đặc trưng: {X.shape}")
    return X, y, class_names

# -------------------------------------------------------------
# 3. CHUẨN HÓA ĐẶC TRƯNG CHUẨN (FITTING CHỈ TRÊN TRAIN)
# -------------------------------------------------------------
def get_prepared_data(force_reload=False):
    """
    Trả về bộ dữ liệu đã được trích xuất và chuẩn hóa đặc trưng:
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, scaler
    """
    X_train_raw, y_train, class_names = load_dataset_features("train", force_reload)
    X_val_raw, y_val, _ = load_dataset_features("validation", force_reload)
    X_test_raw, y_test, _ = load_dataset_features("test", force_reload)

    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    
    print("\n[CHUẨN HÓA] Khởi tạo và Fit StandardScaler CHỈ TRÊN TẬP TRAIN...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    
    # Transform tập Validation và Test bằng scaler đã fit trên Train
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Lưu scaler để phục vụ dự đoán ảnh mới trong Streamlit app
    joblib.dump(scaler, scaler_path)
    print(f"[OK] Đã lưu StandardScaler tại: {scaler_path}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, scaler

if __name__ == "__main__":
    get_prepared_data(force_reload=False)

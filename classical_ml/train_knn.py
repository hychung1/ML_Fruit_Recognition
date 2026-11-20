"""
BẢNG PHÂN CÔNG ĐỒ ÁN HỌC MÁY - ĐỀ TÀI 07
Module: train_knn.py
Thực hiện: Người 1 – KNN (K-Nearest Neighbors)

Nhiệm vụ chi tiết:
1. Nạp dữ liệu đặc trưng đã chuẩn hóa (X_train, y_train, X_val, y_val, X_test, y_test).
2. Xây dựng mô hình KNN và tối ưu siêu tham số:
   - Thử nghiệm các giá trị k: [1, 3, 5, 7, 9, 11, 15, 21]
   - Thử nghiệm các khoảng cách: ['euclidean', 'manhattan']
   - Đánh giá độ chính xác trên tập Validation để chọn cặp (k, metric) tối ưu nhất.
   - Vẽ biểu đồ đường thể hiện sự biến thiên Validation Accuracy theo k.
3. Huấn luyện mô hình tối ưu và đánh giá toàn diện:
   - Kiểm tra trên tập Test (Accuracy, Macro Precision, Recall, F1-score).
   - Xuất Classification Report chi tiết cho 36 lớp rau củ quả.
   - Vẽ và lưu Confusion Matrix vào results/confusion_matrix_knn.png.
4. Lưu mô hình KNN đã tối ưu vào models/knn_model.joblib.
5. Cung cấp hàm dự đoán ảnh mới để tích hợp vào ứng dụng Streamlit.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from feature_extraction import get_prepared_data, extract_features_from_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_evaluate_knn():
    print("=" * 65)
    print("      NGƯỜI 1: HUẤN LUYỆN VÀ TỐI ƯU MÔ HÌNH KNN")
    print("=" * 65)
    
    # 1. Nạp dữ liệu đã trích xuất & chuẩn hóa
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, scaler = get_prepared_data()
    
    print(f"\nSố mẫu tập Train     : {X_train.shape[0]} (Số chiều đặc trưng: {X_train.shape[1]})")
    print(f"Số mẫu tập Validation: {X_val.shape[0]}")
    print(f"Số mẫu tập Test      : {X_test.shape[0]}")
    
    # 2. Tối ưu siêu tham số trên tập Validation
    print("\n--- BƯỚC 1: TỐI ƯU SIÊU THAM SỐ (k và distance metric) TRÊN TẬP VALIDATION ---")
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    metrics = ['euclidean', 'manhattan']
    
    results_tuning = {m: [] for m in metrics}
    best_val_acc = -1.0
    best_params = {}
    
    for metric in metrics:
        print(f"\n>> Đang thử nghiệm khoảng cách: {metric.upper()}")
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
            knn.fit(X_train, y_train)
            val_acc = knn.score(X_val, y_val)
            results_tuning[metric].append(val_acc)
            print(f"   k = {k:2d} | Val Accuracy: {val_acc * 100:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {'n_neighbors': k, 'metric': metric}
                
    print(f"\n=> CẶP SIÊU THAM SỐ TỐI ƯU NHẤT: k = {best_params['n_neighbors']}, metric = '{best_params['metric']}' (Val Acc: {best_val_acc * 100:.2f}%)")
    
    # 3. Vẽ biểu đồ quá trình tối ưu tham số k
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        plt.plot(k_values, results_tuning[metric], marker='o', linewidth=2, label=f'Metric: {metric.capitalize()}')
    plt.axvline(x=best_params['n_neighbors'], color='red', linestyle='--', label=f"Best k = {best_params['n_neighbors']}")
    plt.title("Tối ưu siêu tham số k trên tập Validation (KNN)", fontsize=14, fontweight="bold")
    plt.xlabel("Số láng giềng k", fontsize=12)
    plt.ylabel("Độ chính xác Validation", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    tuning_plot_path = os.path.join(RESULTS_DIR, "knn_tuning_k.png")
    plt.savefig(tuning_plot_path, dpi=300)
    plt.close()
    print(f"[OK] Đã lưu biểu đồ tối ưu tham số k tại: {tuning_plot_path}")
    
    # 4. Huấn luyện mô hình tối ưu
    print("\n--- BƯỚC 2: HUẤN LUYỆN MÔ HÌNH KNN TỐI ƯU ---")
    best_knn = KNeighborsClassifier(
        n_neighbors=best_params['n_neighbors'],
        metric=best_params['metric'],
        n_jobs=-1
    )
    t0 = time.time()
    best_knn.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Thời gian huấn luyện KNN: {train_time:.2f} giây")
    
    # 5. Đánh giá trên tập Validation và Test
    print("\n--- BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ---")
    t0 = time.time()
    y_pred_test = best_knn.predict(X_test)
    test_time = time.time() - t0
    
    acc = accuracy_score(y_test, y_pred_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='macro', zero_division=0)
    
    print("=" * 55)
    print("      KẾT QUẢ ĐÁNH GIÁ KNN TRÊN TẬP TEST")
    print("=" * 55)
    print(f"📌 Accuracy       : {acc * 100:.2f}%")
    print(f"📌 Macro Precision: {precision * 100:.2f}%")
    print(f"📌 Macro Recall   : {recall * 100:.2f}%")
    print(f"📌 Macro F1-Score : {f1 * 100:.2f}%")
    print(f"⏱️  Thời gian dự đoán Test ({len(X_test)} mẫu): {test_time:.4f}s ({test_time / len(X_test) * 1000:.2f} ms/ảnh)")
    print("=" * 55)
    
    # 6. Classification Report chi tiết
    print("\n[Báo cáo chi tiết từng lớp - Classification Report]:")
    print(classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0))
    
    # 7. Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - KNN (k={best_params['n_neighbors']}, {best_params['metric']}) - Test Acc: {acc*100:.2f}%", fontsize=15, fontweight="bold")
    plt.xlabel("Nhãn Dự đoán (Predicted Label)", fontsize=12)
    plt.ylabel("Nhãn Thực tế (True Label)", fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_knn.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Đã lưu Confusion Matrix KNN tại: {cm_path}")
    
    # 8. Lưu mô hình đã huấn luyện
    model_save_path = os.path.join(MODELS_DIR, "knn_model.joblib")
    joblib.dump({
        "model": best_knn,
        "best_params": best_params,
        "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "test_time": test_time},
        "class_names": class_names
    }, model_save_path)
    print(f"\n[OK] Đã lưu mô hình KNN tại: {model_save_path}")
    
    return best_knn, best_params

# -------------------------------------------------------------
# 9. HÀM DỰ ĐOÁN ẢNH MỚI (TÍCH HỢP CHO STREAMLIT)
# -------------------------------------------------------------
def predict_image_knn(img_input):
    """
    Dự đoán nhãn của một hình ảnh mới bằng mô hình KNN.
    Trả về: predicted_class, confidence (tỉ lệ phiếu bầu láng giềng), top_3_predictions
    """
    model_path = os.path.join(MODELS_DIR, "knn_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    
    saved_data = joblib.load(model_path)
    knn_model = saved_data["model"]
    class_names = saved_data["class_names"]
    scaler = joblib.load(scaler_path)
    
    # Trích xuất đặc trưng và chuẩn hóa
    raw_feat = extract_features_from_image(img_input).reshape(1, -1)
    scaled_feat = scaler.transform(raw_feat)
    
    # Dự đoán xác suất (theo tỉ lệ láng giềng)
    probs = knn_model.predict_proba(scaled_feat)[0]
    top_3_idx = np.argsort(probs)[-3:][::-1]
    
    predicted_idx = top_3_idx[0]
    predicted_class = class_names[predicted_idx]
    confidence = probs[predicted_idx] * 100
    
    top_3_results = [
        {"class": class_names[idx], "probability": probs[idx] * 100}
        for idx in top_3_idx
    ]
    
    return predicted_class, confidence, top_3_results

if __name__ == "__main__":
    train_and_evaluate_knn()


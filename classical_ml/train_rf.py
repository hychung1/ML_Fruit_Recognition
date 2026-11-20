"""
BẢNG PHÂN CÔNG ĐỒ ÁN HỌC MÁY - ĐỀ TÀI 07
Module: train_rf.py
Thực hiện: Người 3 – Random Forest

Nhiệm vụ chi tiết:
1. Sử dụng đồng bộ tập dữ liệu đã chuẩn hóa (StandardScaler fit trên tập Train).
2. Xây dựng mô hình Random Forest Classifier và tối ưu hóa siêu tham số:
   - Số lượng cây (n_estimators): [50, 100, 150, 200]
   - Độ sâu tối đa (max_depth): [10, 15, 20, None]
   - Đánh giá trên tập Validation để chọn cặp (n_estimators, max_depth) tối ưu nhất.
3. Phân tích Feature Importance (Độ quan trọng của đặc trưng):
   - Đánh giá vai trò của đặc trưng Màu sắc (HSV Histogram) so với đặc trưng Hình thái (HOG).
   - Vẽ biểu đồ phân bố Feature Importance.
4. Đánh giá toàn diện trên tập Test:
   - Accuracy, Macro Precision, Recall, F1-score.
   - Classification Report chi tiết cho 36 lớp.
   - Vẽ và xuất hình Confusion Matrix vào results/confusion_matrix_rf.png.
5. Lưu mô hình Random Forest tối ưu vào models/rf_model.joblib.
6. Cung cấp hàm dự đoán ảnh mới cho ứng dụng Streamlit.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from feature_extraction import get_prepared_data, extract_features_from_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_evaluate_rf():
    print("=" * 65)
    print("      NGƯỜI 3: HUẤN LUYỆN VÀ TỐI ƯU MÔ HÌNH RANDOM FOREST")
    print("=" * 65)
    
    # 1. Nạp dữ liệu đã trích xuất & chuẩn hóa
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, scaler = get_prepared_data()
    
    # 2. Tối ưu siêu tham số trên tập Validation
    print("\n--- BƯỚC 1: TỐI ƯU SIÊU THAM SỐ (n_estimators, max_depth) TRÊN TẬP VALIDATION ---")
    n_estimators_list = [50, 100, 150, 200]
    max_depth_list = [10, 20, 30, None]
    
    best_val_acc = -1.0
    best_params = {}
    tuning_matrix = np.zeros((len(n_estimators_list), len(max_depth_list)))
    
    for i, n_est in enumerate(n_estimators_list):
        for j, depth in enumerate(max_depth_list):
            depth_str = "None" if depth is None else str(depth)
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            val_acc = rf.score(X_val, y_val)
            tuning_matrix[i, j] = val_acc
            print(f"   n_estimators = {n_est:3d} | max_depth = {depth_str:>4} | Val Acc: {val_acc * 100:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {'n_estimators': n_est, 'max_depth': depth}
                
    depth_best_str = "None" if best_params['max_depth'] is None else str(best_params['max_depth'])
    print(f"\n=> BỘ SIÊU THAM SỐ TỐI ƯU: n_estimators = {best_params['n_estimators']}, max_depth = {depth_best_str} (Val Acc: {best_val_acc * 100:.2f}%)")
    
    # 3. Vẽ Heatmap tối ưu siêu tham số
    plt.figure(figsize=(8, 6))
    depth_labels = [str(d) for d in max_depth_list]
    sns.heatmap(tuning_matrix * 100, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=depth_labels, yticklabels=n_estimators_list)
    plt.title("Độ chính xác Validation (%) theo n_estimators và max_depth", fontsize=13, fontweight="bold")
    plt.xlabel("Độ sâu tối đa (max_depth)", fontsize=11)
    plt.ylabel("Số lượng cây (n_estimators)", fontsize=11)
    plt.tight_layout()
    heatmap_path = os.path.join(RESULTS_DIR, "rf_tuning_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"[OK] Đã lưu biểu đồ Heatmap tối ưu Random Forest tại: {heatmap_path}")
    
    # 4. Huấn luyện mô hình tối ưu
    print("\n--- BƯỚC 2: HUẤN LUYỆN MÔ HÌNH RANDOM FOREST TỐI ƯU ---")
    best_rf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        random_state=42,
        n_jobs=-1
    )
    t0 = time.time()
    best_rf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Thời gian huấn luyện Random Forest: {train_time:.2f} giây")
    
    # 5. Phân tích Feature Importance
    importances = best_rf.feature_importances_
    # 512 đặc trưng đầu là HSV Hist, 288 đặc trưng sau là HOG
    color_importance_sum = np.sum(importances[:512])
    hog_importance_sum = np.sum(importances[512:])
    print("\n[ĐỘ ĐÓNG GÓP CỦA TỪNG BỘ ĐẶC TRƯNG]:")
    print(f"- Nhóm đặc trưng Màu sắc (HSV Histogram): {color_importance_sum * 100:.2f}%")
    print(f"- Nhóm đặc trưng Hình dạng/Biên (HOG)    : {hog_importance_sum * 100:.2f}%")
    
    plt.figure(figsize=(6, 5))
    plt.bar(["Màu sắc (HSV)", "Hình dạng (HOG)"], [color_importance_sum * 100, hog_importance_sum * 100],
            color=['orange', 'teal'], edgecolor='black')
    plt.title("Tỉ lệ đóng góp đặc trưng trong Random Forest", fontsize=12, fontweight="bold")
    plt.ylabel("Tổng Importance (%)", fontsize=11)
    for idx, val in enumerate([color_importance_sum, hog_importance_sum]):
        plt.text(idx, val * 100 + 1, f"{val * 100:.1f}%", ha='center', fontweight='bold')
    plt.ylim(0, 100)
    plt.tight_layout()
    feat_imp_path = os.path.join(RESULTS_DIR, "rf_feature_importance.png")
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"[OK] Đã lưu biểu đồ Feature Importance tại: {feat_imp_path}")
    
    # 6. Đánh giá trên tập Test
    print("\n--- BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ---")
    t0 = time.time()
    y_pred_test = best_rf.predict(X_test)
    test_time = time.time() - t0
    
    acc = accuracy_score(y_test, y_pred_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='macro', zero_division=0)
    
    print("=" * 55)
    print("      KẾT QUẢ ĐÁNH GIÁ RANDOM FOREST TRÊN TẬP TEST")
    print("=" * 55)
    print(f"📌 Accuracy       : {acc * 100:.2f}%")
    print(f"📌 Macro Precision: {precision * 100:.2f}%")
    print(f"📌 Macro Recall   : {recall * 100:.2f}%")
    print(f"📌 Macro F1-Score : {f1 * 100:.2f}%")
    print(f"⏱️  Thời gian dự đoán Test ({len(X_test)} mẫu): {test_time:.4f}s ({test_time / len(X_test) * 1000:.2f} ms/ảnh)")
    print("=" * 55)
    
    # 7. Classification Report chi tiết
    print("\n[Báo cáo chi tiết từng lớp - Classification Report]:")
    print(classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0))
    
    # 8. Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - Random Forest ({best_params}) - Test Acc: {acc*100:.2f}%", fontsize=15, fontweight="bold")
    plt.xlabel("Nhãn Dự đoán (Predicted Label)", fontsize=12)
    plt.ylabel("Nhãn Thực tế (True Label)", fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_rf.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Đã lưu Confusion Matrix Random Forest tại: {cm_path}")
    
    # 9. Lưu mô hình đã huấn luyện
    model_save_path = os.path.join(MODELS_DIR, "rf_model.joblib")
    joblib.dump({
        "model": best_rf,
        "best_params": best_params,
        "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "test_time": test_time},
        "class_names": class_names
    }, model_save_path)
    print(f"\n[OK] Đã lưu mô hình Random Forest tại: {model_save_path}")
    
    return best_rf, best_params

# -------------------------------------------------------------
# 10. HÀM DỰ ĐOÁN ẢNH MỚI (TÍCH HỢP CHO STREAMLIT)
# -------------------------------------------------------------
def predict_image_rf(img_input):
    """
    Dự đoán nhãn của một hình ảnh mới bằng mô hình Random Forest.
    Trả về: predicted_class, confidence, top_3_predictions
    """
    model_path = os.path.join(MODELS_DIR, "rf_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    
    saved_data = joblib.load(model_path)
    rf_model = saved_data["model"]
    class_names = saved_data["class_names"]
    scaler = joblib.load(scaler_path)
    
    raw_feat = extract_features_from_image(img_input).reshape(1, -1)
    scaled_feat = scaler.transform(raw_feat)
    
    probs = rf_model.predict_proba(scaled_feat)[0]
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
    train_and_evaluate_rf()


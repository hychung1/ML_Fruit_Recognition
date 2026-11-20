"""
BẢNG PHÂN CÔNG ĐỒ ÁN HỌC MÁY - ĐỀ TÀI 07
Module: train_svm.py
Thực hiện: Người 2 – SVM (Support Vector Machine)

Nhiệm vụ chi tiết:
1. Sử dụng đồng bộ tập dữ liệu đã chuẩn hóa (StandardScaler fit trên tập Train).
2. Tối ưu hóa các siêu tham số của SVM:
   - Kernel: ['rbf', 'linear']
   - Hệ số điều chuẩn C: [0.1, 1.0, 10.0]
   - Tham số gamma (cho kernel rbf): ['scale', 0.01]
   - Đánh giá trên tập Validation để tìm ra bộ siêu tham số (C, kernel, gamma) đạt độ chính xác cao nhất.
3. Huấn luyện mô hình tối ưu với tính năng tính xác suất (probability=True) phục vụ dự đoán.
4. Đánh giá toàn diện trên tập Test:
   - Accuracy, Macro Precision, Recall, F1-score.
   - Classification Report chi tiết cho 36 lớp.
   - Vẽ và xuất hình Confusion Matrix vào results/confusion_matrix_svm.png.
5. Lưu mô hình SVM tối ưu vào models/svm_model.joblib.
6. Cung cấp hàm dự đoán ảnh mới cho ứng dụng Streamlit.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from feature_extraction import get_prepared_data, extract_features_from_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_evaluate_svm():
    print("=" * 65)
    print("      NGƯỜI 2: HUẤN LUYỆN VÀ TỐI ƯU MÔ HÌNH SVM")
    print("=" * 65)
    
    # 1. Nạp dữ liệu đã trích xuất & chuẩn hóa
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, scaler = get_prepared_data()
    
    # 2. Tối ưu siêu tham số trên tập Validation
    print("\n--- BƯỚC 1: TỐI ƯU SIÊU THAM SỐ (C, kernel, gamma) TRÊN TẬP VALIDATION ---")
    param_grid = [
        {'kernel': 'linear', 'C': 0.1},
        {'kernel': 'linear', 'C': 1.0},
        {'kernel': 'linear', 'C': 10.0},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 5.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 0.005},
    ]
    
    best_val_acc = -1.0
    best_params = {}
    tuning_logs = []
    
    for p in param_grid:
        if p['kernel'] == 'linear':
            clf = SVC(kernel=p['kernel'], C=p['C'], random_state=42)
            desc = f"Kernel=linear, C={p['C']}"
        else:
            clf = SVC(kernel=p['kernel'], C=p['C'], gamma=p['gamma'], random_state=42)
            desc = f"Kernel=rbf, C={p['C']}, gamma={p['gamma']}"
            
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        val_acc = clf.score(X_val, y_val)
        tuning_logs.append((desc, val_acc))
        print(f"   [Thử nghiệm] {desc:<35} | Val Acc: {val_acc * 100:.2f}% | Thời gian: {fit_time:.2f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = p
            
    print(f"\n=> BỘ SIÊU THAM SỐ SVM TỐI ƯU NHẤT: {best_params} (Val Acc: {best_val_acc * 100:.2f}%)")
    
    # 3. Vẽ biểu đồ so sánh các cấu hình SVM trên tập Validation
    plt.figure(figsize=(12, 6))
    descs, accs = zip(*tuning_logs)
    bars = plt.barh(descs, [a * 100 for a in accs], color='skyblue', edgecolor='navy')
    plt.title("So sánh độ chính xác Validation giữa các cấu hình siêu tham số SVM", fontsize=13, fontweight="bold")
    plt.xlabel("Validation Accuracy (%)", fontsize=11)
    plt.xlim(0, 100)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va='center', fontsize=10)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    svm_plot_path = os.path.join(RESULTS_DIR, "svm_tuning_comparison.png")
    plt.savefig(svm_plot_path, dpi=300)
    plt.close()
    print(f"[OK] Đã lưu biểu đồ tối ưu tham số SVM tại: {svm_plot_path}")
    
    # 4. Huấn luyện mô hình tối ưu với probability=True
    print("\n--- BƯỚC 2: HUẤN LUYỆN MÔ HÌNH SVM TỐI ƯU (Bật probability=True) ---")
    if best_params['kernel'] == 'linear':
        best_svm = SVC(kernel='linear', C=best_params['C'], probability=True, random_state=42)
    else:
        best_svm = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], probability=True, random_state=42)
        
    t0 = time.time()
    best_svm.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Thời gian huấn luyện SVM: {train_time:.2f} giây")
    
    # 5. Đánh giá trên tập Test
    print("\n--- BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ---")
    t0 = time.time()
    y_pred_test = best_svm.predict(X_test)
    test_time = time.time() - t0
    
    acc = accuracy_score(y_test, y_pred_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='macro', zero_division=0)
    
    print("=" * 55)
    print("      KẾT QUẢ ĐÁNH GIÁ SVM TRÊN TẬP TEST")
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - SVM ({best_params}) - Test Acc: {acc*100:.2f}%", fontsize=15, fontweight="bold")
    plt.xlabel("Nhãn Dự đoán (Predicted Label)", fontsize=12)
    plt.ylabel("Nhãn Thực tế (True Label)", fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_svm.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Đã lưu Confusion Matrix SVM tại: {cm_path}")
    
    # 8. Lưu mô hình đã huấn luyện
    model_save_path = os.path.join(MODELS_DIR, "svm_model.joblib")
    joblib.dump({
        "model": best_svm,
        "best_params": best_params,
        "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "test_time": test_time},
        "class_names": class_names
    }, model_save_path)
    print(f"\n[OK] Đã lưu mô hình SVM tại: {model_save_path}")
    
    return best_svm, best_params

# -------------------------------------------------------------
# 9. HÀM DỰ ĐOÁN ẢNH MỚI (TÍCH HỢP CHO STREAMLIT)
# -------------------------------------------------------------
def predict_image_svm(img_input):
    """
    Dự đoán nhãn của một hình ảnh mới bằng mô hình SVM.
    Trả về: predicted_class, confidence, top_3_predictions
    """
    model_path = os.path.join(MODELS_DIR, "svm_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    
    saved_data = joblib.load(model_path)
    svm_model = saved_data["model"]
    class_names = saved_data["class_names"]
    scaler = joblib.load(scaler_path)
    
    raw_feat = extract_features_from_image(img_input).reshape(1, -1)
    scaled_feat = scaler.transform(raw_feat)
    
    probs = svm_model.predict_proba(scaled_feat)[0]
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
    train_and_evaluate_svm()


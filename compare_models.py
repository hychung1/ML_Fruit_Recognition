"""
BẢNG PHÂN CÔNG ĐỒ ÁN HỌC MÁY - ĐỀ TÀI 07
Module: compare_models.py
Thực hiện: Người 3 & Thống nhất toàn nhóm

Nhiệm vụ chi tiết:
1. Nạp kết quả đánh giá của cả 3 mô hình (KNN, SVM, Random Forest) từ các file đã lưu.
2. Tổng hợp bảng so sánh các chỉ số hiệu năng trên cùng tập Test:
   - Accuracy (%)
   - Macro Precision (%)
   - Macro Recall (%)
   - Macro F1-Score (%)
   - Thời gian suy luận (Inference Speed - ms/ảnh)
3. Xuất bảng tổng hợp ra file results/metrics_summary.csv.
4. Vẽ biểu đồ so sánh trực quan (Grouped Bar Chart) lưu vào results/comparison_chart.png.
5. In phân tích, nhận xét ưu - nhược điểm của 3 thuật toán để đưa vào báo cáo và slide.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def compare_all_models():
    print("=" * 70)
    print("      TỔNG HỢP VÀ SO SÁNH HIỆU NĂNG 3 THUẬT TOÁN TRÊN TẬP TEST")
    print("=" * 70)
    
    models_info = {
        "KNN": os.path.join(MODELS_DIR, "knn_model.joblib"),
        "SVM": os.path.join(MODELS_DIR, "svm_model.joblib"),
        "Random Forest": os.path.join(MODELS_DIR, "rf_model.joblib")
    }
    
    rows = []
    
    for name, path in models_info.items():
        if not os.path.exists(path):
            print(f"[CẢNH BÁO] Chưa tìm thấy file mô hình {name} tại {path}. Hãy chạy train_{name.lower().replace(' ', '')}.py trước!")
            return
            
        data = joblib.load(path)
        m = data["metrics"]
        params = str(data["best_params"])
        test_time_ms = (m["test_time"] / 359) * 1000 # tính theo 359 mẫu test
        
        rows.append({
            "Thuật toán": name,
            "Siêu tham số tối ưu": params,
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Macro Precision (%)": round(m["precision"] * 100, 2),
            "Macro Recall (%)": round(m["recall"] * 100, 2),
            "Macro F1-Score (%)": round(m["f1"] * 100, 2),
            "Tốc độ (ms/ảnh)": round(test_time_ms, 2)
        })
        
    df = pd.DataFrame(rows)
    print("\n[BẢNG TỔNG HỢP SO SÁNH 3 THUẬT TOÁN]:\n")
    print(df.to_string(index=False))
    
    # Lưu ra CSV
    csv_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Đã lưu bảng số liệu vào: {csv_path}")
    
    # Vẽ biểu đồ so sánh các metrics
    metrics_to_plot = ["Accuracy (%)", "Macro Precision (%)", "Macro Recall (%)", "Macro F1-Score (%)"]
    df_melt = pd.melt(df, id_vars=["Thuật toán"], value_vars=metrics_to_plot, var_name="Chỉ số", value_name="Giá trị (%)")
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_melt, x="Chỉ số", y="Giá trị (%)", hue="Thuật toán", palette="viridis")
    plt.title("So sánh hiệu năng giữa KNN, SVM và Random Forest trên tập Test", fontsize=14, fontweight="bold")
    plt.ylim(0, 100)
    plt.ylabel("Phần trăm (%)", fontsize=11)
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ghi chú giá trị lên đầu mỗi cột
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=9, color='black',
                        xytext=(0, 3),
                        textcoords='offset points')
                        
    plt.legend(title="Thuật toán", fontsize=11)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "comparison_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[OK] Đã lưu biểu đồ so sánh tại: {chart_path}")
    
    print("\n" + "=" * 70)
    print("      NHẬN XÉT ĐÁNH GIÁ ĐỒ ÁN (PHỤC VỤ BÁO CÁO VÀ THUYẾT TRÌNH)")
    print("=" * 70)
    print("1. KNN (K-Nearest Neighbors):")
    print("   - Ưu điểm: Đơn giản, trực quan, không mất thời gian huấn luyện (lazy learning).")
    print("   - Nhược điểm: Tốc độ dự đoán chậm do phải tính khoảng cách tới toàn bộ tập train; nhạy cảm với chiều dữ liệu lớn.")
    print("\n2. SVM (Support Vector Machine):")
    print("   - Ưu điểm: Phân tách lớp rất mạnh mẽ trong không gian số chiều cao (800 chiều), tìm siêu phẳng tối ưu với margin cực đại.")
    print("   - Nhược điểm: Huấn luyện lâu hơn khi tập dữ liệu lớn, cần tinh chỉnh kỹ C và kernel.")
    print("\n3. Random Forest:")
    print("   - Ưu điểm: Kháng overfitting tốt nhờ kỹ thuật Bagging (Bootstrap Aggregation), đánh giá được Feature Importance.")
    print("   - Nhược điểm: Kích thước mô hình lớn nếu nhiều cây và độ sâu không giới hạn.")
    print("=" * 70)

if __name__ == "__main__":
    compare_all_models()


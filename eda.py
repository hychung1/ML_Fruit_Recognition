"""
BẢNG PHÂN CÔNG ĐỒ ÁN HỌC MÁY - ĐỀ TÀI 07
Module: eda.py
Thực hiện: Người 1 (KNN) - Phân tích bộ dữ liệu & Trực quan hóa dữ liệu

Nhiệm vụ:
1. Thống kê số lượng mẫu của từng lớp trong 3 tập: Train, Validation, Test.
2. Kiểm tra tính cân bằng giữa các lớp và phân phối dữ liệu.
3. Trực quan hóa phân bố các lớp bằng biểu đồ (Bar plot).
4. Hiển thị mẫu ngẫu nhiên của các lớp rau củ quả.
5. Xuất hình ảnh lưu vào thư mục results/ để phục vụ làm báo cáo/slide.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import random

# Thiết lập đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SETS = ["train", "validation", "test"]

def analyze_dataset():
    print("=" * 60)
    print("      PHÂN TÍCH VÀ TRỰC QUAN HÓA BỘ DỮ LIỆU RAU CỦ QUẢ")
    print("=" * 60)
    
    data = []
    class_names = sorted(os.listdir(os.path.join(DATASET_DIR, "train")))
    print(f"Tổng số lớp nhận diện: {len(class_names)}")
    
    for split in SETS:
        split_dir = os.path.join(DATASET_DIR, split)
        for cls in class_names:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                data.append({"Split": split, "Class": cls, "Count": count})
                
    df = pd.DataFrame(data)
    
    # In thống kê tổng hợp
    summary = df.groupby("Split")["Count"].agg(["sum", "mean", "min", "max"])
    print("\n[Thống kê số lượng ảnh theo tập dữ liệu]:")
    print(summary)
    
    total_imgs = df["Count"].sum()
    print(f"\n=> Tổng cộng toàn bộ dataset: {total_imgs} hình ảnh.")
    
    # 1. Vẽ biểu đồ phân bố số lượng ảnh theo từng lớp
    plt.figure(figsize=(16, 7))
    sns.barplot(data=df, x="Class", y="Count", hue="Split", palette="Set2")
    plt.xticks(rotation=90, fontsize=10)
    plt.title("Phân bố số lượng hình ảnh theo từng lớp trên các tập Train / Val / Test", fontsize=14, fontweight="bold")
    plt.xlabel("Tên lớp rau củ quả", fontsize=12)
    plt.ylabel("Số lượng ảnh", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    dist_path = os.path.join(RESULTS_DIR, "eda_distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"\n[OK] Đã lưu biểu đồ phân bố dữ liệu tại: {dist_path}")
    
    # 2. Vẽ bảng mẫu 16 loại rau củ quả ngẫu nhiên
    sampled_classes = random.sample(class_names, min(16, len(class_names)))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, cls in enumerate(sampled_classes):
        cls_dir = os.path.join(DATASET_DIR, "train", cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            img_path = os.path.join(cls_dir, random.choice(imgs))
            img = Image.open(img_path).convert("RGB")
            axes[idx].imshow(img)
            axes[idx].set_title(cls.capitalize(), fontsize=11, fontweight="bold")
        axes[idx].axis("off")
        
    plt.suptitle("Một số hình ảnh mẫu trong bộ dữ liệu huấn luyện", fontsize=15, fontweight="bold")
    plt.tight_layout()
    samples_path = os.path.join(RESULTS_DIR, "eda_samples.png")
    plt.savefig(samples_path, dpi=300)
    plt.close()
    print(f"[OK] Đã lưu hình ảnh mẫu minh họa tại: {samples_path}")
    print("=" * 60)

if __name__ == "__main__":
    analyze_dataset()


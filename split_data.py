import os
import shutil
import random

# Đường dẫn thư mục gốc chứa 9 hãng xe
root_dir = "data_png"

# Thư mục đầu ra
output_dir = "dataset_split"

# Tỉ lệ chia
train_ratio = 0.7
test_ratio = 0.1
val_ratio = 0.2

# Tạo cấu trúc thư mục đầu ra
splits = ['train', 'test', 'val']
subfolders = ['images', 'labels']

for split in splits:
    for sub in subfolders:
        os.makedirs(os.path.join(output_dir, split, sub), exist_ok=True)

# Lặp qua từng hãng xe
for brand in os.listdir(root_dir):
    brand_path = os.path.join(root_dir, brand)
    if not os.path.isdir(brand_path):
        continue

    # Lấy danh sách ảnh
    images = [f for f in os.listdir(brand_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    total = len(images)
    train_count = int(total * train_ratio)
    test_count = int(total * test_ratio)
    val_count = total - train_count - test_count

    print(f" {brand}: {total} ảnh -> train={train_count}, test={test_count}, val={val_count}")

    # Chia ảnh
    splits_data = {
        'train': images[:train_count],
        'test': images[train_count:train_count + test_count],
        'val': images[train_count + test_count:]
    }

    # Copy ảnh và tạo file label tương ứng
    for split, img_list in splits_data.items():
        for img_name in img_list:
            src_img = os.path.join(brand_path, img_name)
            dst_img = os.path.join(output_dir, split, 'images', img_name)
            shutil.copy2(src_img, dst_img)

            # Tạo file nhãn tương ứng (nếu chưa có)
            # label_name = os.path.splitext(img_name)[0] + ".txt"
            # src_label = os.path.join(root_dir.replace("data_png", "labels"), brand, label_name)
            # dst_label = os.path.join(output_dir, split, 'labels', label_name)

            # if os.path.exists(src_label):
            #     shutil.copy2(src_label, dst_label)
            # else:
            #     # Nếu chưa có file label, tạo file rỗng
            #     open(dst_label, 'w').close()

print("\n✅ Hoàn tất chia dữ liệu theo tỉ lệ 70/20/10 cho từng hãng!")

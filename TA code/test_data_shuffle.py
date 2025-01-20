import os
import random
import shutil
import csv

# 目標資料夾路徑
data_test_path = "data/test"

# 確保測試資料夾存在
os.makedirs(data_test_path, exist_ok=True)

# 需要處理的動物資料夾名稱
target_animals = {"elephant", "jaguar", "lion", "parrot", "penguin"}

# 儲存圖片對應關係的 CSV 路徑
csv_file_path = os.path.join(data_test_path, "test_labels.csv")

# 收集所有圖片的路徑和標籤
image_paths = []
for animal in target_animals:
    animal_path = os.path.join(data_test_path, animal)
    if os.path.isdir(animal_path):
        images = [img for img in os.listdir(animal_path) if os.path.isfile(os.path.join(animal_path, img)) and img.lower().endswith('.jpg')]
        image_paths.extend([(os.path.join(animal_path, img), animal) for img in images])

# 隨機打亂所有圖片
random.shuffle(image_paths)

# 初始化 CSV 文件
with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Image_Number", "Animal"])

    # 遍歷圖片並重新命名
    for image_counter, (image_path, animal) in enumerate(image_paths, start=1):
        new_image_name = f"{image_counter}.jpg"
        new_image_path = os.path.join(data_test_path, new_image_name)

        # 移動圖片並重新命名
        shutil.move(image_path, new_image_path)

        # 將圖片資訊寫入 CSV
        csv_writer.writerow([image_counter, animal])

        # 刪除空資料夾
        animal_folder = os.path.dirname(image_path)
        if not os.listdir(animal_folder):
            os.rmdir(animal_folder)

print("圖片重命名與 CSV 建立完成！")

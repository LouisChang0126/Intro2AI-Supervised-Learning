import os
import random
import shutil

# 原始資料夾路徑和目標資料夾路徑
original_data_path = "./original_data"
data_train_path = "data/train"
data_test_path = "data/test"

# 確保目標資料夾存在
os.makedirs(data_train_path, exist_ok=True)
os.makedirs(data_test_path, exist_ok=True)

# 需要處理的動物資料夾名稱
target_animals = {"elephant", "jaguar", "lion", "parrot", "penguin"}

# 遍歷原始資料夾中的每個動物資料夾
for animal in os.listdir(original_data_path):
    if animal in target_animals:
        animal_path = os.path.join(original_data_path, animal)
        if os.path.isdir(animal_path):
            
            # 獲取所有圖片檔案並隨機打亂順序
            images = [img for img in os.listdir(animal_path) if os.path.isfile(os.path.join(animal_path, img)) and img.lower().endswith('.jpg')]
            random.shuffle(images)

            # 分配圖片
            if len(images) > 2000:
                train_images = images[:1600]
                test_images = images[1600:1600 + 400]
            else:
                train_images = images[:int(len(images) * 0.8)]
                test_images = images[int(len(images) * 0.8):]


            # 創建目標子資料夾
            train_animal_path = os.path.join(data_train_path, animal)
            test_animal_path = os.path.join(data_test_path, animal)
            os.makedirs(train_animal_path, exist_ok=True)
            os.makedirs(test_animal_path, exist_ok=True)

            # 將圖片複製到目標資料夾
            for img in train_images:
                shutil.copy(os.path.join(animal_path, img), os.path.join(train_animal_path, img))

            for img in test_images:
                shutil.copy(os.path.join(animal_path, img), os.path.join(test_animal_path, img))

print("圖片分配完成！")

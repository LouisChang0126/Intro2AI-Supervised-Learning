import os

# 目標資料夾路徑
data_train_path = "data/train"

# 需要處理的動物資料夾名稱
target_animals = {"elephant", "jaguar", "lion", "parrot", "penguin"}

# 遍歷資料夾中的每個動物資料夾
for animal in target_animals:
    animal_path = os.path.join(data_train_path, animal)
    if os.path.isdir(animal_path):
        
        # 獲取所有圖片檔案
        images = [img for img in os.listdir(animal_path) if os.path.isfile(os.path.join(animal_path, img)) and img.lower().endswith('.jpg')]

        # 按順序重新命名
        for idx, img in enumerate(sorted(images), start=1):
            old_image_path = os.path.join(animal_path, img)
            new_image_name = f"{idx}.jpg"
            new_image_path = os.path.join(animal_path, new_image_name)
            
            # 重新命名圖片
            os.rename(old_image_path, new_image_path)

print("圖片重新命名完成！")

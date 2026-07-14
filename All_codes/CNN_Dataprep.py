import os
import shutil
import random

def split_images(source_folder, destination_folder, train_ratio=0.75):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    train_folder = os.path.join(destination_folder, 'train')
    test_folder = os.path.join(destination_folder, 'test')
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    day_wise_images = {}
    
    # Group images by day
    for filename in os.listdir(source_folder):
        if filename.endswith(".png"):
            day_label = filename.split("_")[0]  # Extract 'Day X' part
            if day_label not in day_wise_images:
                day_wise_images[day_label] = []
            day_wise_images[day_label].append(filename)
    
    # Split images into train and test folders
    for day, images in day_wise_images.items():
        random.shuffle(images)
        split_index = int(len(images) * train_ratio)
        
        train_images = images[:split_index]
        test_images = images[split_index:]
        
        day_train_folder = os.path.join(train_folder, day)
        day_test_folder = os.path.join(test_folder, day)
        
        os.makedirs(day_train_folder, exist_ok=True)
        os.makedirs(day_test_folder, exist_ok=True)
        
        for img in train_images:
            shutil.move(os.path.join(source_folder, img), os.path.join(day_train_folder, img))
        
        for img in test_images:
            shutil.move(os.path.join(source_folder, img), os.path.join(day_test_folder, img))
    
    print("Images successfully split into train and test folders.")

# Example usage
source_folder = "/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Image_data_working/Mouse wound photos-updated/Cropped images"
destination_folder = "/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/CNN Data"
split_images(source_folder, destination_folder)

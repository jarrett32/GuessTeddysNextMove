import os
import random
import shutil
import sys


def split_data(base_path, val_percentage=0.05):
    source_path = os.path.join(base_path, 'images')
    train_path = os.path.join(source_path, 'train')
    val_path = os.path.join(source_path, 'val')

    for path in [train_path, val_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
    
    os.makedirs(train_path)
    os.makedirs(val_path)

    image_files = [f for f in os.listdir(source_path) if f.endswith('.jpg') and os.path.isfile(os.path.join(source_path, f))]
    num_val = int(len(image_files) * val_percentage)
    val_images = random.sample(image_files, num_val)

    # Move files to respective directories
    for img in image_files:
        src = os.path.join(source_path, img)
        dst = os.path.join(val_path if img in val_images else train_path, img)
        shutil.move(src, dst)  # Using move since we're reorganizing within images/

    print(f"Split complete:")
    print(f"Training set: {len(image_files) - num_val} images")
    print(f"Validation set: {num_val} images")

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python split_train_val_set.py <dataset_name>")
        sys.exit(1)

    base_path = sys.argv[1]
    if not os.path.exists(base_path):
        print(f"Error: {base_path} does not exist")
        sys.exit(1)

    if not os.path.exists(os.path.join(base_path, 'images')):
        print(f"Error: {base_path} does not contain an images folder")
        sys.exit(1)

    split_data(base_path)

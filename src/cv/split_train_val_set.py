import os
import random
import shutil


def split_data(val_percentage=0.05):
    train_image_path = 'src/train/teddy/images/train'
    val_image_path = 'src/train/teddy/images/val'
    train_label_path = 'src/train/teddy/labels/train'
    val_label_path = 'src/train/teddy/labels/val'

    os.makedirs(val_image_path, exist_ok=True)
    os.makedirs(val_label_path, exist_ok=True)

    image_files = [f for f in os.listdir(train_image_path) if f.endswith('.jpg')]
    num_val = int(len(image_files) * val_percentage)
    val_images = random.sample(image_files, num_val)

    for img in val_images:
        src_img = os.path.join(train_image_path, img)
        dst_img = os.path.join(val_image_path, img)

        label = img.replace('.jpg', '.txt')
        src_label = os.path.join(train_label_path, label)
        dst_label = os.path.join(val_label_path, label)

        shutil.move(src_img, dst_img)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)

    print(f"Moved {len(val_images)} images and their labels to validation set")

if __name__ == '__main__':
    split_data()

import os
import random
import shutil
import sys

def split_data(base_path, val_percentage=0.2):
    images_dir = os.path.join(base_path, 'images')
    labels_dir = os.path.join(base_path, 'labels')
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png')) 
                  and os.path.isfile(os.path.join(images_dir, f))]
    
    num_val = int(len(image_files) * val_percentage)
    val_images = set(random.sample(image_files, num_val))

    for img_file in image_files:
        is_val = img_file in val_images
        split = 'val' if is_val else 'train'
        
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(images_dir, split, img_file)
        shutil.move(src_img, dst_img)
        
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(labels_dir, split, label_file)
        
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            print(f"Creating empty label file for {img_file}")
            with open(dst_label, 'w') as f:
                pass  

    print(f"Split complete:")
    print(f"Training set: {len(image_files) - num_val} images")
    print(f"Validation set: {num_val} images")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python split_train_val_set.py <path_to__dataset>")
        sys.exit(1)

    base_path = sys.argv[1]
    
    if not os.path.exists(base_path):
        print(f"Error: {base_path} does not exist")
        sys.exit(1)

    for required_dir in ['images', 'labels']:
        if not os.path.exists(os.path.join(base_path, required_dir)):
            print(f"Error: {base_path} does not contain {required_dir} folder")
            sys.exit(1)

    split_data(base_path)

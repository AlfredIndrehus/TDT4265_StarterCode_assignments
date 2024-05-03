import os
import random
import shutil

# Define paths
images_folder = '/datasets/tdt4265/ad/NAPLab-LiDAR/images'
labels_folder = '/datasets/tdt4265/ad/NAPLab-LiDAR/labels_yolo_v1.1'
output_folder = 'project/YOLO/project_data/data'

# Get list of image and label files
image_files = sorted(os.listdir(images_folder))
label_files = sorted(os.listdir(labels_folder))

# Filter out non-image and non-label files
image_files = [f for f in image_files if f.endswith('.PNG')]
label_files = [f for f in label_files if f.endswith('.txt')]

# Ensure lists are correctly aligned
assert len(image_files) == len(label_files), "Mismatch in number of images and labels"
for img_file, lbl_file in zip(image_files, label_files):
    assert img_file.split('.')[0] == lbl_file.split('.')[0], "Mismatched file names: {} and {}".format(img_file, lbl_file)

# Shuffle while maintaining correspondence
combined = list(zip(image_files, label_files))
random.shuffle(combined)
image_files_shuffled, label_files_shuffled = zip(*combined)

# Define train, test, and validation splits
train_split = 0.7
test_split = 0.2
val_split = 0.1  # Not used explicitly, remainder goes to validation

# Calculate split indices
total_files = len(image_files_shuffled)
train_index = int(total_files * train_split)
test_index = train_index + int(total_files * test_split)

# Create output directories
output_images_folder = os.path.join(output_folder, 'images')
output_labels_folder = os.path.join(output_folder, 'labels')
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

# Create train, test, and validation sub-folders
for subset in ['train', 'test', 'val']:
    os.makedirs(os.path.join(output_images_folder, subset), exist_ok=True)
    os.makedirs(os.path.join(output_labels_folder, subset), exist_ok=True)

# Move files to output directories
for i, (image_file, label_file) in enumerate(zip(image_files_shuffled, label_files_shuffled)):
    subset = 'train' if i < train_index else 'test' if i < test_index else 'val'
    shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_images_folder, subset, image_file))
    shutil.copy(os.path.join(labels_folder, label_file), os.path.join(output_labels_folder, subset, label_file))

#Splitting the data into training and testing data, while also shuffling the data

import os
import random
import shutil

# Define paths

images_folder = 'data/LiDAR/archive/images'  #images
labels_folder = 'data/LiDAR/archive/outputs' #output should have the .XML files (ong Pascal Voc formate)
output_folder = 'data/LiDAR/archive/data' #empty folder where the shuffled val, train and and test data will be

#Get list of image and label files
image_files = os.listdir(images_folder)
label_files = os.listdir(labels_folder)

# Shuffle while maintaining correspondence
combined = list(zip(image_files, label_files))
random.shuffle(combined)
image_files_shuffled, label_files_shuffled = zip(*combined)

# Define train, test, and validation splits
train_split = 0.7
test_split = 0.2
val_split = 0.1

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
    if i < train_index:
        shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_images_folder, 'train'))
        shutil.copy(os.path.join(labels_folder, label_file), os.path.join(output_labels_folder, 'train'))
    elif i < test_index:
        shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_images_folder, 'test'))
        shutil.copy(os.path.join(labels_folder, label_file), os.path.join(output_labels_folder, 'test'))
    else:
        shutil.copy(os.path.join(images_folder, image_file), os.path.join(output_images_folder, 'val'))
        shutil.copy(os.path.join(labels_folder, label_file), os.path.join(output_labels_folder, 'val'))
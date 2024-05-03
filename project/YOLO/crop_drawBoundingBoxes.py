import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def draw_bounding_boxes(image_path, label_path):
    # Open the image
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Open the corresponding label file
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id = parts[0]
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                box_width = float(parts[3]) * width
                box_height = float(parts[4]) * height

                # Calculate the coordinates of the bounding box
                left = x_center - (box_width / 2)
                top = y_center - (box_height / 2)
                right = x_center + (box_width / 2)
                bottom = y_center + (box_height / 2)

                # Draw the bounding box
                draw.rectangle([left, top, right, bottom], outline="red", width=2)
    
    return img

def main():
    # Specify the directory paths
    images_dir = "project_data_cropped/data/images/train"  # Adjust if needed
    labels_dir = "project_data_cropped/data/labels/train"  # Adjust if needed
    sample_images = ["frame_000001.PNG", "frame_000167.PNG", "frame_000282.PNG"]  # Replace with actual filenames

    for filename in sample_images:
        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, filename.replace('.PNG', '.txt'))
        img = draw_bounding_boxes(image_path, label_path)

        # Display the image
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.title(f"Image with Bounding Boxes: {filename}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()

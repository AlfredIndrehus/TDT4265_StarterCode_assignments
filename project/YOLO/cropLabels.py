import os

def adjust_labels(input_folder, output_folder, original_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    crop_percentage = 0.3  # 30% crop from the top
    new_height = original_height * 0.7  # Height after cropping

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, 'r') as file:
                lines = file.readlines()

            with open(output_path, 'w') as file:
                for line in lines:
                    parts = line.strip().split()
                    class_id = parts[0]
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Adjust y_center downwards to account for the crop
                    new_y_center = (y_center * original_height - original_height * crop_percentage) / new_height

                    # Write the adjusted bounding box if it's still within the new image bounds
                    if 0 <= new_y_center <= 1:
                        file.write(f"{class_id} {x_center} {new_y_center} {width} {height}\n")

def main():
    base_dir = "project/YOLO/project_data/data/labels"  # Update this path
    new_base_dir = "project_data_cropped/data/labels"  # Update this path
    original_height = 128  # Example original height in pixels, adjust as necessary
    categories = ["test", "train", "val"]

    for category in categories:
        input_folder = os.path.join(base_dir, category)
        output_folder = os.path.join(new_base_dir, category)
        adjust_labels(input_folder, output_folder, original_height)

if __name__ == "__main__":
    main()

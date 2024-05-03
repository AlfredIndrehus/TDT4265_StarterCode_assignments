import os
from PIL import Image

def crop_image(input_folder, output_folder):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith((".PNG", ".jpg", ".jpeg")):  # Add/check other formats as needed
            # Open the image file
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                # Calculate the crop dimensions
                width, height = img.width, img.height
                new_height = height * 70 // 100

                # Crop the image (left, upper, right, lower)
                img_cropped = img.crop((0, height * 30 // 100, width, height))

                # Save the cropped image to the output directory
                output_path = os.path.join(output_folder, filename)
                img_cropped.save(output_path)

def main():
    # You might need to adjust these paths based on where the script is being run from
    base_dir = "/home/andremaa/Documents/TDT4265_StarterCode_assignments/project/YOLO/project_data/data/images"
    new_base_dir = "project_data_cropped/data/images"
    categories = ["test", "train", "val"]

    for category in categories:
        input_folder = os.path.join(base_dir, category)
        output_folder = os.path.join(new_base_dir, category)
        crop_image(input_folder, output_folder)


if __name__ == "__main__":
    main()

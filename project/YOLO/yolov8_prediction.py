from ultralytics import YOLO

#Loading the model
model = YOLO('runs/detect/yolov8m_crop_Fraction_final_run/weights/best.pt')


# Define path to directory containing images and videos for inference
source = "project/YOLO/project_data/data/images/test/frame_000022.PNG"

# Run inference on the source
results = model.predict(source)  # generator of Results objects

print("done")




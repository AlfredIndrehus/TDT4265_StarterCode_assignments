from ultralytics import YOLO

#Loading the model
model = YOLO('runs/detect/yolov8m_run1/weights/best.pt')

#Prediction

results = model.predict('project/YOLO/project_data/data/images/test/frame_000005.PNG', save = True)



from ultralytics import YOLO

#Loading the model
model = YOLO('runs/detect/yolov8s_custom/weights/best.pt')

#Prediction

results = model.predict('project/YOLO/project_data/data/images/test/frame_000005.PNG', save = True)



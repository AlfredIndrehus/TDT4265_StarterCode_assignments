
from ultralytics import YOLO

# Load the model.

model = YOLO('yolov8m.pt')

# Training.
results = model.train(
   data='project/YOLO/data.yaml',
   epochs=30,
   batch=-1,
   crop_fraction = 0.7,
   name='yolov8m_crop_Fraction_final_run',)


from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8s.pt')


'''
Batch: -1 - Autobatch, dynamicallyt adjusts batch size based on GPU memory available

'''
# Training.
results = model.train(
   data='project/YOLO/data.yaml',
   epochs=50,
   batch=8,
   shear = 45,
   name='yolov8s_shear_90',)





from ultralytics import YOLO

model_detect = YOLO('yolov8n.pt')
model_segment = YOLO('yolov8n-seg.pt')

results = model_detect.track(source=0, show=True)
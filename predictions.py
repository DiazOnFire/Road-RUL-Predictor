import ultralytics
from ultralytics import YOLO

model = YOLO("G:/ml notebooks/Road_RUL/datasets/runs/detect/train3/weights/best.pt")

# Use the model to detect object - goat
model.predict(source="G:/ml notebooks/Road_RUL/datasets/datasets/test/images/India_001989.jpg", save=True, show=True)
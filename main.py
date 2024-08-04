from ultralytics import YOLO

# Load a model
model = YOLO("yolov8l-pose.pt")


# Use the model
model.train(data="coco8-pose.yaml", epochs=500,device='0',pretrained=False,seed=42)  # train the model


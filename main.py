from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")


# Use the model
model.train(data="VOC.yaml", epochs=10,device='0',pretrained=False,seed=42)  # train the model


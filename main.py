from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.yaml")


# Use the model
model.train(data="lsp.yaml", epochs=10,device='0',pretrained=False,seed=42)  # train the model


from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.pt")


# Use the model
model.train(data="lsp.yaml", epochs=500,device='0',pretrained=False,seed=42)  # train the model


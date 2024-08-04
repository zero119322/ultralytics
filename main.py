from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.pt")


# Use the model
model.train(data="lsp.yaml", epochs=50,device='0',pretrained=true,seed=42)  # train the model


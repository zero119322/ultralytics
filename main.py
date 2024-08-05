from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.pt")


# Use the model
model.train(data="lsp-extend.yaml", epochs=50,device='0',pretrained=True,seed=42)  # train the model


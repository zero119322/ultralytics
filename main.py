from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/v8/yolov8n.yaml")  # build a new model from scratch


# Use the model
model.train(data="VOC.yaml", epochs=10,device='0',pretrained=False,seed=42,weights=None)  # train the model


from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch


# Use the model
model.train(data="VOC.yaml", epochs=10,device='0')  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
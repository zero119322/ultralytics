from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    # model = YOLO('yolov8m.pt')  # load an official model
    model = YOLO('ultralytics/runs/detect/train/weights/best.pt')  # 使用相对路径
    results = model.predict(source="ultralytics/assets", device='0',save=True)  # predict on an image
    print(results)

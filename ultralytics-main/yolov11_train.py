from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'yolov11m.yaml')  # 此处以 m 为例，只需写yolov11m即可定位到m模型
    model.train(data=r'data.yaml',
                imgsz=320,
                epochs=100,
                single_cls=True,  # 多类别设置False
                batch=4,
                workers=8,
                device='0',
                )

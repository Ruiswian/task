import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/ruiswian/ultralytics-main/runs/detect/train5/weights/best.pt')
    model.predict(source='/home/ruiswian/ultralytics-main/images.jpg',
                  imgsz=640,
                  device='cpu',
                  save=True,  # 启用保存
                  save_dir='/home/ruisiwan/ultralytics-main/runs/detect/exp2',  # 明确指定保存目录
                  )

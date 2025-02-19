import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt')
    model.val(data='data.yaml',
              imgsz=320,
              batch=4,
              split='test',
              workers=8,
              device='cpu',
              )


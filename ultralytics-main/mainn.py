import cv2
import numpy as np
from typing import Tuple
from ultralytics import YOLO

# 加载 YOLO 模型
#model = YOLO("/home/ruiswian/ultralytics-main/runs/detect/train5/weights/best.pt") #自己训练出来的模型
model = YOLO("/home/ruiswian/ultralytics-main/yolo11m.pt")  #官方预训练模型

def preprocess_image(image_path: str, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    # 读取图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法读取图像：{image_path}")

    # 获取原始图像的尺寸
    original_h, original_w = original_image.shape[:2]

    # YOLO 模型要求填充后的图像尺寸为目标尺寸
    # 计算缩放比例
    scale = min(target_size[0] / original_w, target_size[1] / original_h)

    # 计算缩放后的尺寸
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # 缩放图像
    resized_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算填充尺寸
    pad_w = (target_size[0] - new_w) // 2
    pad_h = (target_size[1] - new_h) // 2

    # 创建目标尺寸的图像（填充为灰色）
    padded_image = np.full((target_size[1], target_size[0], 3), 114, dtype=np.uint8)

    # 将缩放后的图像放置在填充图像的中心
    padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_image

    # 返回图像
    return padded_image

def detect_objects(image_path, save_path="output2.jpg", conf_threshold=0.3):
    # 预处理图像
    image = preprocess_image(image_path)

    # YOLO 进行推理
    results = model(image)  # 进行推理

    # 查看结果
    print("检测到的目标数量:", len(results))

    # 解析检测结果
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # 目标框 (x1, y1, x2, y2)
        confidences = result.boxes.conf.cpu().numpy()  # 置信度
        class_ids = result.boxes.cls.cpu().numpy()  # 目标类别

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if conf < conf_threshold:  # 低于阈值的检测结果会被忽略
                continue

            print(f"目标：{model.names[int(cls_id)]}, 置信度：{conf:.2f}, 坐标：{box}")

            x1, y1, x2, y2 = map(int, box.tolist())
            label = f"{model.names[int(cls_id)]}: {conf:.2f}"

            # 画框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存检测结果
    cv2.imwrite(save_path, image)  # 保存结果
    print(f"检测完成，结果已保存至 {save_path}")

    # 显示检测结果
    cv2.imshow("YOLO Detection", image)  # 显示图像
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 销毁窗口

# 测试代码
image_path = "/home/ruiswian/ultralytics-main/90.jpg"  # 你的图片路径
detect_objects(image_path, conf_threshold=0.3)  # 设置阈值为 0.5

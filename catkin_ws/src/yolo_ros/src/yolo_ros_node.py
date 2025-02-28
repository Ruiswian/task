#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO  # 导入 ultralytics 库
from std_msgs.msg import String

class YOLO_ROS:
    def __init__(self):
        print("正在初始化 YOLO_ROS 类...")
        rospy.init_node('yolo_ros_node', anonymous=True)
        self.bridge = CvBridge()

        # 摄像头初始化
        print("正在初始化摄像头...")
        self.cap = cv2.VideoCapture(0)  # 0 是默认的摄像头ID，若有多个摄像头可以改为1,2等
        if not self.cap.isOpened():
            rospy.logerr("摄像头打开失败！")
            exit()

        # 发布图像到话题
        print("正在初始化 ROS 话题...")
        self.image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
        self.bbox_pub = rospy.Publisher("/yolo/bounding_boxes", String, queue_size=10)

        # 加载 YOLOv11 模型
        self.model_path = "/home/ruiswian/ultralytics-main/runs/detect/train5/weights/best.pt"
        print("模型路径:", self.model_path)
        self.model = YOLO(self.model_path)  # 修改为实际模型路径
        print("YOLO 模型加载完成！")
        print("模型信息:", self.model)

    def capture_and_publish_image(self):
        print("开始捕获并发布图像...")
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()  # 从摄像头读取图像
            if not ret:
                rospy.logerr("无法读取摄像头图像！")
                break

            try:
                # 将 OpenCV 图像转换为 ROS 图像消息
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.image_pub.publish(ros_image)  # 发布图像到 /camera/image_raw 话题
            except CvBridgeError as e:
                rospy.logerr("CvBridge 转换失败: %s", str(e))

            # 使用 YOLOv11 模型进行物体检测
            results = self.model(frame)  # 使用 YOLOv11 进行推理
            print("推理结果:", results)

            # 发布检测到的物体坐标
            for result in results:  # 可能有多个检测框，使用结果中的第一个
                boxes = result.boxes  # 获取所有检测框
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xywh[0]  # 获取框坐标（相对坐标）
                        confidence = box.conf[0]  # 置信度
                        label = result.names[int(box.cls[0])]  # 标签
                        if (confidence>=0.5):
                           print(f"检测到物体: {label}, 置信度: {confidence}, 边界框: ({x1}, {y1}, {x2}, {y2})")
                        bbox_msg = f"{label} {confidence} {x1} {y1} {x2} {y2}"
                        self.bbox_pub.publish(bbox_msg)
                else:
                    print("未检测到任何物体")

            # 显示图像（可选）
            cv2.imshow("YOLOv11 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
                break

    def shutdown(self):
        # 关闭摄像头和窗口
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        print("启动 YOLO_ROS 节点...")
        yolo_ros = YOLO_ROS()
        yolo_ros.capture_and_publish_image()  # 捕获并发布图像
        rospy.spin()
    except rospy.ROSInterruptException:
        print("ROS 节点中断")
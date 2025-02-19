#!/usr/bin/env python3
# coding:utf-8

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def image_callback(msg):
    global bridge
    cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围
    lower_red1 = np.array([0, 125, 125])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 125, 125])
    upper_red2 = np.array([180, 255, 255])

    # 创建红色区域的掩码
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 查找红色区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算红色区域的中心坐标
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"红色区域中心坐标: ({cX}, {cY})")
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = f"({cX},{cY})"
            cv2.putText(cv_img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # 显示结果
    cv2.imshow("Red Areas", cv_img)
    cv2.waitKey(1)

    # 将处理后的图像转换回ROS消息并发布
    processed_image_msg = bridge.cv2_to_imgmsg(cv_img, "bgr8")
    pub.publish(processed_image_msg)


if __name__ == '__main__':
    rospy.init_node('image_subscriber', anonymous=True)
    pub = rospy.Publisher('processed_image_topic', Image, queue_size=10)
    bridge = CvBridge()
    rospy.Subscriber('camera_image_raw', Image, image_callback)
    rospy.spin()
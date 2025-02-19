#!/usr/bin/env python
# coding:utf-8

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def publish_image():
    rospy.init_node('camera_node', anonymous=True)  # 定义节点
    image_pub = rospy.Publisher('/camera_image_raw', Image, queue_size=10)  # 创建Publisher，发布图像话题

    bridge = CvBridge()  # 创建CvBridge对象
    rospy.loginfo("CvBridge对象创建成功")

    capture = cv2.VideoCapture(0, cv2.CAP_V4L2)  # 定义摄像头
    if not capture.isOpened():
        rospy.logerr("无法打开摄像头")
        return

        # 获取摄像头帧率
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 10  # 默认帧率
        rospy.logwarn(f"无法获取摄像头帧率，使用默认帧率: {fps} Hz")
    else:
        rospy.loginfo(f"摄像头帧率: {fps} Hz")

    # 设置发布频率
    rate = rospy.Rate(fps)

    try:
       while not rospy.is_shutdown():  # Ctrl C正常退出，如果异常退出会报错device busy！
             ret, frame = capture.read()
             if not ret:  # 如果有画面再执行
                rospy.logerr("无法读取摄像头帧")
                break

             frame = cv2.flip(frame, 1)  # 水平镜像操作
             cv2.imshow("Camera Output", frame) # 显示图像从而调试
             cv2.waitKey(1)
             #检查帧是否有效
             if frame is None:
                 rospy.logerr("读取到的帧为空")
                 continue
             try:
                ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")# 将OpenCV图像转换为ROS图像消息
                rospy.loginfo("图像转换成功，准备发布")
             except Exception as e:
                rospy.logerr(f"Error converting image: {e}")
                continue

            #发布图像
             try:
                image_pub.publish(ros_image)
                rospy.loginfo("图像发布成功")
             except Exception as e:
                rospy.logerr(f"图像发布失败: {e}")

             rate.sleep() #控制发布频率

    except rospy.ROSInterruptException :
       rospy.loginfo("ROS中断，退出程序")
    finally:   # 释放摄像头资源
       if capture is not None:
          capture.release()
          rospy.loginfo("摄像头资源已释放")
       cv2.destroyAllWindows()
       rospy.loginfo("程序退出成功")

if __name__ == '__main__':
    try:
        publish_image()
    except rospy.ROSInterruptException:
        pass
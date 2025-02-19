import cv2
import numpy as np

# 调用摄像头摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("摄像头未打开，请检查设备连接！")
    exit()  # 退出程序
else:
    print("摄像头已成功打开！")
while(True):
# 获取摄像头拍摄到的画面
    ret, frame = cap.read()
# 读取图像
    image = frame
    image2 = image.copy()

# 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色的HSV范围
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
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
          cv2.rectangle(image, (x,y), (x+w,y+h),(0, 0, 255), 2)
          text = f"({cX},{cY})"
          cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    img =cv2.hconcat([image2,image])
       # 显示结果
    cv2.imshow("Red Areas", image)
    if cv2.waitKey(5) &0xFF == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()

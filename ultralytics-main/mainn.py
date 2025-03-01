def detect_objects(image_path, save_path="output.jpg", conf_threshold=0.5):
    image, original_image = preprocess_image(image_path)

    # YOLO 进行推理
    results = model(image, verbose=True)  # 设置 verbose=True 输出详细信息

    # 查看结果
    print("检测到的目标数量:", len(results))

    # 解析检测结果
    for result in results:
        boxes = result.boxes.xyxy  # 目标框 (x1, y1, x2, y2)
        confidences = result.boxes.conf  # 置信度
        class_ids = result.boxes.cls  # 目标类别

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if conf < conf_threshold:  # 低于阈值的检测结果会被忽略
                continue

            print(f"目标：{model.names[int(cls_id)]}, 置信度：{conf:.2f}, 坐标：{box}")

            x1, y1, x2, y2 = map(int, box.tolist())
            label = f"{model.names[int(cls_id)]}: {conf:.2f}"

            # 画框
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存检测结果
    cv2.imwrite(save_path, original_image)
    print(f"检测完成，结果已保存至 {save_path}")


# 测试代码
image_path = "input.jpg"  # 你的图片路径
detect_objects(image_path, conf_threshold=0.5)  # 设置阈值为 0.5

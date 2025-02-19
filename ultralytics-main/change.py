import xml.etree.ElementTree as ET
import os, cv2
import numpy as np
from os import listdir
from os.path import join

classes = []


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(xmlpath, xmlname, imgpath, txtpath):
    with open(xmlpath, "r", encoding='utf-8') as in_file:
        txtname = xmlname[:-4] + '.txt'
        txtfile = os.path.join(txtpath, txtname)
        tree = ET.parse(in_file)
        root = tree.getroot()
        filename = root.find('filename')
        img = cv2.imdecode(np.fromfile('{}/{}.{}'.format(imgpath, xmlname[:-4], postfix), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        res = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                classes.append(cls)
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
        if len(res) != 0:
            with open(txtfile, 'w+') as f:
                f.write('\n'.join(res))


if __name__ == "__main__":
    postfix = 'jpg'  # 图像后缀
    basepath = r'/home/ruiswian/ultralytics-main/data/valid'  # 基础路径，包含图片和XML文件
    imgpath = os.path.join(basepath, 'images')  # 图像文件路径
    xmlpath = os.path.join(basepath, 'annotations')  # xml文件文件路径
    txtpath = os.path.join(basepath, 'labels')  # 生成的txt文件路径

    # 遍历基础路径下的所有文件
    error_file_list = []
    for filename in os.listdir(basepath):
        filepath = os.path.join(basepath, filename)
        if filename.endswith('.xml') or filename.endswith('.XML'):
            # 如果是XML文件，移动到annotations目录
            os.rename(filepath, os.path.join(xmlpath, filename))
            print(f'file {filename} moved to annotations directory.')
        elif filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # 如果是图片文件，移动到images目录
            os.rename(filepath, os.path.join(imgpath, filename))
            print(f'file {filename} moved to images directory.')
        else:
            print(f'file {filename} is not xml or image format.')

    # 处理XML文件
    list = os.listdir(xmlpath)
    for i in range(0, len(list)):
        try:
            path = os.path.join(xmlpath, list[i])
            if ('.xml' in path) or ('.XML' in path):
                convert_annotation(path, list[i], imgpath, txtpath)
                print(f'file {list[i]} convert success.')
            else:
                print(f'file {list[i]} is not xml format.')
        except Exception as e:
            print(f'file {list[i]} convert error.')
            print(f'error message:\n{e}')
            error_file_list.append(list[i])
    print(f'this file convert failure\n{error_file_list}')
    print(f'Dataset Classes:{classes}')
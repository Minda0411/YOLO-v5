import xml.etree.ElementTree as ET
import os
import argparse
from utils.tien_utility import findAllImagFiles, split_filename_extension
import cv2
from utils.Draw_Utility import draw_rectangle
#classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

##def convert_annotation(image_id, list_file):
##    image_id = image_id.strip('\n')
##    in_file = open(train_path+'/Annotations/%s.xml'%(image_id))
##    tree=ET.parse(in_file)
##    root = tree.getroot()
##
##    for obj in root.iter('object'):
##        difficult = obj.find('difficult').text
##        cls = obj.find('name').text
##        if cls not in classes or int(difficult)==1:
##            continue
##        cls_id = classes.index(cls)
##        xmlbox = obj.find('bndbox')
##        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
##        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
##

def convert_annotation(data_path, image_id, list_file, classes):
    image_id = image_id.strip('\n')
    in_file = open(data_path+'/Annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    return

def read_annotation(data_path, image_id, classes):
    image_id = image_id.strip('\n')
    xml_fn = data_path+'/Annotations/%s.xml'%(image_id)
    in_file = open(xml_fn, 'r', encoding="utf-8")
    tree=ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('size'):
        width = int(obj.find('width').text)
        height = int(obj.find('height').text)
        
    annotation_list = list()
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        cxyxy = (cls_id, int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text), width, height)
        #list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        annotation_list.append(cxyxy)
    return annotation_list

wd = os.getcwd()
## add read classes from class.txt

def create_annotation_file(data_path, classes_file, imageSet_path, annotation_path, isShow = False):
    print("Creating classes label... ")
    class_ids = open(classes_file)  ## classes.txt 使用者必須自行產生 class label
    classes_dict = dict()
    classes_list = list()
    for i, cls_id in enumerate(class_ids):
        cls_id = cls_id.strip('\n')
        #print(cls_id)
        classes_dict[cls_id] = i  ## create a dict  {class name: i}
        classes_list.append(cls_id)
    print("classes label: ", classes_list, "\n")
    print("classes dict: ", classes_dict)

    print("Creating annotation file... ")
    ## find all image ids
    image_list = findAllImagFiles(imageSet_path)
    #image_ids = list()
    #annotaions = data_path + "/Annotations"
    ## based on image_id, create labels file image_id.txt
    # format:  class label (0, 1, ..) center_x, center_y, width, height (normalized to image size)
    for fn in image_list:  ##有影像才要產生 label.txt
        if isShow:
            cvImg = cv2.imread(fn, -1)
        #row, col, channel = cvImg.shape  # height, width, 3
        #print(fn, " -- Shape of image: ", row, col, channel)
        __, filename, __ = split_filename_extension(fn)
        anno_list = read_annotation(data_path, filename, classes_list)
        #print(anno_list)
        label_fn = annotation_path + "/" + filename + ".txt"  ## default: image_id.txt
        f_label = open(label_fn, 'w')
        for anno in anno_list:
            cls, x1, y1, x2, y2, img_width, img_height = anno
            if isShow:
                cvImg  = draw_rectangle(cvImg, p1 =(x1, y1), p2 = (x2, y2), color = (0, 0, 255), lineWidth = 2)
            center_x, center_y, w, h = convert_xyxy_yolov5(x1, y1, x2, y2, img_width, img_height)
            line = str(cls) + " " + str(center_x) + " " + str(center_y) + " " + str(w) + " " + str(h) + "\n"
            f_label.write(line)
            print(cls, center_x, center_y, w, h)
        if isShow:
            cvImg = cv2.resize(cvImg, None, fx = 0.3, fy = 0.3)
            cv2.imshow(filename, cvImg)
            cv2.waitKey(0)
        f_label.close()
        #f_label = open(label_fn, 'w')
        #line = "class" + " " + "center_x" + " " + "center_y" 
        #f_label.write(line)
        #f_label.close()
        #image_ids.append(filename)
    #image_ids = open(imageSet_path)
    #print("image ids:", image_ids)
    #list_file = open(annotation_file, 'w')
    # ## based on image_id, create labels file image_id.txt
    # # format:  class label (0, 1, ..) center_x, center_y, width, height (normalized to image size)
    # for image_id in image_ids:
    #     #print(image_id)
    #     list_file.write(data_path+'/JPEGImages/%s%s'%(image_id.strip('\n'), file_extension))
    #     convert_annotation( image_id, list_file, classes_list)
    #     #convert_annotation( image_id, list_file)
    #     list_file.write('\n')
    # list_file.close()
    return

def convert_xyxy_yolov5(x1, y1, x2, y2, img_width, img_height):
    center_x = (x1 + x2) / 2 / img_width
    center_y = (y1 + y2) / 2 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return center_x, center_y, w, h


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type = str, default = "./ELAN_Data/train_data", help = 'Set up the training data path')
    opt = parser.parse_args()
    print(opt.data_path)
    classes_file = opt.data_path + "/classes.txt"
    imageSet_path = opt.data_path+'/JPEGImages'  ## read images from
    annotation_path = opt.data_path+'/labels'  ## path to write label file for yolov5
    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)
    create_annotation_file(opt.data_path, classes_file, imageSet_path, annotation_path, isShow=False)


## For E. Coli Project only -- Wise Precision Co.
## Purpose: convert current format into Yolov5 pytorch format
from utils.tien_utility import findAllImagFiles, split_filename_extension
from utils.Draw_Utility import draw_rectangle
import os
import cv2

def convert_to_yolov5_format(path = "./Cut_S0925087B,S10,090818", save_path = "Cut_S0925087B,S10,090818_label", isShow = False):
    image_list = findAllImagFiles(path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        print("Create saving directory")
    classes_dict={'STOMACH ANONYMIZE': 0}
    for filename in image_list:
        dir, fn, ext = split_filename_extension(filename)
        cvImg = cv2.imread(filename, -1)
        (height, width, _) = cvImg.shape
        ## read .txt file
        txt_fn = dir + "/" + fn + ".txt"
        print(txt_fn)
        label_fn = save_path + "/" + fn + ".txt"
        label_f = open(label_fn, "w")
        with open(txt_fn, "r") as f:
            lines = f.readlines()
            for line in lines:
                [cls, x, y, w, h] = line.strip().split("\t")  ## split with space
                print([cls, cls, x, y, w, h])
                ## format is x, y w, h ==> xc, yc, w, h
                x_center = int(x) + int(w) /2
                y_center = int(y) + int(h) /2
                box_width = int(w)
                box_height = int(h)
                # x_center = ( int(x1) + int(x2) ) /2
                # y_center = ( int(y1) + int(y2) ) /2
                # box_width = int(x2) - int(x1)
                # box_height = int(y2) - int(y1)
                ## normalized by w, h of image size
                s = str(classes_dict[cls]) + " " + str(x_center /width) + " "+ str(y_center /height) \
                 + " " + str(box_width /width) + " "+ str(box_height /height)
                label_f.write(s + "\n")
                ## Draw Rect ##
                cvImg = draw_rectangle(cvImg, (int(x), int(y)), (int(x)+ int(w), int(y) + int(h)), color = (0, 0, 255), lineWidth=1)
            if isShow:
                cv2.imshow(filename, cvImg)
                cv2.waitKey(0) 
        label_f.close()

if __name__ == "__main__":
    #convert_to_yolov5_format()
    #convert_to_yolov5_format(path = "./Cut_S0925428,S10,090820", save_path = "Cut_S0925428,S10,090820_label", isShow = False)
    #convert_to_yolov5_format(path = "./Cut_S0925449A,S10,090820", save_path = "Cut_S0925449A,S10,090820_label", isShow = False)
    #convert_to_yolov5_format(path = "./Cut_S0925449B,S10,090820", save_path = "Cut_S0925449B,S10,090820_label", isShow = False)
    convert_to_yolov5_format(path = "./Cut_S0925457,S10,090820", save_path = "Cut_S0925457,S10,090820_label", isShow = True)
    #convert_to_yolov5_format(path = "./Cut_S0925461,S10,090820", save_path = "Cut_S0925461,S10,090820_label", isShow = False)
    #convert_to_yolov5_format(path = "./Cut_S0925463,S10,090820", save_path = "Cut_S0925463,S10,090820_label", isShow = True)
from utils.yolov5_utils import yolov5_setup, yolov5_load_model, yolov5_predict
import cv2
from utils.tien_utility import findAllImagFiles, split_filename_extension
from utils.Draw_Utility import draw_rectangle

def read_bbox_list(filename): ## read wise format
    f = open(filename, "r")
    lines = f.readlines()
    bbox_list = list()
    for line in lines:
        data = line.split("\t")
        tag = data[0]
        #desc = data[1]
        bbox = (int(data[1]), int(data[2]), int(data[3]), int(data[4]))
        bbox_list.append(bbox)
    return bbox_list

if __name__ == '__main__':
    yolov5_setup()
    img_size = 800
    conf_thres = 0.02
    iou_thres =0.3
    model, device, half, class_names, colors = yolov5_load_model(model_path= "./weights/Best_E_Coli.pt", imgsz = img_size)
    #model.eval()
    path = "./inference/images/E_Coli" #zidane.jpg"
    img_list = findAllImagFiles(path)
    for fn in img_list:
        img0 = cv2.imread(fn, -1)  # BGR
        dir, filename, ext = split_filename_extension(fn)   
        class_names =["E. Coli"]     
        #cvGoodImage = cv2.resize(self.cvGoodImage, (0, 0), fx= self.scale, fy=self.scale)
        assert img0 is not None, 'Image Not Found ' + path
        display_img, det_list = yolov5_predict(model, img0, device, names=class_names, half=half, img_size = img_size, conf_thres = conf_thres, iou_thres = iou_thres, view_img= False, isRandomColor=False)
        bbox_list = read_bbox_list(filename = dir + "/" + filename + ".txt")
        for bbox in bbox_list:
            p1 = (bbox[0], bbox[1])
            p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            display_img = draw_rectangle(display_img, p1= p1, p2 = p2, lineWidth=2, color =(0, 255, 0)) ## draw green
        #print(det_list) ## [x1, y1, x2, y2, conf, class]
        out_fn = "out_"+ filename + ext #"./inference/output/E_Coli_Predict/out_"+ filename + ext
        #out_fn = out_fn.replace(",", "_")
        cv2.imwrite(out_fn, display_img)
        cv2.imshow("yolo v5 detection E Coli:", display_img)
        #cv2.waitKey(0)
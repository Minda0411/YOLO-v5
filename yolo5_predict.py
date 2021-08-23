from utils.yolov5_utils import yolov5_setup, yolov5_load_model, yolov5_predict
import cv2

if __name__ == '__main__':
    yolov5_setup()
    img_size = 640
    model, device, half, class_names, colors = yolov5_load_model(model_path= "./weights/clothing_best.pt", imgsz = img_size)
    #model.eval()
    path = "./inference/images/14.jpeg" #zidane.jpg"
    img0 = cv2.imread(path)  # BGR
    assert img0 is not None, 'Image Not Found ' + path
    display_img, det_list = yolov5_predict(model, img0, device, names=class_names, half=half, img_size = img_size, conf_thres = 0.2, iou_thres = 0.5, view_img= False)
    print(det_list) ## [x1, y1, x2, y2, conf, class]
    cv2.imshow("yolo v5 detection clothing:", display_img)
    cv2.waitKey(0)
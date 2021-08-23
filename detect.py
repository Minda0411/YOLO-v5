import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox
import numpy as np

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

# def yolov5_setup(out = "./output"):
#     # Initialize
#     set_logging()
#     #device = select_device()
#     if os.path.exists(out):
#         shutil.rmtree(out)  # delete output folder
#     os.makedirs(out)  # make new output folder
#     return

# def yolov5_load_model(model_path= "./weights/yolov5s.pt", imgsz =640):
#     t_start = time.perf_counter()
#     device = select_device('')
#     half = device.type != 'cpu'  # half precision only supported on CUDA
#     # Load model
#     model = attempt_load(model_path, map_location=device)  # load FP32 model
#     imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
#     if half:
#         model.half()  # to FP16
#     #print(model)
#     names = model.module.names if hasattr(model, 'module') else model.names
#     print(names)
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
#     ## predict one image
#     img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
#     _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
#     t_end = time.perf_counter()
#     print("Loading Yolo V5 Model spent(inlcuding first shot): ", round((t_end - t_start), 3), " sec.")
#     return model, device, half, names, colors  ## half (F16 not 32), get name, and prepare colors of rect

# def yolov5_predict(model, cvImg, conf_thres = 0.4, iou_thres = 0.5,view_img = True, save_img = False, save_txt = False):
#     img0 = cvImg.copy()
#     img = letterbox(cvImg, new_shape=img_size)[0]
#     #print(img.shape)
#     # Convert
#     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#     img = np.ascontiguousarray(img)
#     img = torch.from_numpy(img).to(device)
#     img = img.half() if half else img.float()  # uint8 to fp16/32
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)
#     # Inference
#     augment = False
#     t1 = time_synchronized()
#     pred = model(img, augment=augment)[0]
#     # Apply NMS
#     #conf_thres = 0.2
#     #iou_thres = 0.5
#     agnostic_nms = False
#     pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=agnostic_nms)
#     t2 = time_synchronized()
#     #print(pred)
#     #view_img = True ## show the result
#     save_txt = False ## do not save text
#     #save_img = False
#     for i, det in enumerate(pred):  # detections per image
#         # if webcam:  # batch_size >= 1
#         #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
#         # else:
#         p, s, im0 = path, '', img0 #im0s

#         #save_path = str(Path(out) / Path(p).name)
#         #txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
#         s += '%gx%g ' % img.shape[2:]  # print string
#         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#         det_list = list()
#         if det is not None and len(det):
#             # Rescale boxes from img_size to im0 size
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#             det_list.append(det) ## detection list: xyxy, conf, class
#             # Print results
#             for c in det[:, -1].unique():
#                 n = (det[:, -1] == c).sum()  # detections per class
#                 s += '%g %ss, ' % (n, names[int(c)])  # add to string

#             # # Write results
#             text_path = "./output/bbox_class"
#             for *xyxy, conf, cls in reversed(det):
#                 if save_txt:  # Write to file
#                     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                     with open(txt_path + '.txt', 'a') as f:
#                         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

#                 if save_img or view_img:  # Add bbox to image
#                      label = '%s %.2f' % (names[int(cls)], conf)
#                      plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

#         # Print time (inference + NMS)
#         print('%sDone. (%.3fs)' % (s, t2 - t1))

#         # Stream results
#         if view_img:
#             cv2.imshow(p, im0)
#             cv2.waitKey(0)
#             # if cv2.waitKey(1) == ord('q'):  # q to quit
#             #     raise StopIteration

#         # Save results (image with detections)
#         # if save_img:
#         #     if dataset.mode == 'images':
#         #         cv2.imwrite(save_path, im0)
#         #     else:
#         #         if vid_path != save_path:  # new video
#         #             vid_path = save_path
#         #             if isinstance(vid_writer, cv2.VideoWriter):
#         #                 vid_writer.release()  # release previous video writer

#         #             fourcc = 'mp4v'  # output video codec
#         #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#         #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
#         #         vid_writer.write(im0)
#     return im0, det_list ## result

if __name__ == '__main__':
    yolov5_setup()
    img_size = 640
    model, device, half, names, colors = yolov5_load_model(model_path= "./weights/clothing_best.pt", imgsz = img_size)
    #model.eval()
    path = "./inference/images/14.jpeg" #zidane.jpg"
    img0 = cv2.imread(path)  # BGR
    assert img0 is not None, 'Image Not Found ' + path
    display_img, det_list = yolov5_predict(model, img0, conf_thres = 0.3, iou_thres = 0.5)
    #print(img0.shape)
    # img = letterbox(img0, new_shape=img_size)[0]
    # #print(img.shape)
    # # Convert
    # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    # img = np.ascontiguousarray(img)
    # img = torch.from_numpy(img).to(device)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    # img /= 255.0  # 0 - 255 to 0.0 - 1.0
    # if img.ndimension() == 3:
    #     img = img.unsqueeze(0)
    # # Inference
    # augment = False
    # t1 = time_synchronized()
    # pred = model(img, augment=augment)[0]
    # # Apply NMS
    # conf_thres = 0.2
    # iou_thres = 0.5
    # agnostic_nms = False
    # pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=agnostic_nms)
    # t2 = time_synchronized()
    # print(pred)
    # view_img = True ## show the result
    # save_txt = False ## do not save text
    # save_img = False
    # for i, det in enumerate(pred):  # detections per image
    #     # if webcam:  # batch_size >= 1
    #     #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
    #     # else:
    #     p, s, im0 = path, '', img0 #im0s

    #     #save_path = str(Path(out) / Path(p).name)
    #     #txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
    #     s += '%gx%g ' % img.shape[2:]  # print string
    #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    #     if det is not None and len(det):
    #         # Rescale boxes from img_size to im0 size
    #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

    #         # Print results
    #         for c in det[:, -1].unique():
    #             n = (det[:, -1] == c).sum()  # detections per class
    #             s += '%g %ss, ' % (n, names[int(c)])  # add to string

    #         # # Write results
    #         for *xyxy, conf, cls in reversed(det):
    #             if save_txt:  # Write to file
    #                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    #                 with open(txt_path + '.txt', 'a') as f:
    #                     f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

    #             if save_img or view_img:  # Add bbox to image
    #                  label = '%s %.2f' % (names[int(cls)], conf)
    #                  plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

    #     # Print time (inference + NMS)
    #     print('%sDone. (%.3fs)' % (s, t2 - t1))

    #     # Stream results
    #     if view_img:
    #         cv2.imshow(p, im0)
    #         cv2.waitKey(0)
    #         # if cv2.waitKey(1) == ord('q'):  # q to quit
    #         #     raise StopIteration
    

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # opt = parser.parse_args()
    # print(opt)

    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
    #             detect()
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect()
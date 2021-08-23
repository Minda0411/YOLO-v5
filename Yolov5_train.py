from utils.yolov5_utils import Yolov5_Train_Params, yolov5_setup_train

if __name__ == '__main__':
    
    train_param = Yolov5_Train_Params()  ## initiate yolo param class
    train_param.epochs = 5
    train_param.batch_size = 8  ## depends on your GPU card
    train_param.img_size = [640, 640] ## define [training, testing] image sizes
    ## define structure of model: 必須先在檔案中修改 num_clasese, 且選定 model 大小: s, m, l, x 
    ## 可以先開啟 yolov5s.ymal, yolov5m.ymal,yolov5l.ymal, yolov5x.ymal，然後修改其中的class 數量
    train_param.cfg = "./models/yolov5s_clothing.yaml" 
    ##  define data source: 必須在該檔案中定義資料來源
    train_param.data = "./data/clothing.yaml"  ##
    train_param.weights = "./weights/yolov5s_clothing.pt" #clothing_best.pt" ## if pretrained wieght then ''
    train_param.adam = False   ## if data size is large, then you can set True，資料少時會使用 SGD
    train_param.resume = False
    train_param.nosave = True  ## true: 只存最後的model，過程中不儲存
    train_param.single_cls = False
    yolov5_setup_train(params = train_param)



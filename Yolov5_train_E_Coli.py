from utils.yolov5_utils import Yolov5_Train_Params, yolov5_setup_train, \
       copy_weight, convert_yolov5_onnx

if __name__ == '__main__':
    
    train_param = Yolov5_Train_Params()  ## initiate yolo param class
    train_param.epochs = 1
    train_param.batch_size = 4  ## depends on your GPU card
    train_param.img_size = [800, 800] ## define [training, testing] image sizes
    ## define structure of model: 必須先在檔案中修改 num_clasese, 且選定 model 大小: s, m, l, x 
    ## 可以先開啟 yolov5s.ymal, yolov5m.ymal,yolov5l.ymal, yolov5x.ymal，然後修改其中的class 數量
    train_param.cfg = "./models/yolov5s_E_Coli.yaml" 
    ##  define data source: 必須在該檔案中定義資料來源
    train_param.data = "./data/E_Coli.yaml"  ##
    train_param.weights = "./weights/best_E_Coli.pt"   #yolov5s_E_Coli.pt" ## if pretrained wieght then ''
    train_param.adam = False   ## if data size is large, then you can set True，資料少時會使用 SGD
    train_param.resume = False
    train_param.nosave = False  ## true: 只存最後的model，過程中不儲存
    train_param.notest = False  ## must have test.py
    train_param.name = "E_Coli_20201030"
    yolov5_setup_train(params = train_param)
    ## the following code: convert the newest model into onnx, copy that into ./weights/
    copy_weight(search_path="./runs", filename = "weights/best.pt", new_path = "./weights/best.pt")
    onnx_model = convert_yolov5_onnx(weights = './weights/best.pt', img_size = [800, 800], batch_size = 1)


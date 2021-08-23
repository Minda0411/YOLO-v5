echo set the root path
set root=C:\ProgramData\Anaconda3\
echo activate base envi
call %root%\Scripts\activate.bat %root%
echo activate the envi yolov5
call activate PyTorch_Yolo5
echo run your program
call python Yolov5_train_E_Coli.py.py
echo update the weight to newest
call python update_weight.py
echo So you can keep training your network, and make sure epoch = 50 or 100
call python Yolov5_train_E_Coli.py.py


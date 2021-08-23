echo set the root path
set root=C:\ProgramData\Anaconda3\
echo activate base envi
call %root%\Scripts\activate.bat %root%
echo activate the envi yolov5
call activate PyTorch_Yolo5
echo run your program
call python yolo5_predict_E_Coli.py


from ultralytics import YOLO
# Load a model
model = YOLO("./yolov8x.pt")  # build a new model from scratch

model.train(data="./mydata.yaml", mosaic=0.1,imgsz=512,amp=False,epochs=200,warmup_epochs=0,batch=64,device=[0,1,2,3])  # train the model


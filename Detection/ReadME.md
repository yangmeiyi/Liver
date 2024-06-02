# Environment

Python>=3.8、PyTorch>=1.8

```bash
pip install ultralytics
```

# Dataset Preparation

Apply the yolo format to train.

Mydata.yaml:

```yaml
train: /datasets/new/train/

val: /datasets/new/val/

names:  [0:lesion]
nc: 1
```

`train` and `val` are the paths of the training and validation sets, respectively. `nc` is the number of categories of target detection.

Example of  data storage is shown as follows,

📁 **lesions**

+ 📁 **train**
  + 📁 **images**
    +  🖼️**a.jpg**
  + 📁 **labels**
    + 📄**a.txt**

+ 📁 **val**
  + 📁 **images**
    +  🖼️**b.jpg**
  + 📁 **labels**
    + 📄**b.txt**

where **images** store the original image, **labels** store labels in .txt format. the name of .txt file and the name of the image correspond one to one. The label storage format is **cls x_center y_center width height**. For example:

```bash
0 0.205078125 0.5537109375 0.296875 0.443359375
0 0.4462890625 0.2705078125 0.115234375 0.072265625
```

# Model Training

**YOLO_V8**:
yolov8x.pt (Download: https://drive.google.com/file/d/1rYGl2g424MoNwc3LMM_sDJ8JwSDA2voi/view?usp=drive_link) 
```python
from ultralytics import YOLO
model = YOLO("detection/yolov8x.pt")
model.train(data="detection/mydata.yaml",mosaic=0.1,imgsz=512,amp=False,epochs=200,warmup_epochs=0,batch=64,device=[0,1,2,3])
```

# Model Testing
best.pt (Download: https://drive.google.com/file/d/1y7TzrwmhK6vb1BeXQFty0FqNKXcgVTBU/view?usp=drive_link)
```bash
python Test.py
```

**model_path** is the verified model location. **csv_file_path** is the csv location where images and disease categories are stored. The model validation results are stored under **detection/mAP-master/input**.

# Model Evaluation

```bash
python Calculate_index.py
```

**true_boxes_dir** is the directory where the real box is stored, and **pred_boxes_dir** is the directory where the prediction box is stored.

# The Result of Detection

| IOU:0.3 | F1   | Precision | Recall |
| ------- | ---- | --------- | ------ |
| HCC     | 94.0 | 93.7      | 94.3   |
| ICC     | 92.1 | 89.2      | 95.2   |
| MET     | 90.7 | 88.7      | 92.8   |
| FNH     | 91.5 | 91.3      | 91.7   |
| HEM     | 95.3 | 99.1      | 91.8   |
| CYST    | 94.9 | 97.7      | 92.3   |
| Overall | 92.8 | 92.0      | 93.7   |

# Model Usage Agreement
The model parameter (best.pt) is intended solely for academic research purposes. Commercial use is strictly prohibited, as is any attempt at reverse engineering or reconstructing the original training data.

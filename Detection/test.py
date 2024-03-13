from PIL import Image
from ultralytics import YOLO
import os
os.environ['CUDA_VISIBLE_DEVICES'] =  '0,1,2,3'
import sys
import contextlib
import io
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageDraw, ImageFont
import torch
from dataset import CustomDataset,CustomDataset1
import sys
import os
import csv
import re 
import shutil


def extract_identifier(file_name):
    file_name1=file_name
    file_name= file_name.rsplit('IMG', 2)[0]
    if '_PVP_' in file_name:
        return file_name.rsplit('_PVP_', 2)[0]
    elif '_AP_' in file_name:
        return file_name.rsplit('_AP_', 2)[0]
    elif '_CT_' in file_name:
        return file_name.rsplit('_CT_', 2)[0]
    else :
        return file_name
        
def find_category(csv_file_path, identifier):
    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            csv_identifier, category = row
            if csv_identifier == identifier:
                return category
    return 'normal'   
     
def clean_identifier(input_string):
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    cleaned_string = cleaned_string.lower()
    return cleaned_string  
    
def read_labels_to_tensor(file_path):
    def line_to_tensor(line):
        cls, xmid, ymid, w, h = map(float, line.split(' '))
        xmin = (xmid - w / 2) * 512
        ymin = (ymid - h / 2) * 512
        xmax = (xmid + w / 2) * 512
        ymax = (ymid + h / 2) * 512
        return torch.tensor([0, xmin, ymin, xmax, ymax])

    with open(file_path, 'r') as file:
        lines = file.readlines()
    if not lines:
        return torch.zeros(0, 5)
    tensors = [line_to_tensor(line.strip()) for line in lines]
    return torch.stack(tensors) 
             
def save_tensors_to_txt(labels, detections, file_name,ground_truth_dir,detection_results_dir):
    file_name, _ = os.path.splitext(file_name)
    ground_truth_file_path = os.path.join(ground_truth_dir, file_name + '.txt')
    detection_results_file_path = os.path.join(detection_results_dir, file_name + '.txt')
    
    if not os.path.exists(ground_truth_dir):
        os.mkdir(ground_truth_dir)
    if not os.path.exists(detection_results_dir):
        os.mkdir(detection_results_dir)
        
    with open(ground_truth_file_path, 'w') as file:
        for label in labels:
            label_str = ' '.join(map(str, label.tolist())) 
            file.write(label_str + '\n')

    with open(detection_results_file_path, 'w') as file:
        for detection in detections:
            detection_reordered = [detection[-1].item(), detection[-2].item(), *detection[:-2].tolist()]
            detection_str = ' '.join(map(str, detection_reordered))
            file.write(detection_str + '\n')        
        
if __name__ == '__main__':
  
    
    model = YOLO("detection/runs/detect/train124/weights/best.pt")
    
    val_dataset= CustomDataset("detection/datasets/new/val/")
    val_loader= DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False)
    
    csv_file_path = "detection/datasets/csv1.csv"
    
    for step,(img_names,images,label_paths) in enumerate(val_loader):
        images = images.to('cuda')
        
        results=model(images,conf=0.25,iou=0.5,device=[0,1 , 2,3])
        
        for (det,ct_data,label_path,img_name) in zip(results,images,label_paths,img_names):
        
          identifier = extract_identifier(img_name)
          identifier= clean_identifier(identifier)
          category = find_category(csv_file_path, identifier)
          
          label_data=read_labels_to_tensor(label_path)          
          detections=det.boxes.data 
          
          if not os.path.exists("detection/mAP-master/input"+category):
              os.mkdir("detection/mAP-master/input"+category)
          if not os.path.exists("detection/mAP-master/input"):
              os.mkdir("detection/mAP-master/input")
            
          g_path="detection/mAP-master/input"+category+"/ground-truth/"
          d_path="detection/mAP-master/input"+category+"/detection-results/"
          save_tensors_to_txt(label_data,detections,img_name,g_path,d_path)
          
          g_path="detection/mAP-master/input"+"/ground-truth/"
          d_path="detection/mAP-master/input"+"/detection-results/"
          save_tensors_to_txt(label_data,detections,img_name,g_path,d_path)
          
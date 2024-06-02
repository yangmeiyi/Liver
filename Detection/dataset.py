import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def read_txt_file(file_path):
  with open(file_path, 'r') as file:
      lines = file.readlines()
      lines = [line.strip() for line in lines]
  return lines
#label_paths=read_txt_file("/home/testPrev/Data_4TDISK/ymy/ZJJ/detection/datasets/new/change.txt")
#print(label_paths)


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.image_files = [file for file in os.listdir(self.image_dir) if  "mask" not in file  ]
        self.label_files = os.listdir(self.label_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.bmp', '.txt'))

        try:
            image = Image.open(img_path).convert("RGB")
    
        except Exception as e:
            print(f"error filename:{img_path}")
            print(f"{e}")
      #  labels = self.read_labels(label_path)
        transform = transforms.Compose([
            transforms.transforms.Resize(512),
            transforms.transforms.ToTensor(),
        ])
        return img_name, transform(image),label_path#,torch.tensor(labels, dtype=torch.float32)
    def read_labels(self, label_path):
        labels = []
        with open(label_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                label = line.strip().split(' ')
                labels.append(list(map(float, label)))
        return labels
def list_files_recursive_relative(directory):
    all_files = []

    for root, dirs, files in os.walk(directory):
        for name in files:
            relative_path = os.path.relpath(os.path.join(root, name), directory)
            all_files.append(relative_path)

    return all_files

class CustomDataset1(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir =root_dir
       # self.label_dir = os.path.join(root_dir, 'labels')
       
        self.image_files = [file for file in list_files_recursive_relative(self.image_dir) if "-"  in file]
      #  self.label_files = os.listdir(self.label_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
      #  label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        try:
            image = Image.open(img_path).convert("RGB")
    
        except Exception as e:
            print(f"error filename:{img_path}")
            print(f"{e}")
      #  labels = self.read_labels(label_path)
        transform = transforms.Compose([
            transforms.transforms.Resize((512,512)),
            transforms.transforms.ToTensor(),
        ])
        return img_name, transform(image),img_path#,torch.tensor(labels, dtype=torch.float32)
   



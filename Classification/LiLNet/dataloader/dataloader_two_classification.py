from torchvision import transforms
import torch
import os
import pandas as pd
from PIL import Image
import torch.utils.data as data
from torch.utils.data import Dataset
import random
from pathlib import Path
from utils.variables import *
from transforms.my_transforms import AddGaussianNoise, AddSaltPepperNoise
# from torchtoolbox.transform import Cutout
# from monai import transforms

train_high_compose = transforms.Compose([
    # Mycrop(),
    transforms.Resize((image_size + padding_size, image_size + padding_size)),
    transforms.RandomCrop(crop_size),
    # MyMask(),
    # transforms.CenterCrop((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.RandomInvert(p=0.5),
    # transforms.RandomAffine(8, translate=(.15, .15)),
    # transforms.RandomAffine(degrees=0, scale=(0.8,1), fillcolor=(0,0,0)),
    # transforms.ColorJitter(brightness=(0.5, 0.9)),
    AddGaussianNoise(mean=0.0, variance=1.0, amplitude=1.0, prob=0.2),
    AddSaltPepperNoise(density=0.2, prob=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
])

train_low_compose = transforms.Compose([
    # Mycrop(),
    transforms.Resize((image_size + padding_size, image_size + padding_size)),
    transforms.RandomCrop(crop_size),
    # MyMask(),
    # transforms.CenterCrop((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.RandomInvert(p=0.5),
    # transforms.ColorJitter(brightness=(0.5, 0.9)),
    # transforms.RandomAffine(8, translate=(.15, .15)),
    AddGaussianNoise(mean=0.0, variance=1.0, amplitude=1.0, prob=0.2),
    AddSaltPepperNoise(density=0.2, prob=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
])


test_compose = transforms.Compose([
    # Mycrop(),
    # Brightness_reduce(),
    transforms.Resize((image_size, image_size)),
    # MyMask(),
    # transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CancerSeT_CSV(Dataset):

    def __init__(self, root, type_):
        self.root = Path(root)
        self.type_ = type_
        if type_ == "train":
            self.csv = self.root / "BM_train.csv"   #
            self.transform_high = train_high_compose
            self.transform_low = train_low_compose
        elif type_ == "test":
            self.csv = self.root / "BM_test.cs"  #
            self.transform_high = test_compose
            self.transform_low = test_compose
        elif type_ == "val_hn":
            self.csv = self.root / "BM_validation_hn.csv"  #
            self.transform_high = test_compose
            self.transform_low = test_compose
        elif type_ == "val_cd":
            self.csv = self.root / "BM_validation_cd.csv"  #
            self.transform_high = test_compose
            self.transform_low = test_compose
        elif type_ == "val_gz":
            self.csv = self.root / "BM_validation_guizhou.csv"  #
            self.transform_high = test_compose
            self.transform_low = test_compose
        elif type_ == "val_les":
            self.csv = self.root / "BM_validation_leshan.csv"  #
            self.transform_high = test_compose
            self.transform_low = test_compose
        elif type_ == "val_sy":
            self.csv = self.root / "BM_validation_sanya.csv"  #
            self.transform_high = test_compose
            self.transform_low = test_compose
        elif type_ == "val_qz":
            self.csv = self.root / "validation_quzhou.csv"  #
            self.transform_high = test_compose
            self.transform_low = test_compose
        self.check_files(self.csv)
        try:
            self.csv = pd.read_csv(self.csv)
        except:
            self.csv = pd.read_csv(self.csv, encoding='gbk')
        self.csv = self.csv.dropna()
        self.csv['ID'] = self.csv['ID'].astype(str)


        print("loading good_bad")
        self.people_classfiy = self.csv.loc[:, 'Cancer'].map(
            lambda x: 0 if x == "good" else (1 if x == "bad" else 2))

        self.people_classfiy.index = self.csv['ID']
        self.people_classfiy = self.people_classfiy.to_dict()

        self.pic_0 = []
        self.pic_1 = []
        self.pic_files = []
        for p in self.people_classfiy:
            if type_ == 'train':
                pic_file = self.root / str(p)  # person
                pic_file_ct = pic_file / str("CT")
                pic_file_ap = pic_file / str("AP")
                pic_file_pvp = pic_file / str("PVP")
                if ct_ap_pvp:
                    pic_file = list(pic_file_ap.rglob('*.bmp')) + list(pic_file_ap.rglob("*.png")) + \
                               list(pic_file_pvp.rglob('*.bmp')) + list(pic_file_pvp.rglob("*.png")) + \
                               list(pic_file_ct.rglob('*.bmp')) + list(pic_file_ct.rglob("*.png"))
                if ap_pvp:
                    pic_file = list(pic_file_ap.rglob('*.bmp')) + list(pic_file_pvp.rglob("*.bmp")) + \
                               list(pic_file_ap.rglob('*.png')) + list(pic_file_pvp.rglob("*.png"))
                if ap:
                    pic_file = list(pic_file_ap.rglob('*.bmp')) + list(pic_file_ap.rglob("*.png"))
                if pvp:
                    pic_file = list(pic_file_pvp.rglob('*.png')) + list(pic_file_pvp.rglob("*.bmp"))

                if ct:
                    pic_file = list(pic_file_ct.rglob('*.bmp')) + list(pic_file_ct.rglob("*.png"))

            elif type_ == 'test':
                pic_file = self.root / str(p)  # person
                pic_file_ct = pic_file / str("CT")
                pic_file_ap = pic_file / str("AP")
                pic_file_pvp = pic_file / str("PVP")
                if ct_ap_pvp:
                    pic_file = list(pic_file_ap.rglob('*.bmp')) + list(pic_file_ap.rglob("*.png")) + \
                               list(pic_file_pvp.rglob('*.bmp')) + list(pic_file_pvp.rglob("*.png")) + \
                               list(pic_file_ct.rglob('*.bmp')) + list(pic_file_ct.rglob("*.png"))
                if ap_pvp:
                    pic_file = list(pic_file_ap.rglob('*.bmp')) + list(pic_file_pvp.rglob("*.bmp")) + \
                               list(pic_file_ap.rglob('*.png')) + list(pic_file_pvp.rglob("*.png"))
                if ap:
                    pic_file = list(pic_file_ap.rglob('*.bmp')) + list(pic_file_ap.rglob("*.png"))
                if pvp:
                    pic_file = list(pic_file_pvp.rglob('*.png')) + list(pic_file_pvp.rglob("*.bmp"))
                if ct:
                    pic_file = list(pic_file_ct.rglob('*.bmp')) + list(pic_file_ct.rglob("*.png"))

            else:
                pic_file = self.root / str(p)  # person
                pic_file_ct = pic_file / str("CT")
                pic_file_ap = pic_file / str("AP")
                pic_file_pvp = pic_file / str("PVP")
                if ct_ap_pvp:
                    pic_file = list(pic_file_ap.rglob('*.bmp')) + list(pic_file_ap.rglob("*.png")) + \
                               list(pic_file_pvp.rglob('*.bmp')) + list(pic_file_pvp.rglob("*.png")) + \
                               list(pic_file_ct.rglob('*.bmp')) + list(pic_file_ct.rglob("*.png"))
                if ap_pvp:
                    pic_file = list(pic_file_ap.rglob('*.bmp')) + list(pic_file_pvp.rglob("*.bmp")) + \
                               list(pic_file_ap.rglob('*.png')) + list(pic_file_pvp.rglob("*.png"))
                if ap:
                    pic_file = list(pic_file_ap.rglob('*.bmp')) + list(pic_file_ap.rglob("*.png"))
                if pvp:
                    pic_file = list(pic_file_pvp.rglob('*.png')) + list(pic_file_pvp.rglob("*.bmp"))
                if ct:
                    pic_file = list(pic_file_ct.rglob('*.bmp')) + list(pic_file_ct.rglob("*.png"))

            self.pic_files = []
            if self.people_classfiy[p] == 0:
                self.pic_0 += pic_file
            elif self.people_classfiy[p] == 1:
                self.pic_1 += pic_file

        print(len(self.pic_0), len(self.pic_1))
        self.pic_files = self.pic_0 + self.pic_1
        # if type_ == 'train':
        #     random.shuffle(self.pic_0)
        #     random.shuffle(self.pic_1)

        if type_ == 'train':
            self.files = []
            if len(self.pic_0) >= len(self.pic_1):
                ratio_1 = int(len(self.pic_0) // len(self.pic_1))
                distance = len(self.pic_0) - (ratio_1 * len(self.pic_1))
                self.pic_1 = (ratio_1) * self.pic_1 + self.pic_1[0: distance]
            else:
                ratio_0 = int(len(self.pic_1) // len(self.pic_0))
                distance = len(self.pic_1) - (ratio_0 * len(self.pic_0))
                self.pic_0 = (ratio_0) * self.pic_0 + self.pic_0[0: distance]

            self.pic_files = self.pic_0 + self.pic_1
            random.shuffle(self.pic_files)
            print("After copying", len(self.pic_0), len(self.pic_1))




        # dowm sampling

        # if type_ == 'train':
        #     self.files = []
        #     if len(self.pic_0) >= len(self.pic_1):
        #         self.pic_0 = self.pic_0[0: len(self.pic_1)]
        #     else:
        #         self.pic_1 = self.pic_1[0: len(self.pic_0)]
        #     self.pic_files = self.pic_0 + self.pic_1
        #     random.shuffle(self.pic_files)
        #     print("After sampling", len(self.pic_0), len(self.pic_1))

    def check_files(self, file):
        print("files:", file)
        assert Path(file).exists(), FileExistsError('{} inexistence'.format(str(file)))

    def __len__(self):
        return len(self.pic_files)

    def __getitem__(self, index):
        img_single = Image.open(str(self.pic_files[index]))
        img_single = img_single.convert("RGB")
        people = str(self.pic_files[index].parent.parent.name)
        cancer = self.csv.loc[self.csv['ID'] == str(people), "Cancer"].iloc[0]
        id = str(people)
        y = self.people_classfiy[str(people)]
        if y == 0:
            img_data = self.transform_low(img_single)
        else:
            img_data = self.transform_high(img_single)
        if img_data.shape[0] == 1:
            img_data = torch.cat([img_data] * 3, 0)
        rs = {
            "img": img_data,
            "label": torch.Tensor([y])[0],
            "id": id,
            "cancer": cancer,
            "image_path": str(self.pic_files[index])
        }
        return rs



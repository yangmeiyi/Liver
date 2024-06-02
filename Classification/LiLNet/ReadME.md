# Environment

+ Ubuntu (It's only tested on Ubuntu, so it may not work on Windows)
+ Python >= 3.9.7
+ PyTorch >= 1.12.1
+ torchvision >= 0.13.1
+ scikit-learn >= 1.1.3
+ spicy >= 1.9.3
+ bumpy >= 1.23.3

# Dataset Preparation

Example of  data storage is shown as follows,

📁 **Data**

+ 📁 **person_ID_1**
+ 📁 **person_ID_2**
  + 📁 **CT**
  + 📁 **AP**
  + 📁 **PVP**
    + 🖼️**IMG-0002-00001_0.bmp**
    + 🖼️**IMG-0002-00001_1.bmp**
    + $\ldots$
+ 📄**train.csv**

The format of **train.csv** is as follows:

| Person ID        | Class |
| ---------------- | ----- |
| 0000-00000-00001 | HCC   |
| 0000-00000-00002 | ICC   |
| 0000-00000-00003 | MET   |

The csv for training, testing, and validation is placed together with the patient folder.

# Data Processing and Loading

The./dataloader/hcc_icc_met_dataloader.py file processes data including:

```python
train_high_compose = transforms.Compose([
    transforms.Resize((image_size + padding_size, image_size + padding_size)),
    transforms.RandomCrop(crop_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(360),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

data loading:

```python
Liver_loader_train = CancerSeT_CSV(PATH, 'train')
Liver_loader_test = CancerSeT_CSV(PATH, 'test')

train_loader = torch.utils.data.DataLoader(Liver_loader_train, batch_size=args.train_batch, shuffle=True, drop_last=False) 
test_loader = torch.utils.data.DataLoader(Liver_loader_test, batch_size=args.test_batch, shuffle=False)
```

Adjustable hyperparameter in ./utils/variables.py:

```python
up_load = False #Whether to load the trained parameters
pretrained = False #Whether to load pre-training parameters
Need_train = False #Training or not
save_model = False #Whether to train Whether to save the trained model
dis = False  #Whether to train Whether to save the trained model Whether to choose distributed training

image_size = 224 #The size of the input images
padding_size = 0 #Whether enhance the data with padding image
crop_size = 224 #Data enhancement cropped the image to 224
batchsize_train = 256 #Batch size of the training set
batchsize_test = 256 #Batch size of the testing set
batchsize_val = 256 #Batch size of the validating set
gpus = "1,2,3,4,5,6,7" #The ID of the used GPUs
epochs = 200  #Training rounds

ct = False #Only CT images are loaded
ap = False #Only AP images are loaded
pvp = False #Only PVP images are loaded
ap_pvp = True #AP and PVP images are loaded
ct_ap_pvp = False #CT, AP and PVP images are loaded
```

The parameters of training our model:

| Parameters            | Value            |
| --------------------- | ---------------- |
| Image size            | 224              |
| Initial learning rate | 0.01             |
| Epoches               | 50               |
| Schedule              | [20,35]          |
| Weight decay          | 0.0001           |
| Optimizer             | optim.SGD        |
| Criterion             | CrossEntropyLoss |

Note: Training parameters on ImageNet need to be preloaded

# Model Training

Open the training Settings: Need_train = True

The default parameters of BM_train.py are used to train the benign and malignant classification.

The default parameters of Benign_train.py are used to train the benign (FNH, HEM,CYST) 3 classification.

The default parameters of Malignant_train.py are used to train malignant (HCC, ICC, MET) 3 classification.

Run the following script to train with files above:

```bash
python BM_train.py
python Benign_train.py
python Malignant_train.py
```

# Model Testing

Close the training Settings: Need_train = False

Open and load the trained model: up_load = True

The default parameters of BM_train.py are used to test the classification of benign and malignant. Run the following script to validate with files above:

```
# The trained model parameters is: BM.pth.tar
python BM_train.py
```

The default parameters of Benign_train.py are used to test the benign (FNH, HEM,CYST) 3 classification. Run the following script to validate with files above:

```
# The trained model parameters is: B.pth.tar
python Benign_train.py
```

The default parameters of Malignant_train.py are used to test the malignancy (HCC, ICC, MET) 3 classification. Run the following script to validate with files above:

```
# The trained model parameters is: M.pth.tar
python Malignant_train.py
```

# Model Usage Agreement

The model parameters (BM.pth.tar, B.pth.tar, and M.pth.tar, ) are intended solely for academic research purposes. Commercial use is strictly prohibited, as is any attempt at reverse engineering or reconstructing the original training data.

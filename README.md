# Focal Liver Lesions Diagnosis
<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/background.png" width="1000" height="300" /></div>


## Introduction
In this repository we release models from the paper "Advancing Focal Liver Lesion Diagnosis: Harnessing the Power of Deep Learning with Large-Scale Data and Multistage CT Imaging".
This study focused on developing a deep-learning AI-assisted system for liver clinical diagnosis. Previous studies have indicated that deep-learning algorithms outperformed health-care professionals with respect to some clinical outcomes.14,24,25 We propose a liver lesion diagnosis system based on deep learning, LiLNet, can automate the analysis of radiation imaging, rapidly screening and identifying suspicious regions for further examination by radiologists. With the utilization of multi-center and large sample data, this system offers a relatively comprehensive diagnostic approach. 

<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/frame.png" width="1000" height="850" /></div>




## Content
- 📁Classification
  - 📁LiLNet (Classification Diagnosis Model Folder for Deep Learning)
    - 📄process.py  (The detail of our method)
    - 📄resnet50_our.py(Our frame based on Resnet50)
    - 📄BM_train.py  (Training file for distinguishing between benign and malignant liver lesions)
    - 📄Benign_train.py(Training file for classifying benign liver lesions into three categories (fnh, hem, cyst))
    - 📄Malignant_train.py(Training file for classifying malignant liver lesions into three categories (hcc, icc, met))
  - 📁dataloader(Used for deep learning to load image data)
    - 📄dataloader_two_classification.py  (The dataloader for distinguishing between benign and malignant liver lesions)
    - 📄fnh_hem_cyst_dataloader.py  (The dataloader for distinguishing fnh, hem, and cyst lesions)
    - 📄hcc_icc_met_dataloader.py  (The dataloader for distinguishing hcc, icc, and met lesions)
  - 📁utils(Used for deep learning to load image data)
- 📁Detection
  - 📄train.py ()
  - 📄test.py ()
  - 📄calculate_index.py ()

- 📁Web testing data  (Data used to test the diagnostic system for liver lesions)
  - 📁Background 
  - 📁Lesions  (Randomly selected six types of lesion images(hcc, icc, met, fnh, hem, and cyst))
- 📄Readme.md (help)


## Code 

### Requirements
* Ubuntu (It's only tested on Ubuntu, so it may not work on Windows.)
* Python >= 3.9.7
* PyTorch >= 1.12.1
* torchvision >= 0.13.1
* scikit-learn >=1.1.3
* scipy >= 1.9.3
* numpy >=1.23.3


### Parameters
| Parameters | Value |
|-----------|:---------:|
| image size | 224 | 
| Initial learning rate | 0.01 | 
| Epoches | 50 | 
| Schedule | [20, 35] | 
| Weight decay | 0.0001 | 
| Optimizer | optim.SGD | 
| Criterion | CrossEntropyLoss | 


### Usage
```
cd ./Code/LiLNet/
python3 BM_train.py
```


### Category Metrics
* Accuracy
* Area under the receiver operating characteristic curve (AUC)
* Recall
* Precision
* F1 score

## Validation Scoring
 In this work, the model demonstrated excellent performance in identifying benign and malignant tumors, achieving an ACC of 94.7% and an AUC of 97.2% in the testing cohort. Additionally, for the differential diagnosis of HCC, ICC, and MET, the model achieved an ACC of 87.7% and an AUC of 95.6%. When extended to differentiate FNH, HEM, and CYST, the model achieved an ACC of 88.6% and an AUC of 95.9%. The system could be implemented in clinical practice for the diagnosis of liver lesions, particularly in regions where there is a shortage of radiologists. 


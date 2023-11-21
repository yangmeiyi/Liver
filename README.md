# Focal Liver Lesions Diagnosis
<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/background.png" width="1000" height="300" /></div>


## Introduction
In this repository we release models from the paper "CT-based Transformer model for non-invasively predicting the Fuhrman nuclear grade of clear cell renal cell carcinoma".
This study focused on developing a deep-learning AI-assisted system for liver clinical diagnosis. Previous studies have indicated that deep-learning algorithms outperformed health-care professionals with respect to some clinical outcomes.14,24,25 We propose a liver lesion diagnosis system based on deep learning, LiLNet, can automate the analysis of radiation imaging, rapidly screening and identifying suspicious regions for further examination by radiologists. With the utilization of multi-center and large sample data, this system offers a relatively comprehensive diagnostic approach. In relation to diagnostic accuracy and performance, LiLNet achieves excellent performance with an AUC of 97·6%, an ACC of 92·6%, a sensitivity of 93·7%, and a specificity of 92·7% for the classification of benign and malignant tumors. For the classification of malignant tumors, the model achieves an AUC of 96·5%, an ACC of 88·1%, an f1 of 88·5%, a recall of 87·1%, and a precision of 91·1%. Similarly, for the classification of benign tumors, the model achieves an AUC of 95·5%, an ACC of 89·9%, an f1 of 89·9%, a recall of 90·1%, and a precision of 90·2%. Notably, our model achieved a significant ACC improvement of 10% on the Henan external validation set, 3% on the Chengdu external validation set, 2% on the Leshan external validation set, and an impressive 20% on the Guizhou external validation set. These results highlight the effectiveness of our model in enhancing diagnostic performance across different external validation sets.

<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/workflow.png" width="1000" height="850" /></div>




## Content
- Code
  - LiLNet (Classification Diagnosis Model Folder for Deep Learning)
    - process.py  (The detail of our method)
    - resnet50_our.py(Our frame based on Resnet50)
    - BM_train.py  (Training file for distinguishing between benign and malignant liver lesions)
    - Benign_train.py(Training file for classifying benign liver lesions into three categories (fnh, hem, cyst))
    - Malignant_train.py(Training file for classifying malignant liver lesions into three categories (hcc, icc, met))
  - dataloader(Used for deep learning to load image data)
    - dataloader_two_classification.py  (The dataloader for distinguishing between benign and malignant liver lesions)
    - fnh_hem_cyst_dataloader.py  (The dataloader for distinguishing fnh, hem, and cyst lesions)
    - hcc_icc_met_dataloader.py  (The dataloader for distinguishing hcc, icc, and met lesions)
  - utils(Used for deep learning to load image data)
- Web testing data  (Data used to test the diagnostic system for liver lesions)
  - Background 
  - Lesions  (Randomly selected six types of lesion images(hcc, icc, met, fnh, hem, and cyst))
- Readme.md (help)


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






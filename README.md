# Focal Liver Lesions Diagnosis
<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/background.png" width="1000" height="300" /></div>


## Introduction
In this repository we release models from the paper "CT-based Transformer model for non-invasively predicting the Fuhrman nuclear grade of clear cell renal cell carcinoma".

<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/workflow" width="900" height="650" /></div>

This study focused on developing a deep-learning AI-assisted system for liver clinical diagnosis. Previous studies have indicated that deep-learning algorithms outperformed health-care professionals with respect to some clinical outcomes.14,24,25 We propose a liver lesion diagnosis system based on deep learning, LiLNet, can automate the analysis of radiation imaging, rapidly screening and identifying suspicious regions for further examination by radiologists. With the utilization of multi-center and large sample data, this system offers a relatively comprehensive diagnostic approach. In relation to diagnostic accuracy and performance, LiLNet achieves excellent performance with an AUC of 97·6%, an ACC of 92·6%, a sensitivity of 93·7%, and a specificity of 92·7% for the classification of benign and malignant tumors. For the classification of malignant tumors, the model achieves an AUC of 96·5%, an ACC of 88·1%, an f1 of 88·5%, a recall of 87·1%, and a precision of 91·1%. Similarly, for the classification of benign tumors, the model achieves an AUC of 95·5%, an ACC of 89·9%, an f1 of 89·9%, a recall of 90·1%, and a precision of 90·2%. Notably, our model achieved a significant ACC improvement of 10% on the Henan external validation set, 3% on the Chengdu external validation set, 2% on the Leshan external validation set, and an impressive 20% on the Guizhou external validation set. These results highlight the effectiveness of our model in enhancing diagnostic performance across different external validation sets.




## Content
> ├──Readme.md               // help  <br>
> ├──ccRCC              <br>  
> > ├──external             // external validation (from public dataset TCGA-KIRC)  <br>
> > ├── fpr_tpr_data        // tpr and fpr data, which can be used to draw ROC curves  <br>
> > ├── roc_curve           //  ROC curves  <br>
> > ├── image <br>
> > ├── pre_models          // models for integration and comparison  <br>
> > ├──transfer             // transfer learning  <br>
> > ├── TransResNet_model                      // TransResNet training  <br>
> > ├── ensemble     // model based on ensemble learning  <br>
> > ├── heat_map    // heat map  <br>
> > ├── hot_view    // CAM <br>


## Code 

### Requirements
* Ubuntu (It's only tested on Ubuntu, so it may not work on Windows.)
* Python >= 3.6.8
* PyTorch >= 1.0.1
* torchvision
* einops
* cuda

### Parameters
<div align=left><img src="https://github.com/yangmeiyi/ccRCC_project/blob/main/parameters.png" width="400" height="250" /></div>

### Usage

python3 ./TransResNet_model/train.py

### Getting Started
* For TransResNet (Table 2 in paper), please see TransResNet_model/train.py for detailed instructions.
* For Figure 2 in paper, plearse see rotate.py and data_enhancement.py.
* For Ensembel learning (Table 3 in paper), please see pre_models and ensemble.py.
* For Performance with Data Enhancement (Figure 4 in paper), please see heat_map.py
* For Transfer Learnig (Table 5 in paper), please see transfer.
* For ROC curves (Figure 5 in paper), please see roc_curve.
* For Figures 7 and 8 in paper,  please see hot_view.py.






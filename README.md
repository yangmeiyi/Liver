# Focal Liver Lesions Diagnosis
<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/background.png" width="1000" height="300" /></div>


## Introduction
In this repository we release models from the paper "CT-based Transformer model for non-invasively predicting the Fuhrman nuclear grade of clear cell renal cell carcinoma".
This study focused on developing a deep-learning AI-assisted system for liver clinical diagnosis. Previous studies have indicated that deep-learning algorithms outperformed health-care professionals with respect to some clinical outcomes.14,24,25 We propose a liver lesion diagnosis system based on deep learning, LiLNet, can automate the analysis of radiation imaging, rapidly screening and identifying suspicious regions for further examination by radiologists. With the utilization of multi-center and large sample data, this system offers a relatively comprehensive diagnostic approach. In relation to diagnostic accuracy and performance, LiLNet achieves excellent performance with an AUC of 97·6%, an ACC of 92·6%, a sensitivity of 93·7%, and a specificity of 92·7% for the classification of benign and malignant tumors. For the classification of malignant tumors, the model achieves an AUC of 96·5%, an ACC of 88·1%, an f1 of 88·5%, a recall of 87·1%, and a precision of 91·1%. Similarly, for the classification of benign tumors, the model achieves an AUC of 95·5%, an ACC of 89·9%, an f1 of 89·9%, a recall of 90·1%, and a precision of 90·2%. Notably, our model achieved a significant ACC improvement of 10% on the Henan external validation set, 3% on the Chengdu external validation set, 2% on the Leshan external validation set, and an impressive 20% on the Guizhou external validation set. These results highlight the effectiveness of our model in enhancing diagnostic performance across different external validation sets.

<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/workflow.png" width="1000" height="850" /></div>




## Content
- 一级标题(#一级标题)
  - [二级标题1](#二级标题1)
    - [三级标题1.1](#三级标题1.1)
    - [三级标题1.2](#三级标题1.2)
  - [二级标题2](#二级标题2)
- [另一个一级标题](#另一个一级标题)
  - [另一个二级标题1](#另一个二级标题1)
    - [另一个三级标题1.1](#另一个三级标题1.1)
    - [另一个三级标题1.2](#另一个三级标题1.2)
  - [另一个二级标题2](#另一个二级标题2)

 
- > ├──Readme.md               // help  <br>
> ├──Code              <br>  
> > ├──LiLNet             // external validation (from public dataset TCGA-KIRC)  <br>
> > > ├─BM_train.py             // external validation (from public dataset TCGA-KIRC)  <br>
> > > ├─Benign_train.py             // external validation (from public dataset TCGA-KIRC)  <br>
> > > ├─Malignant_train.py           // external validation (from public dataset TCGA-KIRC)  <br>
> > > ├─process.py             // external validation (from public dataset TCGA-KIRC)  <br>
> > > ├─resnet50.py             // external validation (from public dataset TCGA-KIRC)  <br>
> > ├──dataloader       // tpr and fpr data, which can be used to draw ROC curves  <br>
> > > ├─resnet50.py             // external validation (from public dataset TCGA-KIRC)  <br>
> > > ├─resnet50.py             // external validation (from public dataset TCGA-KIRC)  <br>
> > > ├─resnet50.py             // external validation (from public dataset TCGA-KIRC)  <br>
> > ├──utils        //  ROC curves  <br>
> ├──Web testing data   <br>
> > ├──Background        //  ROC curves  <br>
> > ├──Lesions        //  ROC curves  <br>


## Code 

### Requirements
* Ubuntu (It's only tested on Ubuntu, so it may not work on Windows.)
* Python >= 3.6.8
* PyTorch >= 1.0.1
* torchvision
* einops
* cuda

### Parameters
| Parameters | Value |
|-----------|:---------:|
| Row 1, Col 1 | Row 1, Col 2 | 
| Row 2, Col 1 | Row 2, Col 2 | 
| Row 3, Col 1 | Row 3, Col 2 | 



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






# Focal Liver Lesions Diagnosis
<div align=center><img src="https://github.com/yangmeiyi/Liver/blob/main/background.png" width="1000" height="350" /></div>


## Introduction
In this repository we release models from the paper "CT-based Transformer model for non-invasively predicting the Fuhrman nuclear grade of clear cell renal cell carcinoma".

<div align=center><img src="https://github.com/yangmeiyi/ccRCC_project/blob/main/network.jpg" width="900" height="650" /></div>

TransResNet was proposed to predict the low and high nuclear grade ccRCC, which combines the advantages of Transformer and ResNet. The proposed TransResNet model obtained an average accuracy of 87.1%, a sensitivity of 91.3%, a specificity of 85.3%, and an area under the curve (AUC) of 90.3% on the collected ccRCC data set containing 759 patients.




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






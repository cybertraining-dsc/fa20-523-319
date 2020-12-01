# Detect and classify pathologies in chest X-rays using PyTorch library

[![Check Report](https://github.com/cybertraining-dsc/fa20-523-319/workflows/Check%20Report/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-319/actions)
[![Status](https://github.com/cybertraining-dsc/fa20-523-319/workflows/Status/badge.svg)](https://github.com/cybertraining-dsc/fa20-523-319/actions)
Status: in progress


Rama Asuri, [fa20-523-319](https://github.com/cybertraining-dsc/fa20-523-319/), [Edit](https://github.com/cybertraining-dsc/fa20-523-319/blob/main/project/project.md)

{{% pageinfo %}}

## Abstract
Chest X-rays reveal many diseases. Early detection of disease often improves the survival chance for Patients. It is one of the important tools for Radiologists to detect and identify underlying health conditions. However, they are two major drawbacks. First, it takes time to analyze a radiograph. Second,  Radiologists make errors. Whether it is an error in diagnosis or delay in diagnosis, both outcomes result in a loss of life. With the technological advances in AI, Deep Learning models address these drawbacks. The  Deep Learning models analyze the X-rays like a Radiologist and accurately predict much better than the Radiologists. In our project, first, we develop a Deep Learning model and train our model to use the labels for Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion that corresponds to 5 different diseases, respectively. Second, we test our model's performance: how well our model predicts the diseases. Finally, we visualize our model's performance using the AUC-ROC curve.

Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** PyTorch, CheXpert


## 1. Introduction
Radiologists widely use chest X-Rays to identify and detect underlying conditions. However, analyzing Chest X-Rays takes too much time, and accurately diagnosing without errors requires considerable experience. On the one hand, if the analyzing process is expedited, it might result in misdiagnosis, but on the other hand, lack of experience means long analysis time and/or errors; even with the correct diagnosis, it might be too late to prescribe a treatment. Radiologists are up against time and experience. With the advancements in AI, Deep Learning can easily solve this problem quickly and efficiently.

Deep Learning methods are becoming very reliable at achieving expert-level performance using large labeled datasets. Deep learning is a technique to extract and transform data using multiple layers of neural networks. Each layer takes inputs from previous layers and incrementally refines it. An algorithm is used to train these layers to minimize errors and improve these layers' overall accuracy [^10]. It enables the network to learn to perform a specified task and gain an expert level performance by training on large datasets. The scope of this project is to identify and detect the following 5 pathologies using an image classification algorithm: Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion.  We use the CheXpert dataset, which consists of Chest X-rays. CheXpert dataset contains 224,316 chest Radiographs of 65,240 patients. The dataset has 14 observations in radiology reports and captures uncertainties inherent in radiograph interpretation using uncertainty labels. Our focus is on 5 observations (Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion). We impute uncertainty labels with randomly selected Boolean values.  Our Deep Learning models are developed using the PyTorch library, enabling fast, flexible experimentation and efficient production through a user-friendly front-end, distributed training, and ecosystem of tools and libraries[^7]. It was primarily developed by Facebook's AI Research lab (FAIR) and used for Computer Vision and NLP applications. PyTorch supports Python and C++ interfaces. There are popular Deep Learning applications built using PyTorch, including Tesla Autopilot, Uber's Pyro, etc. [^4].

In this report, we will cover the following:
* Overview of PyTorch library and DenseNet
* Overview of CheXpert dataset
* Basics of AUC-ROC Curve
* Chest X-Rays - Multi-Image Classification Using Deep Learning Model
* Conclusion
* Acknowledgements
* References

## 2. Overview Of PyTorch Library And DenseNet

The PyTorch library is based on Python and is used for developing Python deep learning models. Many of the early adopters of the PyTorch are from the research community. It grew into one of the most popular libraries for deep learning projects. PyTorch provides great insight into Deep Learning. PyTorch is widely used in real-world applications. PyTorch makes an excellent choice for introducing deep learning because of clear syntax, streamlined API, and easy debugging. PyTorch provides a core data structure, the tensor, a multidimensional array similar to NumPy arrays. It performs accelerated mathematical operations on dedicated hardware, making it convenient to design neural network architectures and train them on individual machines or parallel computing resources [^2].

We use a pre-trained DenseNet model, which classifies the images. DenseNet is efficient on image classification benchmarks as compared to ResNet [^11].

Figure 1 shows the DenseNet architecture.

![Figure 1](https://github.com/cybertraining-dsc/fa20-523-319/raw/main/project/images/densetnet.png)

**Figure 1:** DenseNet Architecture [^12]

(Note- adding more about DenseNet)

## 3. Overview of CheXpert Dataset

CheXpert is a large public dataset. It contains an interpreted chest radiograph consisting of 224,316 chest radiographs
of 65,240 patients labeled for the presence of 14 observations as positive, negative, or uncertain [^3].

Figure 2 shows the CheXpert labels and the Probability


![Figure 2](https://github.com/cybertraining-dsc/fa20-523-319/raw/main/project/images/chest_disease.png)

**Figure 2:** Probability of different observations



### 3.1 Data Collection

CheXpert's dataset is a collection of chest radiographic studies from Stanford Hospital, performed between October
2002 and July 2017 in both inpatient and outpatient centers, along with their associated radiology reports.
From these, created a sampled set of 1000 reports for manual review by a board certified radiologist to determine
feasibility for extraction of observations. The final set consist of 14 observations based on the prevalence in the
reports and clinical relevance, conforming to the Fleischner Society’s recommended glossary. *Pneumonia*, despite
being a clinical diagnosis, was included as a label in order to represent the images that suggested primary
infection as the diagnosis. The *No Finding* observation was intended to capture the absence of all pathologies [^3].

### 3.2 Data Labelling

Labels were developed using an automated rule-based labeler to extract observations from the free text radiology
reports to be used as structured labels for the images [^3].

### 3.3 Label Extraction

The labeler extracts mentions from a list of observations from the Impression section of radiology reports, which
summarizes the key findings in the radiographic study. A large list of phrases was manually curated by multiple
board-certified radiologists to match various ways observations are mentioned in the reports [^3].

### 3.4 Label Classification

After extracting mentions of observations, CheXpert Labler classify them as negative , uncertain or positive.
The ‘uncertain’ label can capture both the uncertainty of a radiologist in the diagnosis as well as ambiguity
inherent in the report. The mention classification stage is a 3-phase pipeline consisting of pre-negation uncertainty,
negation, and post-negation uncertainty. Each phase consists of rules which are matched against the mention; if a
match is found, then the mention is classified accordingly . If a mention is not matched in any of the phases, it is
classified as positive [^3].

### 3.5 Label Aggregation

CheXpert use the classification for each mention of observations to arrive at a final label for 14 observations that
consist of 12 pathologies as well as the *Support Devices* and *No Finding* observations. Observations with at least
one mention that is positively classified in the report is assigned a positive (1) label. An observation is assigned
an uncertain (u) label if it has no positively classified mentions and at least one uncertain mention, and a negative
label if there is at least one negatively classified mention. Assign (blank) if there is no mention of an observation.
The *No Finding* observation is assigned a positive label (1) if there is no pathology classified as positive
or uncertain [^3].

## 4. Basics Of AUC-ROC Curve

![Figure 3](https://github.com/cybertraining-dsc/fa20-523-319/raw/main/project/images/confusion_matrix.png)

**Figure 3:** Confusion Matrix

## 5. Chest X-Rays - Multi-Image Classification Using Deep Learning Model

Our Deep Learning model loads and process the raw data files and implement a Python class to represent data by
converting it into a format usable by PyTorch. We then, visualize the training and validation data.

Our approach to detecting pathologies will have 5 steps.

* Load raw Chest X-rays and put raw data into PyTorch
* Identify the one of the five pathologies
* Group the Pathologies
* Classify the pathology
* Diagnose the patient

AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, Higher the AUC, better the model is at distinguishing between patients with disease and no disease.


AUCROC's Sensitivity and Specificity are inversely proportional to each other. So when we increase Sensitivity, Specificity decreases and vice versa.
When we decrease the threshold, we get more positive values thus it increases the sensitivity and decreasing the specificity.
Similarly, when we increase the threshold, we get more negative values thus we get higher specificity and lower sensitivity [^5].

## 6. Conclusion


Model was able to predict False Positives. Below is AUCROC table.
![Figure 4](https://github.com/cybertraining-dsc/fa20-523-319/raw/main/project/images/roc.png)

**Figure 4:** AUC - ROC Curve

## 7. Acknowledgements

The author would like to thank Dr. Gregor Von Laszewski, Dr. Geoffrey Fox, and the associate instructors for providing continuous guidance and feedback for this final project.
## 8. Project plan

* October 26, 2020
  * Test train and validate functionality on PyTorch Dataset
  * Update Project.md with project plan
*  November 02, 2020
   * Test train and validate functionality on manual uploaded CheXpert Dataset
   * Update project.md with specific details about Deep learning models
* November 09, 2020
  * Test train and validate functionality on downloaded CheXpert Dataset using "wget"
  * Update project.md with details about train and validation data set
  * Capture improvements to loss function
* November 16, 2020
  * Self review - code and project.md
* December 02, 2020
  * Review with TA/Professor - code and project.md
* December 07, 2020
  * Final submission - code and project.md

## 9. References

[^1]: Chest X-ray Dataset <https://stanfordmlgroup.github.io/competitions/chexpert/>

[^2]: Introduction to PyTorch and documentation  <https://pytorch.org/deep-learning-with-pytorch>

[^3]: Whitepaper - CheXpert Dataset and Labelling  <https://arxiv.org/pdf/1901.07031.pdf>

[^4]: Overview of PyTorch Library <https://en.wikipedia.org/wiki/PyTorch>

[^5]: PyTorch Deep Learning Model for CheXpert Dataset <https://www.kaggle.com/hmchuong/chexpert-pytorch-densenet121>

[^6]: Overview of AUC-ROC Curve in Machine Learning <https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/>

[^7]: An open source machine learning framework that accelerates the path from research prototyping to production deployment <https://pytorch.org/>

[^8]: PyTorch Computer Vision Cookbook <https://www.packtpub.com/product/pytorch-computer-vision-cookbook/9781838644833>

[^9]: PyTorch 1.x Reinforcement Learning Cookbook <https://www.packtpub.com/product/pytorch-1-x-reinforcement-learning-cookbook/9781838551964>

[^10]: Howard, Jeremy; Gugger, Sylvain. Deep Learning for Coders with fastai and PyTorch . O'Reilly Media. Kindle Edition <https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527/ref=sr_1_5?dchild=1&keywords=pytorch&qid=1606487426&sr=8-5>

[^11]: The efficiency of densenet121 <https://medium.com/@smallfishbigsea/densenet-2b0889854a92>

[^12]: Densetnet architecture <https://miro.medium.com/max/1050/1*znemMaROmOd1CzMJlcI0aA.png>

[^13]: Overview of DenseNet <https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a>

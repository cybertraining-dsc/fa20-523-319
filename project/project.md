# Detect and classify pathologies in chest X-rays using PyTorch library

- [ ] please fix refernces

Rama Asuri, [fa20-523-319](https://github.com/cybertraining-dsc/fa20-523-319/), [Edit](https://github.com/cybertraining-dsc/fa20-523-319/blob/master/project/project.md)

{{% pageinfo %}}

## Abstract

Deep learning gains expert level performance by training on large datasets. Our project uses chest x-rays dataset 
called CheXpert and apply deep learning to detect and diagnose a condition based on image data. CheXpert dataset 
contain 224,316 chest radiographs of 65,240 patients. There are 14 observations in a radiology report and labels are 
mapped to these 14 observations. We train our models to use the labels that corresponds to different diseases. Our 
deep learning models are based on PyTorch library.


Contents

{{< table_of_contents >}}

{{% /pageinfo %}}

**Keywords:** PyTorch, CheXpert


## 1. Introduction

Deep learning methods are becoming very reliable at achieving expert level performance using large labeled datasets. 
CheXpert is a large dataset that contains 224,316 chest radiographs of 65,240 patients. This dataset contains
14 observations in radiology reports and also capturing uncertainties inherent in radiograph interpretation using
uncertainty labels. There is also a validation set of 200 chest radiographic studies which were manually annotated 
by 3 board-certified radiologists. Deep learning model should detect different pathologies. CheXpert is a public 
dataset [^3] and has a strong radiologist-annotated ground truth and expert scores against which we can compare 
the model [^1].

We will use PyTorch library which is an open source machine learning library based on the Torch library [^2]. It is 
used for applications such as computer vision and natural language processing which is primarily developed by 
Facebook's AI Research lab (FAIR). It is free and open-source software released under the Modified BSD license.
Although the Python interface is more polished and the primary focus of development, PyTorch also has a C++ interface. 
There are number of Deep Learning applications that are built on top of PyTorch, including Tesla Autopilot, Uber's 
Pyro etc [^4].

## 2. PyTorch Library
PyTorch library is based on Python and used for developing Python deep learning models. Many of the early adopters 
of the PyTorch are from research community. It grew into one of the most popular library for deep learning projects. 
PyTorch provides great insight into Deep Learning. PyTorch is widely used in real-world applications. PyTorch makes an
excellent choic for introducing deep learning because clear syntax, streamlined API and easy debugging. PyTorch 
provides a core data structure, the tensor, which is a multidimensional array similar to NumPy arrays. It perform 
accelerated mathematical operations on dedicated hardware, which makes it convenient to design neural network 
architectures and train them on individual machines or parallel computing resources [^2].

## 3. Dataset
CheXpert is a large public dataset for chest radiograph interpretation, consisting of 224,316 chest radiographs
of 65,240 patients labeled for the presence of 14 observations as positive, negative, or uncertain [^3].


![Probability of different observations](https://github.com/cybertraining-dsc/fa20-523-319/raw/master/project/images/chest_disease.png)

**Figure 1:** Probability of different observations



### 3.1 Data Collection
CheXpert's dataset is a collection of chest radiographic studies from Stanford Hospital, performed between October 
2002 and July 2017 in both inpatient and outpatient centers, along with their associated radiology reports. 
From these, created a sampled set of 1000 reports for manual review by a boardcertified radiologist to determine
feasibility for extraction of observations. The final set consist of 14 observations based on the prevalence in the 
reports and clinical relevance, conforming to the Fleischner Society’s recommended glossary. “Pneumonia”, despite 
being a clinical diagnosis, was included as a label in order to represent the images that suggested primary 
infection as the diagnosis. The “No Finding” observation was intended to capture the absence of all pathologies [^3].

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
consist of 12 pathologies as well as the “Support Devices” and “No Finding” observations. Observations with at least 
one mention that is positively classified in the report is assigned a positive (1) label. An observation is assigned 
an uncertain (u) label if it has no positively classified mentions and at least one uncertain mention, and a negative
label if there is at least one negatively classified mention. Assign (blank) if there is no mention of an observation.
The “No Finding” observation is assigned a positive label (1) if there is no pathology classified as positive 
or uncertain [^3].

## 4. Use case and deep learning model

Our Deep Learning model loads and process the raw data files and implement a Python class to represent data by 
converting it into a format usable by PyTorch. We then, visualize the training and validation data. 

Our approach to detecting pathologies will have 5 steps.

* Load raw Chest X-rays and put raw data into PyTorch
* Identify the one of the five pathologies
* Group the Pathologies
* Classify the pathology
* Diagnose the patient 

## 5. Inference

This section will be addressed upon project completion.

## 6. Conclusion

This section will be addressed upon project completion.

## 7. Acknowledgements

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
  * Final review of code and project.md

## 9. References

[^1]: Dataset <https://stanfordmlgroup.github.io/competitions/chexpert/>

[^2]: DETAILS MISSING  <https://pytorch.org/deep-learning-with-pytorch>

[^3]: details missing <https://arxiv.org/pdf/1901.07031.pdf>

[^4]: details missing <https://en.wikipedia.org/wiki/PyTorch>

# Detect and classify pathologies in chest X-rays using PyTorch library

| Rama Asuri
| asurirk@gmail.com, rasuri@indiana.edu
| Indiana University
| hid: fa20-523-319


---

Keywords: PyTorch, CheXpert

---

## Abstract

Deep learning gains expert level performance by training on large datasets.  Our project uses chest x-rays dataset 
called CheXpert and apply deep  learning to detect and diagnose a condition based on image data. CheXpert dataset 
contain 224,316 chest radiographs of 65,240 patients. There are 14 observations in a radiology report and labels are 
mapped to these 14 observations. We train our models to use the labels that corresponds to different diseases. Our 
deep learning models are based on PyTorch library. 


## Introduction

Deep learning methods are becoming very reliable at achieving expert level performance using large labeled datasets. 
CheXpert is a large dataset that contains 224,316 chest radiographs of 65,240 patients. This dataset contains
14 observations in radiology reports and also capturing uncertainties inherent in radiograph interpretation using
uncertainty labels. There is also a validation set of 200 chest radiographic studies which were manually annotated 
by 3 board-certified radiologists. Deep learning model should detect different pathologies. CheXpert is a public 
dataset [[3]] and has a strong radiologist-annotated ground truth and expert scores against which we can compare 
the model [[1]].

We will use PyTorch library which is an open source machine learning library based on the Torch library [[2]]. It is 
used for applications such as computer vision and natural language processing which is primarily developed by 
Facebook's AI Research lab (FAIR). It is free and open-source software released under the Modified BSD license.
Although the Python interface is more polished and the primary focus of development, PyTorch also has a C++ interface. 
There are number of Deep Learning applications that are built on top of PyTorch, including Tesla Autopilot, Uber's 
Pyro etc [[4]].

## Requirements

> Work in progress

we develop the Deep Learning Software using PyTorch library to perform the following.
1. Load raw Chest X-rays and put raw data into PyTorch
2. Identify the one of the five pathologies
3. Group the Pathologies
4. Classify the pathology
5. Diagnose the patient 

## Design
> Missing design
## Architecture

## Dataset

## Implementation

## Benchmark

> TBD - Other deep learning models data is available online for benchmarking.

## Conclusion

## Acknowledgement
## Project plan
> Missing - Adding detailed project plan

This project has following deliverables
1. Design documents
2. Final Project report
3. Working Software demo

## Footnotes
[1]: https://stanfordmlgroup.github.io/competitions/chexpert/
[2]: https://pytorch.org/deep-learning-with-pytorch
[3]: https://arxiv.org/pdf/1901.07031.pdf
[4]: https://en.wikipedia.org/wiki/PyTorch

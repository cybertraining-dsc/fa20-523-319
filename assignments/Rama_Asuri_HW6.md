# Rama Asuri - Homework 6 AI in Health and Medicine

AI in health and medicine is achieving expert-level performance. In this paper, we examine two different examples where AI could detect and predict the malign tumors [1].

In the first example, we will cover Cervical cancer. Cervical cancer kills more women in India than in any other country. It is a preventable disease that kills 67000 women in India. Screening and detection can help reduce the number of deaths, but the challenge is the testing process, which takes enormous time. SRL diagnostics partnered with Microsoft to co-create an AI Network of Pathology to reduce cytopathologists and histopathologists' burden. Cytopathologists at SRL Diagnostics manually marked their observations. These observations were used as training data
for Cervical Cancer Image Detection. However, there was a different challenge, the way cytopathologists examine different elements are unique even though they all have come to the same conclusion. This was because these experts may approach a problem from a different direction. The Manish Gupta, Principal Applied Researcher at Microsoft Azure Global Engineering, who worked closely with the team at SRL Diagnostics, said the idea was to create an AI algorithm that could identify areas that everybody was looking at and “create a consensus on the areas assessed.” Cytopathologists across multiple labs and locations annotated thousands of tile images of a cervical smear. They created discordant and concordant notes on each sample image. “The images for which annotations were found to be discordant — that is if they were viewed differently by three team members — were sent to senior cytopathologists for final analysis”. SRL Diagnostics has started an internal preview to use Cervical Cancer Image Detection API. The Cervical Cancer Image Detection API, which runs on Microsoft’s Azure, can quickly screen liquid-based cytology slide images to detect cervical cancer in the early stages and return insights to pathologists in labs. The AI model can now differentiate between normal and abnormal smear slides with accuracy and is currently under validation in labs. It can also classify smear slides based on the seven-subtypes of cervical cytopathological scale [1].

The second example is about detecting lung cancer. The survival rate is really high if lung cancer is detected during
the early stages. Nevertheless, the problem is that it is difficult to do it manually when there millions of 3D X-rays.
Reviewing scans is done by a highly trained specialist, and a majority of the reviews result in no detection.
Moreover, this is also monotonous work, which might lead to errors by the reviewers. The LUNA Grand Challenge is an
open dataset with high-quality labels of patient CT scans. The gLUNA Grand Challenge encourages improvements in
nodule detection by making it easy for teams to compete for high positions on the leader board. A project team can
test the efficacy of their detection methods against standardized criteria [3].

## Emerging AI Applications in Oncology
Note - updating this section shortly
### Improving Cancer Screening and Diagnosis
The MRI-guided biopsy was developed by National Cancer Institute (NCI) researchers works without a need for clinics
because of the AI tool [4].
### Aiding the Genomic Characterization of Tumors
Identifying mutations using noninvasive techniques is a particularly challenging problem when it comes to brain
tumors [4]. NCI and other partners concluded that AI could help identify gene mutations in innovative ways.
### Accelerating Drug Discovery
Using AI, scientists were able to target mutations in the KRAS gene, one of the most frequently mutated
oncogenes in tumors [4].
### Improving Cancer Surveillance
AI will help predicting treatment response, recurrence and survival based on the detection from the images.

## References

[1]: https://drive.google.com/file/d/1qqd-P0zvQY8MEZDdzuxS-DhObiMSyKRO/view?usp=sharing
[2]: https://techcrunch.com/2019/11/09/microsoft-srl-diagnostics-cervical-cancer/
[3]: https://pytorch.org/deep-learning-with-pytorch
[4]: https://www.cancer.gov/research/areas/diagnosis/artificial-intelligence

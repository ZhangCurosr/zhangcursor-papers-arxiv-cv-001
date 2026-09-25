# DOMAIN RECENTERING AND CONFIDENCE-WEIGHTED PRIOR CALIBRATION FOR VISION-LANGUAGE MODELS

Youngeun Seol<sup>⋆</sup> Jimin Shin<sup>⋆</sup> Heeseo Yoon<sup>⋆</sup> Uiwon Hwang<sup>†</sup>

Department of Computer Science and Engineering, Ewha Womans University

## ABSTRACT

Vision-language models such as CLIP achieve strong zero-shot classification, yet under distribution shift, visual embeddings drift from fixed text embeddings. Training-free calibration avoids the per-sample optimization of prompt learning, but prior feature calibration gives each image the full bias of one hard cluster. We propose Domain Recentering with Confidence Calibration (DRC), a training-free method adapting CLIP from a set of unlabeled target images. DRC fits a Gaussian mixture once and subtracts from each embedding a posterior-weighted average of component means. It then removes residual class preference with a log-prior correction, estimating the prior from confidence-weighted predictions. Among compared methods, DRC achieves the highest average accuracy on cross-domain datasets, exceeding zero-shot CLIP by 4.13 and 5.07 points with ViT-B/16 and ResNet-50, with gains over CLIP also holding under ImageNet distribution shifts.

Index Terms— Vision-language models, domain recentering, confidence-weighted calibration

## 1. INTRODUCTION

Vision-language models have emerged as powerful foundation models for visual recognition. CLIP [1] learns a multimodal embedding space for images and text from natural language supervision at scale, enabling zero-shot classification by aligning an image’s embedding with the text embeddings of class descriptions. However, when target images differ from the pretraining distribution, their visual embeddings can become misaligned with the fixed text embeddings, degrading classification accuracy.

Recent works have adapted CLIP under distribution shift along two broad directions. One line performs prompt learning with few-shot labeled data [2, 3] or test-time entropy minimization [4, 5], requiring labels or per-image backpropagation. Another line calibrates features without training, using unlabeled target data. UMFC [6] corrects visual encoder bias by subtracting from each feature the mean of its hard-assigned cluster. However, a feature near a cluster boundary then receives one cluster’s full mean, so a small change in the feature can flip its correction. Correcting the image features alone also leaves the text classifier unchanged, so any preference it holds for certain classes persists.

We propose Domain Recentering with Confidence Calibration (DRC), which models the structure of unlabeled CLIP visual features using a Gaussian mixture model. Rather than assigning each image to a single cluster, DRC uses mixturecomponent posterior probabilities to construct a soft, imagespecific bias vector as a weighted combination of component means, so the correction changes continuously as the feature moves between components. This vector is subtracted from the image feature to suppress domain-related variation while preserving class-discriminative content. DRC further corrects residual class preferences through confidence-weighted prior calibration, without any backpropagation or parameter updates. Among the compared methods, DRC achieves the highest average accuracy on ten cross-domain datasets with both ResNet-50 and ViT-B/16, and it also improves the average accuracy of zero-shot CLIP over four ImageNet distribution shifts.

## 2. RELATED WORK

Prompt learning adapts CLIP [1] to downstream data by tuning its textual prompts. CoOp [2] and CoCoOp [3] learn these prompts from labeled few-shot data. Test-time adaptation instead uses unlabeled test samples to improve robustness under distribution shifts. TPT [4] optimizes prompts for each test sample, and DiffTPT [5] extends TPT with diffusion-based augmentation. Both repeat backpropagation for every test sample, which makes inference costly and motivates trainingfree adaptation that leaves the pretrained model unchanged.

Training-free methods instead correct the systematic biases that zero-shot vision-language models exhibit in representations and predictions [6, 7]. UMFC [6] subtracts each hard-assigned cluster’s mean, giving all images within a cluster the same correction. Label-free logit adjustment [8] offsets label bias in predictions. Our method instead assigns each image its own bias by weighting Gaussian mixture component means with the image’s soft assignment. It then estimates the class prior from the recentered predictions with confidence weighting, so that the prior targets the bias left after feature correction.

## 3. METHOD

## 3.1. Background

CLIP. CLIP [1] maps images and text into a shared embedding space via a visual encoder $E _ { v }$ and a text encoder $E _ { t } .$ . For zero-shot classification, each class name $y _ { c }$ is inserted into prompt templates and encoded by $E _ { t }$ , and the averaged embedding serves as the classifier weight $\mathbf { t } _ { c } .$ Given an image $x _ { i } ,$ its visual feature $\mathbf { v } _ { i } = E _ { v } ( x _ { i } )$ is compared to each $\mathbf { t } _ { c }$ via scaled cosine similarity

$$
\ell _ { i c } = \exp ( \tau ) \left( \frac { \mathbf { v } _ { i } } { \| \mathbf { v } _ { i } \| _ { 2 } } \right) ^ { \top } \frac { \mathbf { t } _ { c } } { \| \mathbf { t } _ { c } \| _ { 2 } } ,\tag{1}
$$

where $\exp ( \tau )$ is the learned logit scale of CLIP, clipped at 100. CLIP predicts the class with the highest logit $\ell _ { i c } ,$ and under domain shift, $\mathbf { v } _ { i }$ can become misaligned with the fixed $\mathbf { t } _ { c } .$

Problem Setting. We adapt a frozen CLIP model to a target domain whose images may differ from the pretraining distribution. An unlabeled adaptation set $\mathcal { D } _ { u } = \{ x _ { i } \} _ { i = 1 } ^ { N }$ of targetdomain images is given. DRC estimates its statistics once from $\mathcal { D } _ { u }$ without any target labels and then classifies each test image with these fixed statistics. The visual encoder $E _ { v } ,$ the text encoder $E _ { t } .$ , and the classifier weights $\mathbf { t } _ { c }$ remain unchanged throughout. Figure 1 illustrates the DRC pipeline, which recenters each image feature with a Gaussian mixture fitted on $\mathcal { D } _ { u }$ and then calibrates the logits with a confidenceweighted class prior.

## 3.2. Domain Recentering

DRC models the features of $\mathcal { D } _ { u }$ with a Gaussian mixture and estimates the bias of each image from all mixture components, weighting each component by how strongly the image belongs to it.

Each image, whether in $\mathcal { D } _ { u }$ or at test time, is represented by averaging its ℓ -normalized feature with that of its horizontal flip, followed by re-normalization

$$
\mathbf { f } _ { i } = \operatorname { n o r m } \left( \operatorname { n o r m } ( E _ { v } ( x _ { i } ) ) + \operatorname { n o r m } ( E _ { v } ( \operatorname { F l i p } ( x _ { i } ) ) ) \right) .\tag{2}
$$

With the mean $\begin{array} { r } { \pmb { \mu } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { f } _ { i } } \end{array}$ over $\mathcal { D } _ { u } ,$ the centered features are projected via PCA onto min $( 1 6 , D , N )$ dimensions, where D is the CLIP embedding dimension, so that the mixture is fitted on the dominant directions of variation

$$
\mathbf { z } _ { i } = \mathbf { P } ( \mathbf { f } _ { i } - \mu ) .\tag{3}
$$

A diagonal-covariance GMM with K components,

$$
p ( { \mathbf { z } } ) = \sum _ { k = 1 } ^ { K } \pi _ { k } \mathcal { N } ( { \mathbf { z } } ; \nu _ { k } , { \boldsymbol { \Sigma } } _ { k } ) ,\tag{4}
$$

![](images/0729a05b80f2baa1ddf6c3cfc86d4ebebe1757c1d6b1e4bdf20710bf43c44882.jpg)  
Fig. 1. Overview of DRC. DRC recenters CLIP image features with a GMM-based soft bias correction and calibrates the resulting logits with a confidence-weighted class prior.

is fitted to $\left\{ \mathbf { z } _ { i } \right\}$ , where $\pi _ { k } , \nu _ { k }$ , and $\Sigma _ { k }$ denote the mixture weight, mean, and diagonal covariance of component k. Unlike hard clustering such as K-means, the GMM assigns each feature softly to the K components through its posterior $q _ { i , k } = p ( k \mid \mathbf { z } _ { i } )$

Each component mean is computed in the original embedding space from the features whose most probable component is $k , \mathcal { T } _ { k } = \{ i \ | \ k = \arg \operatorname* { m a x } _ { j } q _ { i , j } \}$ , so that it summarizes the features belonging mainly to that component

$$
\mathbf { m } _ { k } = \frac { 1 } { \vert \mathcal { T } _ { k } \vert } \sum _ { i \in \mathcal { T } _ { k } } \mathbf { f } _ { i } .\tag{5}
$$

For an input feature f, DRC estimates its bias by weighting these means with the posterior

$$
{ \bf b } ( { \bf f } ) = \sum _ { k = 1 } ^ { K } q _ { k } ( { \bf f } ) { \bf m } _ { k } , \qquad q _ { k } ( { \bf f } ) = p \big ( k \mid { \bf P } ( { \bf f } - \mu ) \big ) .\tag{6}
$$

Since $q _ { k } ( \mathbf { f } )$ varies continuously with f, the bias changes gradually as a feature moves between components, whereas a hard assignment would switch it abruptly at component boundaries.

The recentered feature is obtained by subtracting this bias and re-normalizing

$$
\hat { \mathbf { f } } = \frac { \mathbf { f } - \beta \mathbf { b } ( \mathbf { f } ) } { \left\| \mathbf { f } - \beta \mathbf { b } ( \mathbf { f } ) \right\| _ { 2 } } ,\tag{7}
$$

where $\beta$ controls the strength of recentering.

## 3.3. Confidence-Weighted Prior Calibration

Recentering acts only on the image features and leaves the classifier weights $\mathbf { t } _ { c }$ unchanged, so the class preference of the zero-shot classifier [8, 7] can persist in the recentered predictions. DRC estimates this preference from predictions on $\mathcal { D } _ { u }$ and removes it with a log-prior adjustment, weighting each prediction by its confidence.

For each $x _ { i } \in \mathcal { D } _ { u }$ , let $\hat { \ell } _ { i c }$ be the logit from the recentered feature. The class probability is then

$$
p _ { i } ( c ) = \frac { \exp ( \hat { \ell } _ { i c } ) } { \sum _ { c ^ { \prime } } \exp ( \hat { \ell } _ { i c ^ { \prime } } ) } .\tag{8}
$$

Since an uncertain prediction provides weak evidence of class preference, each prediction is weighted by one minus its normalized entropy

$$
H _ { i } = - \sum _ { c } p _ { i } ( c ) \log p _ { i } ( c ) , \qquad w _ { i } = 1 - { \frac { H _ { i } } { \log C } } ,\tag{9}
$$

where C is the number of classes, so that near-uniform predictions receive weights close to zero. The confidence-weighted class prior is

$$
\pi _ { c } ^ { \mathrm { C o n f } } = \frac { \sum _ { i = 1 } ^ { N } w _ { i } p _ { i } ( c ) } { \sum _ { i = 1 } ^ { N } w _ { i } } .\tag{10}
$$

The computed distribution summarizes the class preference reflected in reliable predictions on the adaptation set. A large $\pi _ { c } ^ { \mathrm { C o n f } }$ indicates that class c receives a relatively large amount of prediction mass, while a small value denotes that the class is less frequently preferred by the model. Confidence weighting reduces the influence of uncertain predictions that could otherwise distort this estimate.

The log-prior correction and its zero-centered form are

$$
r _ { c } = - \log ( \pi _ { c } ^ { \mathrm { C o n f } } + \epsilon ) , \qquad \tilde { r } _ { c } = r _ { c } - \frac { 1 } { C } \sum _ { c ^ { \prime } = 1 } ^ { C } r _ { c ^ { \prime } } ,\tag{11}
$$

where ϵ is a small constant for numerical stability. A test image with recentered feature <sup>ˆ</sup>f is classified with the calibrated logits

$$
\ell _ { c } ^ { \prime } = \hat { \ell } _ { c } + \tilde { r } _ { c } , \qquad \hat { \ell } _ { c } = \exp ( \tau ) \hat { \mathbf { f } } ^ { \top } \frac { \mathbf { t } _ { c } } { \lVert \mathbf { t } _ { c } \rVert _ { 2 } } .\tag{12}
$$

This adjustment lowers the logits of classes that receive excessive prediction mass on $\mathcal { D } _ { u }$ and raises those of less preferred classes.

## 4. EXPERIMENTS

## 4.1. Experimental Setup

Datasets. We evaluate DRC on the cross-domain benchmark and the out-of-distribution (OOD) benchmark. For cross-domain evaluation, we use the same ten datasets as TPT [4], which are Aircraft, Caltech101, Cars, DTD, EuroSAT, Flower102, Food101, Pets, SUN397, and UCF101. The OOD benchmark measures robustness to distribution shifts with four variants of ImageNet [9], ImageNet-A [10], ImageNet-V2 [11], ImageNet-R [12] and ImageNet-Sketch [13].

Implementation details. Text embeddings obtained from dataset-specific prompt templates are averaged into a single vector for each class, and image features are extracted from each image and its horizontal flip. DRC uses PCA and a diagonal-covariance GMM $( n _ { \mathrm { i n i t } } = 5 ,$ , seed 42). The adaptation set $\mathcal { D } _ { u }$ is the validation split of each cross-domain dataset and the unlabeled evaluation images for ImageNet and its variants. K and β are selected by accuracy on the validation split of each cross-domain dataset, and on the ImageNet validation set for the OOD benchmark, with the ImageNet values reused for all four variants.

## 4.2. Comparison with State-of-the-Art

Cross-Domain Benchmark. Table 1 evaluates DRC across ten datasets with disjoint class spaces. DRC achieves the highest average accuracy with both CLIP-ViT-B/16 and CLIP-ResNet-50, reaching 68.72% and 61.70%, respectively. These results improve over zero-shot CLIP by 4.13 and 5.07 points, and DRC ranks first on 7 and 6 of the 10 datasets, respectively. On EuroSAT, DRC improves over the best competing method by 10.75 and 3.80 points, and on UCF101, the gains are 3.52 and 2.33 points, respectively.

OOD Benchmark. Table 2 further evaluates DRC on the OOD benchmark. With CLIP-ViT-B/16, DRC achieves the highest Average and OOD Average of 62.79% and 60.94%. With CLIP-ResNet-50, DRC achieves 48.53% and 45.14%, improving over zero-shot CLIP by 2.10 and 2.05 points, respectively. The gains are particularly pronounced under strong appearance shifts. DRC achieves the highest accuracy on ImageNet-R/S for both backbones (61.39%/39.94% for CLIP-ResNet-50, 78.14%/52.09% for CLIP-ViT-B/16).

## 4.3. Ablation Study

We perform ablation studies on the cross-domain benchmark with CLIP-ViT-B/16 to examine the effectiveness of domain recentering and confidence-weighted prior calibration.

As shown in Table 3, both components improve over the CLIP baseline. Soft recentering outperforms hard recentering both without and with prior calibration. Confidence-weighted prior calibration provides the largest gain, increasing the average accuracy to 68.35%. Combining both components yields the best result, with DRC achieving an average accuracy of 68.72%.

<table><tr><td>Method</td><td>Aircraft</td><td>Caltech101</td><td>Cars</td><td>DTD</td><td>EuroSAT</td><td>Flower102</td><td>Food101</td><td>Pets</td><td>SUN397</td><td>UCF101</td><td>Average</td></tr><tr><td>CLIP-ResNet-50</td><td>16.11</td><td>87.26</td><td>55.89</td><td>40.37</td><td>25.79</td><td>62.77</td><td>74.82</td><td>82.97</td><td>60.85</td><td>59.48</td><td>56.63</td></tr><tr><td>CoOp</td><td>15.12</td><td>86.53</td><td>55.32</td><td>37.29</td><td>26.20</td><td>61.55</td><td>75.59</td><td>87.00</td><td>58.15</td><td>59.05</td><td>56.18</td></tr><tr><td>CoCoOp</td><td>14.61</td><td>87.38</td><td>56.22</td><td>38.53</td><td>28.73</td><td>65.57</td><td>76.20</td><td>88.39</td><td>59.61</td><td>57.10</td><td>57.23</td></tr><tr><td>TPT</td><td>17.58</td><td>87.02</td><td>58.46</td><td>40.84</td><td>28.33</td><td>62.69</td><td>74.88</td><td>84.49</td><td>61.46</td><td>60.82</td><td>57.66</td></tr><tr><td>DiffTPT</td><td>17.60</td><td>86.89</td><td>60.71</td><td>40.72</td><td>41.04</td><td>63.53</td><td>79.21</td><td>83.40</td><td>62.72</td><td>62.67</td><td>59.85</td></tr><tr><td>UMFC</td><td>17.76</td><td>85.48</td><td>56.25</td><td>42.85</td><td>37.63</td><td>65.57</td><td>77.47</td><td>86.10</td><td>60.49</td><td>63.39</td><td>59.30</td></tr><tr><td>DRC (Ours)</td><td>18.81</td><td>87.99</td><td>59.66</td><td>44.33</td><td>44.84</td><td>66.30</td><td>78.83</td><td>88.01</td><td>62.46</td><td>65.72</td><td>61.70</td></tr><tr><td>CLIP-ViT-B/16</td><td>23.22</td><td>93.55</td><td>66.11</td><td>45.04</td><td>50.42</td><td>66.99</td><td>82.86</td><td>86.92</td><td>65.63</td><td>65.16</td><td>64.59</td></tr><tr><td>CoOp</td><td>18.47</td><td>93.70</td><td>64.51</td><td>41.92</td><td>46.39</td><td>68.71</td><td>85.30</td><td>89.14</td><td>64.15</td><td>66.55</td><td>63.88</td></tr><tr><td>CoCoOp</td><td>22.29</td><td>93.79</td><td>64.90</td><td>45.45</td><td>39.23</td><td>70.85</td><td>83.97</td><td>90.46</td><td>66.89</td><td>68.44</td><td>64.63</td></tr><tr><td>TPT</td><td>24.78</td><td>94.16</td><td>66.87</td><td>47.75</td><td>42.44</td><td>68.98</td><td>84.67</td><td>87.79</td><td>65.50</td><td>68.04</td><td>65.10</td></tr><tr><td>DiffTPT</td><td>25.60</td><td>92.49</td><td>67.01</td><td>47.00</td><td>43.13</td><td>70.10</td><td>87.23</td><td>88.22</td><td>65.74</td><td>68.22</td><td>65.47</td></tr><tr><td>UMFC</td><td>25.47</td><td>91.44</td><td>66.35</td><td>45.57</td><td>50.98</td><td>70.60</td><td>85.99</td><td>88.91</td><td>65.73</td><td>68.91</td><td>66.00</td></tr><tr><td>DRC (Ours)</td><td>27.39</td><td>92.17</td><td>68.82</td><td>47.46</td><td>61.73</td><td>71.78</td><td>86.80</td><td>91.11</td><td>67.52</td><td>72.43</td><td>68.72</td></tr></table>

Table 1. Results on the Cross-Domain Benchmark with CLIP-ResNet-50 and CLIP-ViT-B/16. CoOp and CoCoOp serve as supervised transfer baselines trained on ImageNet with 16 labeled samples per class. CLIP, CoOp, CoCoOp, and TPT results are taken from the original TPT paper [4], and DiffTPT results are taken from the original DiffTPT paper [5].
<table><tr><td>Method</td><td>ImageNet</td><td>-A</td><td>-V2</td><td>-R</td><td>-S</td><td>Average</td><td>OOD Average</td></tr><tr><td>CLIP-ResNet-50</td><td>59.81</td><td>23.24</td><td>52.91</td><td>60.72</td><td>35.48</td><td>46.43</td><td>43.09</td></tr><tr><td>CoOp</td><td>63.33</td><td>23.06</td><td>55.40</td><td>56.60</td><td>34.67</td><td>46.61</td><td>42.43</td></tr><tr><td>CoCoOp</td><td>62.81</td><td>23.32</td><td>55.72</td><td>57.74</td><td>34.48</td><td>46.81</td><td>42.82</td></tr><tr><td>TPT</td><td>60.74</td><td>26.67</td><td>54.70</td><td>59.11</td><td>35.09</td><td>47.26</td><td>43.89</td></tr><tr><td>DiffTPT</td><td>60.80</td><td>31.06</td><td>55.80</td><td>58.80</td><td>37.10</td><td>48.71</td><td>45.69</td></tr><tr><td>UMFC</td><td>59.70</td><td>23.47</td><td>52.85</td><td>60.94</td><td>35.82</td><td>46.56</td><td>43.27</td></tr><tr><td>DRC (Ours)</td><td>62.12</td><td>23.77</td><td>55.44</td><td>61.39</td><td>39.94</td><td>48.53</td><td>45.14</td></tr><tr><td>CLIP-ViT-B/16</td><td>68.34</td><td>49.89</td><td>61.88</td><td>77.65</td><td>48.24</td><td>61.20</td><td>59.42</td></tr><tr><td>CoOp</td><td>71.51</td><td>49.71</td><td>64.20</td><td>75.21</td><td>47.99</td><td>61.72</td><td>59.28</td></tr><tr><td>CoCoOp</td><td>71.02</td><td>50.63</td><td>64.07</td><td>76.18</td><td>48.75</td><td>62.13</td><td>59.91</td></tr><tr><td>TPT</td><td>68.98</td><td>54.77</td><td>63.45</td><td>77.06</td><td>47.94</td><td>62.44</td><td>60.81</td></tr><tr><td>DiffTPT</td><td>70.30</td><td>55.68</td><td>65.10</td><td>75.00</td><td>46.80</td><td>62.28</td><td>60.52</td></tr><tr><td>UMFC</td><td>68.21</td><td>50.33</td><td>61.99</td><td>77.98</td><td>48.77</td><td>61.46</td><td>59.77</td></tr><tr><td>DRC (Ours)</td><td>70.17</td><td>49.71</td><td>63.83</td><td>78.14</td><td>52.09</td><td>62.79</td><td>60.94</td></tr></table>

Table 2. Results on the OOD Benchmark with CLIP-ResNet-50 and CLIP-ViT-B/16. CoOp and CoCoOp serve as supervised transfer baselines trained on ImageNet with 16 labeled samples per class. CLIP, CoOp, CoCoOp, and TPT results are taken from the original TPT paper [4], and DiffTPT results are taken from the original DiffTPT paper [5].

<table><tr><td rowspan="2">Recentering</td><td colspan="2">Prior calibration</td></tr><tr><td>without</td><td>with</td></tr><tr><td>None</td><td>64.59</td><td>68.35</td></tr><tr><td>Hard</td><td>66.89</td><td>68.57</td></tr><tr><td>Soft</td><td>67.01</td><td>68.72</td></tr></table>

Table 3. Ablation on the cross-domain benchmark with CLIP-ViT-B/16, reporting the average accuracy over the ten datasets. The bottom-right entry is DRC.

## 5. CONCLUSION

In this paper, we proposed DRC, a training-free method that improves the robustness of CLIP under distribution shift without any backpropagation or parameter updates. DRC fits a Gaussian mixture to unlabeled target features and subtracts from each feature a soft, image-specific bias, so that the correction varies continuously across mixture components. It then corrects class preferences remaining in the recentered predictions with a confidence-weighted class prior. DRC achieves the highest average accuracy among the compared methods on the cross-domain benchmark and improves the OOD Average of zero-shot CLIP with both backbones.

## 6. ACKNOWLEDGEMENTS

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-00561169), Global – Learning & Academic research institution for Master’s · PhD students, and Postdocs (G-LAMP) Program of the National Research Foundation of Korea (NRF) grant funded by the Ministry of Education (No. RS-2025-25442252), and Institute of Information & communications Technology Planning & Evaluation (IITP) under the Leading Generative AI Human Resources Development (IITP-2027-RS-2026-25546026) grant funded by the Korea government (MSIT).

## 7. REFERENCES

[1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[2] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu, “Learning to prompt for vision-language models,” International journal of computer vision, vol. 130, no. 9, pp. 2337–2348, 2022.

[3] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu, “Conditional prompt learning for visionlanguage models,” in 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR). IEEE, 2022, pp. 16795–16804.

[4] Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom Goldstein, Anima Anandkumar, and Chaowei Xiao, “Test-time prompt tuning for zero-shot generalization in vision-language models,” Advances in Neural Information Processing Systems, vol. 35, pp. 14274–14289, 2022.

[5] Chun-Mei Feng, Kai Yu, Yong Liu, Salman Khan, and Wangmeng Zuo, “Diverse data augmentation with diffusions for effective test-time prompt tuning,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 2704–2714.

[6] Jiachen Liang, Ruibing Hou, Minyang Hu, Hong Chang, Shiguang Shan, and Xilin Chen, “Umfc: Unsupervised multi-domain feature calibration for vision-language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 114072–114093, 2024.

[7] Shubham Parashar, Zhiqiu Lin, Tian Liu, Xiangjue Dong, Yanan Li, Deva Ramanan, James Caverlee, and

Shu Kong, “The neglected tails in vision-language models,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 12988–12997.

[8] Xingyu Zhu, Beier Zhu, Yi Tan, Shuo Wang, Yanbin Hao, and Hanwang Zhang, “Enhancing zero-shot vision models by label-free prompt distribution learning and bias correcting,” Advances in Neural Information Processing Systems, vol. 37, pp. 2001–2025, 2024.

[9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision andpattern recognition. Ieee, 2009, pp. 248–255.

[10] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song, “Natural adversarial examples,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 15262–15271.

[11] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar, “Do imagenet classifiers generalize to imagenet?,” in International conference on machine learning. PMLR, 2019, pp. 5389–5400.

[12] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al., “The many faces of robustness: A critical analysis of out-ofdistribution generalization,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2021, pp. 8320–8329.

[13] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing, “Learning robust global representations by penalizing local predictive power,” Advances in neural information processing systems, vol. 32, 2019.
# Cross-Layer Refinement and Frequency-aware Fusion Network for Multi-Source Remote Sensing Image Classification

Wenhan Xiao, Chuanzheng Gong, Feng Gao, Member, IEEE, Junyu Dong, Member, IEEE, Qian Du, Fellow, IEEE

Abstract—Multi-source remote sensing image classification benefits from complementary spectral and structural information. However, identifying discriminative cross-modal features remains challenging. This letter proposes a Cross-Layer Refinement and Frequency-aware Fusion Network (CRFNet) within a lightweight dual-branch architecture. Specifically, a Cross-Layer Refinement Attention (CLRA) module propagates selected attention relationships across successive layers and employs top relation selection to progressively strengthen discriminative spatial dependencies while suppressing irrelevant responses. Furthermore, a Frequency-aware Complementary Fusion (FCF) module combines frequency-wise interactions with cross-modal channel attention in the Fourier domain. The resulting representations are then converted into spatial-domain gates to selectively modulate complementary features. Experiments on the Houston 2013 and Augsburg datasets demonstrate that the proposed CRFNet consistently outperforms several state-of-the-art methods. The codes are publicly available at https://github.com/oucailab/CRFNet.

Index Terms—Hyperspectral image, multi-source fusion, land cover classification, cross-layer refinement, synthetic aperture radar.

## I. INTRODUCTION

ULTI-SOURCE remote sensing data classification comneous sensors [1]. By integrating complementary information acquired from different sensors, it provides a more comprehensive characterization of land-cover properties than relying on a single data source. Among these data sources, hyperspectral images (HSIs) provide abundant spectral information, and the detailed spectral signatures enable the discrimination of landcover categories with similar visual appearances [2]. Hence, HSI is particularly valuable for fine-grained classification [3].

However, HSI commonly suffers from limited spatial resolution and high spectral redundancy, which may hinder the accurate characterization of spatial structures. Therefore, integrating HSI with complementary data, such as synthetic aperture radar (SAR) or light detection and ranging (LiDAR), offers an effective way to jointly exploit spectral, structural, and elevation information [4]. The all-weather, day-and-night imaging capability of SAR data enables reliable observations under cloud cover or poor illumination [5]. LiDAR data captures accurate elevation and three-dimensional geometric information [6]. Motivated by these complementary properties, this letter investigates the fusion of HSI with LiDAR or SAR data for land-cover classification.

Accordingly, the fusion of multi-source data has attracted considerable attention in land-cover classification. Existing methods generally integrate heterogeneous observations at the data, feature, or decision level [7]. Among these strategies, feature-level fusion provides a flexible means of combining complementary representations learned from different modalities. Bi et al. [8] employ a synergistic Mamba fusion module to dynamically align and integrate optical and elevation features through cross-modal spatial interactions at the encoder level. Gong et al. [4] propose RSCNet, which selects task-relevant HSI bands using cross-source guidance and adaptively fuses them with SAR or LiDAR features through cross-source attention and local–global refinement. Zhang et al. [9] progressively fuse multi-source features through multi-scale alignment and adaptively gated cross-modal attention, combining the outputs of all stages using learnable fusion weights. Effective fusion remains challenging because different sensors exhibit substantial discrepancies in imaging mechanisms and spatial characteristics.

Despite these promising advances, two key challenges persist in multi-source remote sensing data classification: ➊ How to progressively identify and strengthen discriminative features. In multi-source feature extraction, complementary cues are often intertwined with noise and background interference, making it difficult to isolate features that reliably distinguish land-cover classes. This challenge calls for a progressive refinement mechanism that preserves informative features and increasingly focuses feature aggregation on task-relevant cues. ➋ How to jointly reinforce shared cross-modal features and preserve modality-specific complementary cues. Most existing fusion strategies that primarily emphasize cross-modal similarity may suppress useful differential features, while indiscriminate aggregation can introduce irrelevant information. The challenge therefore lies in selectively retaining complementary differential cues that enhance class discrimination.

To address these challenges, we propose Cross-Layer Refinement and Frequency-aware Fusion Network (CRFNet) for multi-source remote sensing classification. Specifically, we introduce Cross-Layer Refinement Attention (CLRA), which propagates retained attention relations across successive layers to guide feature refinement. By combining cross-layer guidance with top relation selection, CLRA progressively strengthens discriminative associations while suppressing irrelevant responses. Furthermore, we develop a Frequency-aware Complementary Fusion (FCF) module to facilitate cross-modal interactions in the Fourier domain. FCF combines frequency interactions with cross-modal channel attention and transforms the aggregated representation into a spatial-domain gate for feature modulation.

![](images/e26314762d5b0e544e84ca92286095b53913c20cc40696525adcbb48d1ef93fb.jpg)  
Fig. 1. Overview of the proposed Cross-Layer Refinement and Frequency-aware Fusion Network (CRFNet) for multi-source remote sensing classification. The dual branches extract features from HSI and SAR/LiDAR data, respectively. Cross-Layer Refinement Attention (CLRA) blocks progressively refine features through cross-layer interactions, while Frequency-aware Complementary Fusion (FCF) modules integrate complementary information across modalities at successive stages.

The contributions of this letter are summarized as follows:

• We design CLRA that propagates retained attention relations across successive layers. Cross-layer guidance and layer-wise top-k selection progressively strengthen discriminative associations.

• We develop FCF module that combines frequency interactions with cross-modal channel attention. The obtained features are transformed into a spatial-domain gate to selectively enhance cross-modal information.

• Extensive experiments on two benchmark datasets demonstrate that the proposed CRFNet outperforms several state-of-the-art methods.

## II. METHODOLOGY

## A. Overall Framework of the Proposed CRFNet

As illustrated in Fig. 1, the proposed CRFNet adopts a dualbranch architecture for multi-source remote sensing classification. A 3D convolution first extracts spectral–spatial features from the HSI input, while a convolutional layer encodes structural information from the structural SAR/LiDAR data. Both branches subsequently employ stacked Cross-Layer Refinement Attention (CLRA) blocks to progressively refine their representations through interactions between features from different depths. Each CLRA block comprises a cross-layer attention unit and a multi-layer perceptron (MLP), with layer normalization and residual connections. At successive stages, Frequency-aware Complementary Fusion (FCF) modules facilitate cross-modal information exchange by reinforcing shared features and retaining modality-specific complementary cues. The resulting features are fed into subsequent refinement module to produce the land-cover classification map.

![](images/68a6ee599ea1af1a6c45ed0c1af8d4cd95d60553466749746f4cf838a7c2984f.jpg)  
Fig. 2. Illustration of the Cross-Layer Refinement Attention (CLRA).

## B. Cross-Layer Refinement Attention

In multi-source feature extraction, complementary cues are often intertwined with noise and background interference, making it difficult to isolate features that reliably distinguish land-cover classes. To address this issue, we design the Cross-Layer Refinement Attention (CLRA) to use attention information retained from the previous layer to guide relation modeling in the current layer, and then selects the dominant features to restrict the propagation of irrelevant information. The same spatial CLRA design is applied independently in the HSI and structural branches; their attention masks are not shared. Therefore, CLRA can focus feature aggregation on complementary cues that contribute to class discrimination.

Current-Layer Attention Computation. The input feature tensor at l-th layer is $\mathbf { X } _ { l } \in \mathbb { R } ^ { R \times R \times C }$ , where R and C denote the input patch size, and number of channels, respectively. A linear layer is employed to the generate query Q, key K, and value V. The similarities between spatial positions are first calculated and normalized using Softmax over the key positions:

$$
\widetilde { \mathbf { A } } _ { l } = \mathrm { S o f t m a x } \left( \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } \right) .\tag{1}
$$

This step establishes global spatial relationships within the current layer but does not yet incorporate the selection information from the previous layer.

Cross-Layer Attention Refinement. The refined attention propagated from the previous layer is $\mathbf { A } _ { l - 1 }$ . To introduce cross-layer guidance, the current attention map $\bar { \mathbf { A } } _ { l }$ is multiplied element-wise by the refined attention map $\mathbf { A } _ { l - 1 }$ propagated from the previous layer:

$$
\mathbf { B } _ { l } = \widetilde { \mathbf { A } } _ { l } \odot \mathbf { A } _ { l - 1 } ,\tag{2}
$$

where $\odot$ denotes element-wise multiplication. At the first layer, $\mathbf { B } _ { 1 } ~ = ~ \widetilde { \mathbf { A } } _ { 1 }$ . This operation uses previously retained relations to modulate current attention responses, assigning higher selection priority to associations supported by both layers and attenuating those receiving weak support from the previous layer.

Top Relation Selection. We compute the refined attention map by applying sparse selection to $\mathbf { B } _ { l } \mathbf { : }$

$$
\mathbf { A } _ { l } = \mathcal { T } ( \mathbf { B } ) ,\tag{3}
$$

where $\tau ( \cdot )$ is the learnable top-k selection operator:

$$
[ T ( \mathbf { B } ) ] _ { i j } = { \left\{ \begin{array} { l l } { \mathbf { B } _ { i j } , } & { \mathbf { B } _ { i j } \in \mathrm { t o p } \mathrm { - } k ( j ) , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{4}
$$

An adaptive selection of the top-k contribution scores are employed to preserve the most dominant components and remove the useless ones. This dynamic selection makes the attention from dense to sparse. The top-k selection ratio α is an important parameter, and its effect will be analyzed in the experiments.

The refined attention map ${ \bf A } _ { l }$ is combined with V to generate the output:

$$
\mathbf { O } = \mathbf { A } _ { l } \otimes \mathbf { V } .\tag{5}
$$

Meanwhile, ${ \bf A } _ { l }$ is propagated to the next layer to guide subsequent attention refinement, enabling successive layers to build upon the retained associations and progressively focus feature aggregation on discriminative cues.

## C. Frequency-aware Complementary Fusion

In multi-source data fusion, most existing fusion models emphasize cross-modal similarity may suppress useful differential features, while indiscriminate aggregation can introduce irrelevant information. To this end, we propose Frequencyaware Complementary Fusion (FCF) module for multi-source feature fusion. As shown in Fig. 3, given the hyperspectral features ${ \bf { F } } _ { h }$ and LiDAR or SAR features $\mathbf { F } _ { x }$ , three linear projections generate the query Q, key K, and value V representations as:

![](images/7bda83a153cdaffb8579d5bc579ba17ac4a12c74954808faefad6a7421139dd3.jpg)  
Fig. 3. Illustration of the Frequency-aware Complementary Fusion (FCF) module.

$$
{ \bf Q } = \phi _ { q } ( { \bf F } _ { h } ) , ~ { \bf K } = \phi _ { k } ( { \bf F } _ { x } ) , ~ { \bf V } = \phi _ { v } ( { \bf F } _ { x } ) ,\tag{6}
$$

where $\phi _ { q } , \phi _ { k }$ and $\phi _ { v }$ denote independent linear mappings. The projected features share the same dimensions $C \times H \times W$ . We apply two-dimensional discrete Fourier transform (DFT) to Q and K, followed by spatial flattening:

$$
\mathbf { F } _ { q } = \mathcal { F } ( \mathbf { Q } ) , \qquad \mathbf { F } _ { k } = \mathcal { F } ( \mathbf { K } ) ,\tag{7}
$$

where $\mathcal { F }$ denotes the Fourier transform. Two parallel paths subsequently capture complementary cross-modal interactions. The first computes an element-wise frequency interaction:

$$
\mathbf { P } = \mathbf { F } _ { q } \odot \mathbf { F } _ { k } ,\tag{8}
$$

which retains interactions at corresponding channels and frequency locations. The second constructs a cross-modal channel attention matrix:

$$
\mathbf { A } = \mathrm { S o f t m a x } \left( \frac { \mathbf { F } _ { q } \mathbf { F } _ { k } ^ { T } } { \sqrt { d } } \right) ,\tag{9}
$$

where Softmax is applied along each row, yielding ${ \textbf { A } } \in$ $\mathbb { R } ^ { C \times C }$ . The channel attention then aggregates the frequency interaction features:

$$
\begin{array} { r } { \mathbf { Z } = \mathbf { A } \mathbf { P } , \qquad \mathbf { Z } \in \mathbb { R } ^ { C \times M } . } \end{array}\tag{10}
$$

Finally, the aggregated representation is reshaped and transformed back into the spatial domain to modulate the value features:

$$
\mathbf { G } = \mathcal { F } ^ { - 1 } \left( \mathbf { Z } \right) , \qquad \mathbf { F } _ { f u s } = \mathbf { G } \odot \mathbf { V } .\tag{11}
$$

Here, ${ \mathcal { F } } ^ { - 1 }$ denotes the inverse Fourier transform. By combining frequency-wise interactions with cross-channel aggregation, FCF produces a joint cross-modal response that adaptively modulates structural information for subsequent classification.

## III. EXPERIMENTAL RESULTS AND ANALYSIS

## A. Datasets and Settings

We evaluate the performance of the proposed CRFNet on two benchmark datasets for multi-source remote sensing data classification. The first is the Houston 2013 dataset, released for the 2013 IEEE GRSS Data Fusion Contest. It covers the University of Houston campus and its surrounding urban areas, USA, and comprises an HSI and a co-registered LiDAR-derived digital surface model. Both modalities contain $3 4 9 \times 1 9 0 5$ pixels at a spatial resolution of 2.5 m. The second is the Augsburg dataset [10], acquired over Augsburg, Germany, from which we use the co-registered HSI and SAR observations. The scene contains 332 × 485 pixels at a spatial resolution of 30 m and includes seven land-cover classes. Its HSI comprises 180 spectral bands covering 0.4–2.5 µm, while the SAR data provide complementary microwave scattering information.

![](images/ea183f8d7cf6385599c89710d1174186b373aa6248efdf4240e02a80deffb22c.jpg)  
(a)

![](images/d991f0eb08d2dc5cbb63a2d18e22f0ee590293b3a288c815038687621cb201fa.jpg)  
(b)  
Fig. 4. Parameter analysis of CRFNet. (a) Number of CLRA blocks. (b) Ratio in top-k selection.

Our experiments were conducted on an NVIDIA GeForce RTX 4090 GPU. It is trained for 200 epochs using the AdamW optimizer, with a batch size of 128, a learning rate of $1 \times 1 0 ^ { - 4 }$ , and an embedding dimension of C = 64. To evaluate the performance of the proposed CRFNet, we compare it against other state-of-the-art methods, including S<sup>2</sup>ENet [11], DFINet [12], AsyFF [13], ExViT [14], HCT [15], MACN [16], and RSCNet [4]. To quantitatively evaluate the experimental results, we employ three metrics: Overall Accuracy (OA), Average Accuracy (AA), and Kappa coefficient. OA and AA are reported as percentages, and Kappa is scaled by 100 in the tables. Accuracy differences are expressed in percentage points.

## B. Parameter Analysis

Number of CLRA Blocks. We investigate the effect of the number of CLRA blocks by varying the block number from 1 to 4. As shown in Fig. 4(a), both datasets achieve their highest OA with three blocks. Adding a fourth block reduces OA, particularly on Augsburg. These results suggest that progressively refining attention benefits feature discrimination, but additional network depth does not necessarily yield further improvements. Therefore, we use three CLRA blocks in our experiments.

Ratio in Top-k Selection. We examine the influence of the top-k selection ratio by varying it from 0.4 to 0.9. As shown in Fig. 4(b), the OA on Houston 2013 initially increases, peaks at a ratio of 0.7, and then decreases as the ratio increases further. This suggests that an appropriate selection ratio balances retaining informative attention contributions and suppressing irrelevant ones. In contrast, the performance on Augsburg remains relatively stable across the tested ratios, indicating lower sensitivity to this hyperparameter.

## C. Classification Performance Comparison

Tables I and II report class-wise recall together with OA, AA, and Kappa. On Augsburg, CRFNet achieves the highest

TABLE I  
CLASSIFICATION PERFORMANCE OF DIFFERENT METHODS ON THE AUGSBURG DATASET. THE BOLD AND UNDERLINE DENOTE THE BEST AND SECOND RESULTS.
<table><tr><td>Class</td><td>|S2ENet DFINet AsyFF</td><td></td><td></td><td>ExViT</td><td>HCT</td><td>MACN</td><td>RSCNet</td><td>CRFNet</td></tr><tr><td rowspan="3">Forest Residential Industrial Low plants</td><td>98.10</td><td>97.38</td><td>97.56 99.16</td><td>90.04 97.26 95.44</td><td>98.63</td><td>97.84 98.73</td><td>96.07 96.40</td><td>96.77 97.94</td></tr><tr><td>99.08 12.19</td><td>98.37 61.31</td><td>61.38</td><td>34.58</td><td>36.61</td><td>43.94</td><td>78.38</td><td>56.48</td></tr><tr><td>91.78</td><td>92.63</td><td>83.47</td><td>90.68</td><td>93.54</td><td>94.11</td><td>94.08</td><td>96.62</td></tr><tr><td rowspan="3">Allotment Commercial Water</td><td>45.12</td><td>49.33</td><td>42.64</td><td>51.82</td><td>51.43</td><td>52.39</td><td>58.32</td><td>61.38</td></tr><tr><td>1.22</td><td>3.54 26.61</td><td>5.58 45.81</td><td>28.63</td><td>7.97</td><td>6.80</td><td>1.96</td><td>3.24</td></tr><tr><td>24.09</td><td></td><td></td><td>17.65</td><td>46.34</td><td>48.54</td><td>50.07</td><td>48.91</td></tr><tr><td rowspan="3">OA AA Kappa</td><td>88.22</td><td>90.66</td><td>88.25</td><td>86.65</td><td>90.33</td><td>91.06</td><td>91.50</td><td>92.04</td></tr><tr><td>53.08</td><td>61.31</td><td>62.23</td><td>58.41</td><td>61.68</td><td>63.19</td><td>67.89</td><td>65.90</td></tr><tr><td>86.47</td><td>86.47</td><td>83.13</td><td>80.79</td><td>86.03</td><td>87.06</td><td>87.76</td><td>88.43</td></tr></table>

TABLE II

CLASSIFICATION PERFORMANCE OF DIFFERENT METHODS ON THE HOUSTON2013 DATASET. THE BOLD AND UNDERLINE DENOTE THE BEST AND SECOND RESULTS.
<table><tr><td>Class</td><td>[S2ENet DFINet AsyFF</td><td></td><td></td><td></td><td></td><td>ExViT HCT MACN 1</td><td>RSCNet</td><td>CRFNet</td></tr><tr><td>Healthy grass</td><td>81.48</td><td>82.91</td><td>82.72</td><td>82.72</td><td>81.10</td><td>82.90</td><td>82.53</td><td>82.81</td></tr><tr><td>Stressed grass</td><td>84.68</td><td>85.15</td><td>83.93</td><td>85.15</td><td>86.09</td><td>100.0</td><td>79.98</td><td>94.83</td></tr><tr><td>Synthetic grass</td><td>99.01</td><td>99.60</td><td>100.0</td><td>99.21</td><td>99.80</td><td>99.80</td><td>99.60</td><td>100.0</td></tr><tr><td>Trees</td><td>92.14</td><td>92.33</td><td>91.95</td><td>91.95</td><td>98.01</td><td>98.11</td><td>99.15</td><td>97.16</td></tr><tr><td>Soil</td><td>100.0</td><td>99.81</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.23</td><td>100.0</td></tr><tr><td>Water</td><td>100.0</td><td>100.0</td><td>95.80</td><td>98.60</td><td>95.10</td><td>95.80</td><td>98.60</td><td>95.80</td></tr><tr><td>Residential</td><td>89.93</td><td>93.47</td><td>96.18</td><td>86.66</td><td>83.49</td><td>78.17</td><td>86.57</td><td>88.43</td></tr><tr><td>Commercial</td><td>90.18</td><td>82.58</td><td>81.14</td><td>91.05</td><td>86.43</td><td>80.46</td><td>87.97</td><td>94.02</td></tr><tr><td>Road</td><td>92.63</td><td>89.05</td><td>86.12</td><td>91.12</td><td>86.69</td><td>90.93</td><td>90.93</td><td>92.82</td></tr><tr><td>Highway</td><td>64.86</td><td>56.56</td><td>64.58</td><td></td><td>64.86 79.05</td><td>55.88</td><td>96.14</td><td>89.38</td></tr><tr><td>Railway</td><td>97.60</td><td>94.15</td><td>86.76</td><td>77.06</td><td>95.97</td><td>99.33</td><td>99.33</td><td>93.26</td></tr><tr><td>Parking lot 1</td><td>88.09</td><td>94.72</td><td>90.59</td><td>88.76</td><td>98.46</td><td>99.71</td><td>96.93</td><td>98.75</td></tr><tr><td>Parking lot 2</td><td>91.93</td><td>89.12</td><td>92.28</td><td>87.72</td><td>90.88</td><td>88.77</td><td>89.47</td><td>88.77</td></tr><tr><td>Tennis court</td><td>100.0</td><td>100.0</td><td>95.14</td><td>97.17</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Running track</td><td>100.0</td><td>100.0</td><td>100.0</td><td>98.52</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.79</td></tr><tr><td>OA</td><td>89.56</td><td>88.59</td><td>87.95</td><td>87.42</td><td>90.66</td><td>89.81</td><td>92.66</td><td>93.75</td></tr><tr><td>AA</td><td>91.50</td><td>90.63</td><td>89.81</td><td>89.37</td><td>92.07</td><td>91.33</td><td>93.76</td><td>94.39</td></tr><tr><td>Kappa</td><td>88.69</td><td>87.64</td><td>86.96</td><td>86.38</td><td>89.86</td><td>88.93</td><td>92.04</td><td>93.22</td></tr></table>

TABLE III

COMPARISON OF MODEL COMPLEXITY ON THE AUGSBURG DATASET.
<table><tr><td>Metric</td><td>S2ENet</td><td>DFINet</td><td>AsyFF</td><td>ExViT</td><td>HCT</td><td>MACN</td><td>RSCNet</td><td>CRFNet</td></tr><tr><td>Params (M)</td><td>1.7055</td><td>0.8460</td><td>2.0987</td><td>1.8807</td><td>4.7811</td><td>2.0199</td><td>2.8812</td><td>0.3492</td></tr><tr><td>FLOPs (G)</td><td>0.2065</td><td></td><td>0.124640.1789</td><td>0.2897</td><td>0.0437</td><td>0.1672</td><td>0.3346</td><td>0.0821</td></tr></table>

OA of 92.04%. Its strongest class-wise results are obtained for low plants and allotment. The corresponding classification maps in Fig. 5 further verify these results. In the enlarged region, CRFNet reproduces the main forest area and the adjacent low-plants region in the ground truth, with little false labeling. It is demonstrated that our CRFNet preserves local land-cover patterns while distinguishing vegetation classes.

On Houston2013, CRFNet obtains the highest OA, AA, and Kappa of 93.75%, 94.39%, and 93.22%, respectively. It ranks first or second in nine of the fifteen classes. In terms of class-wise accuracy, CRFNet achieves the best results for Commercial and Road, reaching 94.02% and 92.82%, respectively, and attains 100% accuracy for Synthetic grass, Soil, and Tennis court. These results demonstrate the strong overall classification performance of our CRFNet on this dataset.

## D. Computational Complexity

Table III compares the parameter counts and FLOPs of different methods on the Augsburg dataset. CRFNet requires only 0.3492 M parameters, achieving the smallest model size among all compared methods. Specifically, it reduces the parameter count by 58.72% and 87.88% compared with DFINet and RSCNet, respectively. In terms of computational complexity, CRFNet requires 0.0821 GFLOPs, ranking second only to HCT. These results demonstrate that CRFNet combines a compact model size with low computational cost.

![](images/1e341c2e066a7527f31aa8137c19b9eb11e33f31e86fdfe40efc2d9710cd0ff4.jpg)

![](images/e1fef22cb3e3532c842c7851a4bbbdf0b17f4c08dbf831e349c0f5bf623c7f25.jpg)  
Fig. 5. Qualitative comparison of the classification maps on the Augsburg dataset, with enlarged views of a selected region. (a) Ground truth. (b) S<sup>2</sup>ENet. (c) DFINet. (d) AsyFF. (e) ExViT. (f) HCT. (g) MACN. (h) RSCNet. (i) Proposed CRFNet.

TABLE IV  
ABLATION EXPERIMENTS. OA (%) IS REPORTED.
<table><tr><td>CLRA</td><td>FCF</td><td>Augsburg</td><td>Houston2013</td></tr><tr><td>X</td><td>×</td><td>90.71</td><td>89.36</td></tr><tr><td>X</td><td>√</td><td>91.51</td><td>92.24</td></tr><tr><td>√</td><td>X</td><td>91.24</td><td>92.03</td></tr><tr><td></td><td>√</td><td>92.04</td><td>93.75</td></tr></table>

## E. Ablation Study

Table IV presents the ablation results of CLRA and FCF on both datasets. Without either component, the baseline achieves OA values of 90.71% and 89.36%, respectively. Introducing CLRA alone yields gains of 0.53 and 2.67 percentage points. Similarly, incorporating FCF alone increases OA to 91.51% and 92.24%. Combining both components achieves the best OA values. These results demonstrate that both components contribute to classification performance and that their combination provides further improvements on both datasets.

## IV. CONCLUSION

We proposed CRFNet, a lightweight dual-branch network for multi-source remote sensing image classification. Specifically, CLRA progressively enhances discriminative spatial dependencies through cross-layer attention refinement and top relation selection, while FCF exploits cross-modal complementary information through bidirectional frequency-domain interactions. Extensive experiments on two benchmark datasets demonstrate that the proposed CRFNet consistently outperforms state-of-the-art methods.

## REFERENCES

[1] Q. Song, J. Peng, W. Song, B. Sun, R. Dian, and S. Li, “Multidomain adaptive fusion network for multi-source remote sensing data classification,” Science China Information Sciences, vol. 69, no. 8, p. 182107, 2026.

[2] J. Li, K. Zheng, L. Gao, Z. Han, Z. Li, and J. Chanussot, “Enhanced deep image prior for unsupervised hyperspectral image super-resolution,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–18, 2025.

[3] Y. Chu, C. Wang, J. Peng, W. Ding, and P. Li, “Ugmft: Uncertainty guided multimodal fusion transformer for multi-source remote sensing classification,” Pattern Recognition, p. 114812, 2026.

[4] C. Gong, F. Gao, J. Lin, J. Dong, and Q. Du, “Representative spectral correlation network for multisource remote sensing image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 1– 14, 2026.

[5] P. Duan, Y. Shang, Z. Wang, X. Kang, and S. Li, “Spectral–spatial dynamic scan mamba for multi-source remote sensing data classification,” IEEE Transactions on Image Processing, vol. 35, pp. 5324–5336, 2026.

[6] Y. Zhao, C. Gong, B. Huan, F. Gao, J. Dong, and Q. Du, “Semanticguided fusion network for multisource remote sensing image classification,” IEEE Geoscience and Remote Sensing Letters, vol. 23, pp. 1–5, 2026.

[7] C. Liang, Y. Zhao, Y. Song, and K. Ni, “Idnet: Intensity-constrained detail-enhanced network for hyperspectral and lidar collaborative classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–15, 2025.

[8] Z. Bi, B. Zhang, W. Zhang, and C. Wang, “Synergistic Mamba-Transformer fusion for high-fidelity land cover classification from multimodal remote sensing imagery,” Computers & Geosciences, vol. 213, p. 106169, 2026.

[9] Y. Zhang, H. Gao, Z. Chen, S. Fei, J. Zhou, P. Ghamisi, and B. Zhang, “Adaptive multi-stage fusion of hyperspectral and LiDAR data via selective state space models,” Information Fusion, vol. 125, p. 103488, 2026.

[10] D. Hong, J. Hu, J. Yao, J. Chanussot, and X. X. Zhu, “Multimodal remote sensing benchmark datasets for land cover classification with a shared and specific feature learning model,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 178, pp. 68–80, 2021.

[11] S. Fang, K. Li, and Z. Li, “S<sup>2</sup>ENet: Spatial–spectral cross-modal enhancement network for classification of hyperspectral and lidar data,” IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1–5, 2022, art. no. 6504205.

[12] Y. Gao et al., “Hyperspectral and multispectral classification for coastal wetland using depthwise feature interaction network,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, 2022, art. no. 5512615.

[13] W. Li, Y. Gao, M. Zhang, R. Tao, and Q. Du, “Asymmetric feature fusion network for hyperspectral and SAR image classification,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 10, pp. 8057–8070, 2022.

[14] J. Yao, B. Zhang, C. Li, D. Hong, and J. Chanussot, “Extended vision transformer (ExViT) for land use and land cover classification: A multimodal deep learning framework,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, 2023, art. no. 5514415.

[15] G. Zhao, Q. Ye, L. Sun, Z. Wu, C. Pan, and B. Jeon, “Joint classification of hyperspectral and lidar data using a hierarchical CNN and transformer,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, 2023, art. no. 5500716.

[16] K. Li, D. Wang, X. Wang, G. Liu, Z. Wu, and Q. Wang, “Mixing self-attention and convolution: A unified framework for multisource remote sensing data classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, 2023, art. no. 5523216.
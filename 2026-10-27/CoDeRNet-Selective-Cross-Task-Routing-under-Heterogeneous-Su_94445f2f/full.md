# CoDeRNet: Selective Cross-Task Routing under Heterogeneous Supervision for Change Detection and Captioning

Eunki Cho Hyeon Bae Kim Seong Tae Kim

Kyung Hee University, Republic of Korea {cekkec, hyeonbae.kim, st.kim}@khu.ac.kr

## Abstract

Change Detection & Captioning (CDC) aims to jointly localize changed regions and describe their semantic content from bi-temporal remote sensing images. While recent approaches have demonstrated the effectiveness of unified modeling, they primarily rely on shared representations or jointly optimized architectures, leaving task-specific communication under heterogeneous supervision less explored. However, detection and captioning differ fundamentally in their supervision granularity, leading to heterogeneous representations with uneven reliability across tasks and spatial locations. In this paper, we reformulate CDC as a selective information routing problem between heterogeneous task branches, where the key challenge is to determine when and where information should be transferred. To this end, we propose CoDeRNet, a unified framework that preserves task-specific representations while enabling bidirectional exchange of complementary information. CoDeRNet introduces a Confidence-Decomposed Routing (CoDeR) mechanism, which decomposes the routing condition into receiver-side need for support and sender-side reliability cues to guide selective transfer. Experiments on LEVIR-MCI and WHU-CDC show that CoDeRNet achieves strong overall CDC performance, particularly improving caption generation while preserving competitive change localization. Extensive analyses further show that the benefits arise not merely from joint learning or generic cross-task communication, but from explicitly modeling selective cross-task information routing. These results highlight the importance of conditional information transfer for unified CDC under heterogeneous supervision.

## 1 Introduction

Understanding changes between bi-temporal remote sensing observations is a fundamental problem in remote sensing, with broad applications including urban planning [14], natural disaster assessment [43], and environmental monitoring [16]. Two representative tasks have been widely studied in this context: Change Detection (CD) [1, 7, 9, 11, 13, 18, 19], which localizes changed regions at the pixel level, and Change Captioning (CC) [4, 22–24, 32, 36, 37, 49], which generates natural-language descriptions of semantic changes. Although their output formats differ, both tasks are grounded in the same underlying change event and provide complementary views of the same phenomenon. Motivated by this commonality, recent studies have introduced the unified setting of Change Detection & Captioning (CDC) [25, 35]. This unified setting is particularly appealing because it allows a single model to jointly reason about both where a change occurs and what the change means, offering a more complete form of change interpretation than either task alone.

However, these tasks differ fundamentally in their supervision granularity. Detection is densely supervised at the pixel level, whereas captioning is weakly supervised at the sentence level. This heterogeneity introduces a structural mismatch in how information is learned and represented across the two tasks. Existing CDC approaches primarily rely on shared representations or jointly optimized architectures [25, 35], establishing a unified framework for learning spatial localization and semantic description together. While this paradigm is effective, it mainly emphasizes joint modeling of the two tasks, leaving the role of task-specific communication under heterogeneous supervision less explored. Importantly, uniform interaction or jointly learned interaction mechanisms implicitly assume that cross-task information is equally reliable and equally useful across spatial locations and tasks. However, under heterogeneous supervision, task representations often exhibit asymmetric reliability and support requirements, making unconditional information transfer suboptimal. Therefore, the central challenge is to determine how complementary information should be selectively transferred between heterogeneous task branches, rather than treating CDC merely as a joint optimization problem.

This perspective is particularly important in CDC because the usefulness of cross-task information can vary across both tasks and spatial locations. For example, detection-oriented features may provide strong spatial evidence around changed regions and boundaries, but they have limited capacity to explicitly organize such evidence into sentence-level semantic descriptions. Conversely, captioning-oriented features may capture global context and semantic relations, but sentence-level supervision does not explicitly constrain their spatial precision. This suggests that cross-task communication in CDC should be spatially and task-wise conditional, rather than uniformly applied across branches. Each branch should receive complementary information only when its own representation is insufficient and the other branch provides reliable supporting evidence.

Based on this motivation, we reformulate CDC as a selective information routing problem between heterogeneous task branches. To this end, we propose CoDeRNet, a unified framework that preserves task-specific branch identities while enabling bidirectional exchange of complementary information. CoDeRNet performs selective transfer via Confidence-Decomposed Routing (CoDeR), which decomposes the routing condition into receiver-side need for support and sender-side reliability cues. An overview of the proposed framework is shown in Fig. 1.

We demonstrate that CoDeRNet achieves strong overall CDC performance on representative benchmarks, including LEVIR-MCI and WHU-CDC, with notable gains in caption generation while maintaining competitive change localization. Moreover, through systematic analyses of joint learning, routing directionality, mask decomposition, and alternative interaction operators, we show that the gains arise not merely from joint learning or generic interaction, but from explicitly modeling selective cross-task routing. The code is publicly available at https://github.com/cekkec/CoDeRNet.

The main contributions of this paper are summarized as follows.

• We reformulate CDC as a selective cross-task information routing problem under heterogeneous supervision, where complementary information should be transferred in a conditional and task-dependent manner.

• We propose CoDeRNet, a unified CDC framework that preserves the distinct roles of detection and captioning while enabling task-aware bidirectional routing. Its core mechanism, Confidence-Decomposed Routing (CoDeR), guides selective transfer by decomposing the routing condition into receiver-side need for support and sender-side reliability cues.

• We validate CoDeRNet on representative CDC benchmarks, including LEVIR-MCI and WHU-CDC, and demonstrate through systematic analyses of joint learning, routing directionality, mask decomposition, and alternative interaction operators the effectiveness of selective cross-task routing under heterogeneous supervision.

## 2 Related Work

Remote Sensing Change Detection. Remote sensing change detection (RSCD) aims to identify meaningful changes from bi-temporal images of the same geographic area. Deep RSCD methods have evolved from CNN-based fully convolutional and Siamese encoder-decoder architectures [5, 8, 9, 11, 13, 18, 19, 47] to Transformer-based models that capture long-range dependencies and global context [1, 6, 12, 30, 34, 45, 48]. These studies have improved change localization by enhancing multi-scale representation learning, temporal feature comparison, and contextual reasoning over bi-temporal observations. More recently, Mamba-based architectures have been explored for efficient spatiotemporal modeling [7, 46], further expanding RSCD toward scalable and globally aware change representation.

Remote Sensing Image Change Captioning. Remote sensing image change captioning (RSICC) aims to generate natural-language descriptions of semantic changes from bi-temporal images. Existing methods have improved change captioning through encoder-decoder architectures [32], Transformerbased cross-temporal modeling and multi-scale change representation [22, 23], pretrained language models and decoupled reasoning [24], attentive or lightweight architectures [4, 36], bi-temporal pretraining [49], and diffusion-based change modeling [37]. These approaches have progressively strengthened the ability to localize visual change cues, model temporal differences, and translate them into semantically coherent descriptions.

Remote Sensing Change Detection & Captioning. Remote sensing change detection and captioning has recently emerged as a unified task for comprehensive change interpretation, combining pixel-level localization with semantic-level change description. MCINet [25] introduced LEVIR-MCI and proposed a dual-branch framework that jointly performs change detection and change captioning. CTMTNet [35] further formalized this direction as the RS-CDC task and developed a shared multitask architecture with task-specific decoders. These studies established unified CDC as a promising direction for jointly modeling spatial and semantic change information.

## 3 Method

## 3.1 Overview

CoDeRNet addresses remote sensing change detection and change captioning within a unified yet selective information routing framework. Given a bi-temporal image pair $\left( I _ { t _ { 0 } } , I _ { t _ { 1 } } \right)$ , the model first extracts multi-scale visual features from the two images and constructs shared change representations. These shared change features are then passed through branch-specific interfaces that provide separate feature spaces for detection and captioning. On top of these branch-specific features, Confidence-Decomposed Routing (CoDeR) selectively routes complementary information between the two branches. Finally, task-specific decoders produce the change mask and the change description, respectively. As illustrated in Fig. 1, the entire framework is optimized end-to-end under a joint training objective.

## 3.2 Shared Change Representation

Given a bi-temporal image pair $( I _ { t _ { 0 } } , I _ { t _ { 1 } } )$ , where $I _ { t _ { 0 } }$ and $I _ { t _ { 1 } }$ denote the earlier and later observations, respectively, we first extract multi-scale visual features from the two images using a shared encoder. Let $F _ { t _ { 0 } } ^ { ( s ) } , \bar { F } _ { t _ { 1 } } ^ { ( s ) } \in \mathbb { R } ^ { B \times C _ { s } \times H _ { s } \times W _ { s } }$ denote the features of the two observations at scale s. To construct a shared representation of temporal change, we combine the two temporal features with two complementary change cues: a magnitude change cue, which captures the strength of change, and a signed change cue, which preserves directional temporal differences. The resulting change representation at each scale is defined as

$$
\begin{array} { r } { \boldsymbol { F } ^ { ( s ) } = \operatorname { C o n v } _ { 1 \times 1 } \big ( [ F _ { t _ { 0 } } ^ { ( s ) } , F _ { t _ { 1 } } ^ { ( s ) } , F _ { | \Delta | } ^ { ( s ) } , F _ { \Delta } ^ { ( s ) } ] \big ) , } \end{array}
$$

where $F _ { | \Delta | } ^ { ( s ) } = | F _ { t _ { 1 } } ^ { ( s ) } - F _ { t _ { 0 } } ^ { ( s ) } |$ and $F _ { \Delta } ^ { ( s ) } = F _ { t _ { 1 } } ^ { ( s ) } - F _ { t _ { 0 } } ^ { ( s ) }$ denote the magnitude and signed change cues.

## 3.3 Branch-Specific Feature Interface

Before cross-task routing, we instantiate branch-specific feature interfaces from the shared change representation. This step provides separate feature spaces in which CoDeR can estimate branchdependent routing cues and apply residual information transfer:

$$
F _ { \mathrm { s e g } } ^ { ( s ) } = \phi _ { \mathrm { s e g } } ^ { ( s ) } ( F ^ { ( s ) } ) , \qquad F _ { \mathrm { c a p } } ^ { ( s ) } = \phi _ { \mathrm { c a p } } ^ { ( s ) } ( F ^ { ( s ) } ) ,
$$

where $\phi _ { \mathrm { s e g } } ^ { ( s ) }$ and $\phi _ { \mathrm { c a p } } ^ { ( s ) }$ are lightweight projection layers. These interfaces provide the feature spaces on which selective information routing is performed, while the actual cross-task transfer is handled by CoDeR. The same interfaces are used for all compared variants, so that subsequent comparisons focus on the routing mechanism rather than differences in branch construction.

![](images/1946129c577a233ff67e807381e65d355ded80e54373a097f761663cd88fd3a6.jpg)  
Figure 1: Overview of CoDeRNet. Shared change features are transformed into task-specific detection and captioning features, followed by Confidence-Decomposed Routing (CoDeR) for selective bidirectional transfer before task-specific decoding.

## 3.4 Confidence-Decomposed Routing

After branch-specific interfaces are constructed, the two branches encode different yet complementary aspects of the same change. The detection branch tends to provide stronger cues for modeling where a change occurs, whereas the captioning branch tends to provide richer cues for reasoning about what the change means. Rather than directly fusing the two branches, we explicitly model where cross-task transfer is likely to be beneficial. Consider a target feature map $F _ { t } \overset { \mathbf { \lambda } } { \in } \mathbb { R } ^ { B \times C \times H \times W }$ from the receiving branch and a source feature map $F _ { s } \in \mathbb { R } ^ { B \times C \times H \times W }$ from the sending branch. CoDeR estimates target-side sufficiency and source-side reliability cues using lightweight routing estimators and derives a routing mask as

$$
C _ { t } = \mathcal { R } _ { t } ( F _ { t } ) , \qquad C _ { s } = \mathcal { R } _ { s } ( F _ { s } ) , \qquad M = ( 1 - C _ { t } ) \cdot C _ { s } ,
$$

where $\mathcal { R } _ { t } ( \cdot ) = \sigma ( h _ { t } ( \cdot ) )$ and $\mathcal { R } _ { s } ( \cdot ) = \sigma ( h _ { s } ( \cdot ) )$ are lightweight routing estimators implemented by $1 \times 1$ convolution heads with sigmoid activation, and $\breve { C } _ { t } , C _ { s } , \breve { M } \in [ 0 , 1 ] ^ { \breve { B } \times 1 \times H \times W }$ . Here, $C _ { t }$ and $\dot { C _ { s } }$ are not calibrated predictive probabilities, but task-dependent routing cues learned to estimate where cross-task transfer is expected to be complementary. The term $( 1 - \bar { C } _ { t } )$ represents the receiver-side need for support, while $C _ { s }$ represents the sender-side reliability cue. Their multiplicative combination acts as a soft conjunction, making the routing mask large only when the receiving branch requires support and the sending branch provides reliable evidence. In this way, CoDeR routes information conditionally rather than applying cross-task transfer uniformly across all spatial locations.

The source feature is then mapped to the target feature space through a lightweight adapter g and selectively injected via residual transfer:

$$
F _ { t } ^ { \prime } = F _ { t } + M \odot g ( F _ { s } ) ,
$$

where ⊙ denotes element-wise multiplication with spatial broadcasting. When M is small, the original target representation is largely preserved; when M is large, the target branch receives

Table 1: Comparison on LEVIR-MCI [25]. “Change Detection” and “Change Captioning” denote single-task models trained for only one task, whereas “Change Detection & Captioning” denotes unified models that perform both tasks jointly within a single framework. Bold and underlined values indicate the best and second-best results, respectively.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td colspan="8">Metrics</td></tr><tr><td>mIoU</td><td>BLEU-1</td><td>BLEU-2</td><td>BLEU-3</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td></tr><tr><td rowspan="6">Change Detection</td><td>SNUNet [11]</td><td>82.53</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ChangeFormer [1]</td><td>84.35</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TFI-GR [18]</td><td>79.34</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>A2Net [19]</td><td>80.82</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HANet [13]</td><td>71.23</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ChangeMamba [7]</td><td>87.61</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="8">Change Captioning</td><td>DUDA [32]</td><td></td><td>81.44</td><td>72.22</td><td>64.24</td><td>57.79</td><td>37.15</td><td>71.04</td><td>124.32</td></tr><tr><td>MCCFormers [33]</td><td></td><td>82.16</td><td>72.95</td><td>65.42</td><td>59.41</td><td>38.26</td><td>72.10</td><td>128.34</td></tr><tr><td>RSICCformer [22]</td><td></td><td>84.11</td><td>75.40</td><td>68.01</td><td>61.93</td><td>38.79</td><td>73.02</td><td>131.40</td></tr><tr><td>Prompt-CC [24]</td><td></td><td>83.66</td><td>75.73</td><td>69.10</td><td>63.54</td><td>38.82</td><td>73.72</td><td>136.44</td></tr><tr><td>Chg2Cap [4]</td><td></td><td>85.14</td><td>76.91</td><td>69.86</td><td>64.09</td><td>39.83</td><td>74.62</td><td>135.41</td></tr><tr><td>SparseFocus [36]</td><td></td><td>84.56</td><td>75.87</td><td>68.64</td><td>62.87</td><td>39.93</td><td>74.69</td><td>137.05</td></tr><tr><td>SEN [49]</td><td></td><td>85.10</td><td>77.05</td><td>70.01</td><td>64.09</td><td>39.59</td><td>74.57</td><td>136.02</td></tr><tr><td>MaskApproxNet [37]</td><td></td><td>85.90</td><td>77.12</td><td>70.72</td><td>64.32</td><td>39.91</td><td>75.67</td><td>137.71</td></tr><tr><td>Change</td><td>CTMTNet [35]</td><td>85.37</td><td>85.16</td><td>77.24</td><td>70.13</td><td>64.16</td><td>39.34</td><td>74.03</td><td>134.50</td></tr><tr><td>Detection</td><td>MCINet [25]</td><td>86.43</td><td>86.68</td><td>78.75</td><td>71.74</td><td>65.95</td><td>40.80</td><td>75.96</td><td>140.29</td></tr><tr><td>&amp; Captioning</td><td>CoDeRNet (Ours)</td><td>87.79</td><td>87.25</td><td>79.42</td><td>72.26</td><td>66.27</td><td>41.08</td><td>76.36</td><td>141.79</td></tr></table>

complementary evidence from the source branch. This selective residual update allows CoDeRNet to preserve branch-specific representations while refining regions that may benefit from cross-task support.

CoDeR is applied bidirectionally, i.e., from captioning to detection and from detection to captioning, with independent routing estimators and adapters for the two directions:

$$
\begin{array} { r } { F _ { \mathrm { s e g } } ^ { \prime ( s _ { g } ) } = \mathrm { C o D e R } _ { \mathrm { c a p }  \mathrm { s e g } } \big ( F _ { \mathrm { s e g } } ^ { ( s _ { g } ) } , F _ { \mathrm { c a p } } ^ { ( s _ { g } ) } \big ) , \quad F _ { \mathrm { c a p } } ^ { \prime ( s _ { c } ) } = \mathrm { C o D e R } _ { \mathrm { s e g }  \mathrm { c a p } } \big ( F _ { \mathrm { c a p } } ^ { ( s _ { c } ) } , F _ { \mathrm { s e g } } ^ { ( s _ { c } ) } \big ) . } \end{array}
$$

Within each direction, the source and target features are taken from the same spatial scale, which allows the routing mask to be applied directly without additional resizing. The design is asymmetric across the two routing directions: the detection branch receives routed information at a relatively higher-resolution scale, whereas the captioning branch receives routed information at a lowerresolution scale. This design reflects the different spatial granularity requirements of the two tasks.

## 3.5 Task-Specific Decoders

After branch-specific interfaces and CoDeR-based routing, each branch maintains a multi-scale feature hierarchy. CoDeR is applied only at a designated receiving scale for each branch, while the remaining scales are preserved as branch-specific features. The resulting hierarchies are then decoded by task-specific heads that reflect the distinct granularity and inductive bias required by change detection and change captioning. For notational simplicity, we use the primed feature notation to denote the branch features after CoDeR. At scales without CoDeR, the primed features are identical to the original branch features.

Segmentation Decoder. Change detection requires spatially precise prediction with progressive recovery of spatial detail. We therefore use an FPN-style top-down decoder [21] to aggregate the multi-scale segmentation branch features. The decoder progressively combines coarse semantic features with higher-resolution features and produces pixel-wise class logits:

$$
\hat { M } = \mathcal { D } _ { \mathrm { s e g } } \big ( \{ F _ { \mathrm { s e g } } ^ { \prime ( s ) } \} _ { s \in \mathcal { S } } \big ) \in \mathbb { R } ^ { B \times K \times H \times W } ,
$$

where $s$ denotes the set of feature scales and K denotes the number of change classes.

Captioning Decoder. Unlike segmentation, caption generation requires organizing change evidence into semantic units. We therefore construct a multi-granularity memory from the caption-oriented branch features:

$$
\mathcal { M } = [ t _ { \mathrm { g l o b } } ; T _ { \mathrm { r e g } } ; T _ { \mathrm { l o c } } ] , \qquad \hat { p } _ { t + 1 } = \mathcal { D } _ { \mathrm { c a p } } ( y _ { 1 : t } ^ { * } , \mathcal { M } ) .
$$

Table 2: Comparison on WHU-CDC [35]. “Change Detection” and “Change Captioning” denote single-task models trained for only one task, whereas “Change Detection & Captioning” denotes unified models that perform both tasks jointly within a single framework. Bold and underlined values indicate the best and second-best results, respectively.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td colspan="8">Metrics</td></tr><tr><td>IoU</td><td>BLEU-1</td><td>BLEU-2</td><td>BLEU-3</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td></tr><tr><td rowspan="6">Change Detection</td><td>SNUNet [11]</td><td>74.81</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ChangeFormer [1]</td><td>77.88</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TFI-GR [18]</td><td>77.60</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>A2Net [19]</td><td>80.43</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HANet [13]</td><td>79.95</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ChangeMamba [7]</td><td>89.69</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="8">Change Captioning</td><td>DUDA [32]</td><td></td><td>79.04</td><td>69.53</td><td>61.57</td><td>55.64</td><td>34.29</td><td>68.98</td><td>121.85</td></tr><tr><td>MCCFormers [33]</td><td></td><td>81.12</td><td>75.04</td><td>69.95</td><td>65.34</td><td>42.11</td><td>78.52</td><td>147.09</td></tr><tr><td>RSICCformer [22]</td><td></td><td>80.05</td><td>74.24</td><td>69.61</td><td>66.54</td><td>42.65</td><td>73.91</td><td>133.44</td></tr><tr><td>Prompt-CC [24]</td><td></td><td>81.12</td><td>73.96</td><td>67.22</td><td>61.45</td><td>36.99</td><td>71.88</td><td>134.50</td></tr><tr><td>Chg2Cap [4]</td><td></td><td>77.27</td><td>71.33</td><td>65.10</td><td>58.44</td><td>39.95</td><td>78.67</td><td>160.64</td></tr><tr><td>SparseFocus [36]</td><td></td><td>81.17</td><td>72.90</td><td>66.06</td><td>60.27</td><td>37.34</td><td>72.63</td><td>134.64</td></tr><tr><td>SEN [49]</td><td></td><td>80.60</td><td>74.64</td><td>67.69</td><td>61.97</td><td>36.76</td><td>71.70</td><td>133.57</td></tr><tr><td>MaskApproxNet [37]</td><td></td><td>81.34</td><td>75.68</td><td>71.16</td><td>67.73</td><td>43.89</td><td>75.41</td><td>135.31</td></tr><tr><td>Change</td><td>CTMTNet [35]</td><td>81.54</td><td>83.56</td><td>77.66</td><td>72.76</td><td>69.00</td><td>45.39</td><td>79.23</td><td>149.40</td></tr><tr><td>Detection</td><td>MCINet [25]</td><td>88.23</td><td>87.82</td><td>82.84</td><td>78.66</td><td>75.49</td><td>49.03</td><td>82.54</td><td>159.57</td></tr><tr><td>&amp; Captioning</td><td>CoDeRNet (Ours)</td><td>90.25</td><td>87.44</td><td>82.38</td><td>77.85</td><td>74.17</td><td>49.04</td><td>83.58</td><td>163.37</td></tr></table>

Here, $t _ { \mathrm { g l o b } } , T _ { \mathrm { r e g } } ,$ , and $T _ { \mathrm { l o c } }$ denote global, region-level, and local tokens that capture overall context, intermediate semantic units, and fine spatial evidence, respectively. The Transformer decoder $\mathcal { D } _ { \mathrm { c a p } }$ predicts the next-token distribution using the caption prefix and the multi-granularity memory.

## 3.6 Training Objective

CoDeRNet is trained in a multi-task setting for change detection and change captioning. Following prior CDC practice [25], we use task-specific losses for the two branches and optimize them jointly:

$$
\mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { C E } } ( \hat { M } , M ^ { * } ) + \mathcal { L } _ { \mathrm { D i c e } } ( \hat { M } , M ^ { * } ) , \quad \mathcal { L } _ { \mathrm { c a p } } = - \sum _ { t = 1 } ^ { T } \log p ( y _ { t } ^ { * } \mid y _ { < t } ^ { * } , \mathcal { M } ) .
$$

Here, $M ^ { * }$ denotes the ground-truth change mask. The segmentation branch is supervised by pixelwise cross-entropy and Dice loss, while the captioning branch is trained with token-level cross-entropy under teacher forcing over valid non-padded positions.

For stable joint optimization, we normalize each loss by its detached current magnitude:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \frac { \mathcal { L } _ { \mathrm { s e g } } } { \mathrm { s g } ( \mathcal { L } _ { \mathrm { s e g } } ) + \epsilon } + \frac { \mathcal { L } _ { \mathrm { c a p } } } { \mathrm { s g } ( \mathcal { L } _ { \mathrm { c a p } } ) + \epsilon } ,
$$

where $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator and ϵ is a small constant for numerical stability. This normalization is used only as a training stabilization strategy to balance the scales of the two task losses.

## 4 Experiments

## 4.1 Datasets and Evaluation Metrics

Datasets. We evaluate CoDeRNet on two representative CDC benchmarks, LEVIR-MCI [25] and WHU-CDC [35]. LEVIR-MCI, extended from LEVIR-CC [22], contains 10,077 bi-temporal image pairs of size 256 × 256 at 0.5 m resolution, each annotated with a multi-class change mask for buildings and roads and five human-written captions. Following the standard split, it is divided into 6,815 training, 1,333 validation, and 1,929 test samples.

WHU-CDC, derived mainly from WHU-CD [15], contains 7,434 bi-temporal image pairs of size 256 × 256 at 0.075 m resolution, focusing on post-earthquake urban reconstruction in Christchurch, New Zealand. Each pair has a binary building-change mask and five captions describing multiple change categories, including buildings, parking lots, roads, vegetation, and water. The training, validation, and test sets contain 5,947, 743, and 744 samples, respectively. These datasets allow us to evaluate CoDeRNet under different levels of mask granularity and caption semantics.

![](images/fb9a1b6de4f1689a5ba7e901d9d3df1ca77f3404bdad7845536a34653a8d20cd.jpg)  
Figure 2: Qualitative comparison on LEVIR-MCI. In the captions, building-related words (e.g., house, building) and road-related words (e.g., road) are highlighted using colors consistent with the corresponding mask classes.

Both datasets can contain multiple change events per image pair and cover multiple change categories in their annotations, although WHU-CDC masks are restricted to building changes.

Evaluation Metrics. For change detection, we follow prior CDC benchmarks and report mIoU on LEVIR-MCI [25], which provides multi-class change masks for buildings, roads, and no-change regions. On WHU-CDC [35], which provides binary building-change masks, we report the IoU of the change class. For change captioning, we adopt standard language generation metrics, including BLEU-N (N = 1,2,3,4) [31], METEOR [2], ROUGE-L [20], and CIDEr-D [39], covering n-gram overlap, semantic matching, sequence similarity, and consensus-based caption quality.

## 4.2 Implementation Details

All experiments are implemented in PyTorch and conducted on a single NVIDIA RTX 4090 GPU. We use a ConvNeXt-Base backbone [27] with ImageNet-1K pretrained weights and apply it to the two temporal images in a weight-shared manner. The maximum caption length is set to 41 for LEVIR-MCI [25] and 26 for WHU-CDC [35], and greedy autoregressive decoding is used during inference. For training, we use Adam [17] with a learning rate of $\mathrm { \bar { 1 } \times 1 0 ^ { - 4 } }$ and an effective batch size of 64, achieved through a micro-batch size of 16 and four gradient accumulation steps. The learning rate is decayed by a factor of 0.5 every 50 epochs, and the model is trained for 100 epochs on LEVIR-MCI and 150 epochs on WHU-CDC. We apply random horizontal and vertical flips together with 90<sup>◦</sup> rotations during training. When geometric augmentation is applied, direction-sensitive caption words are remapped accordingly. Detailed remapping rules are provided in Sec. 7.3. For both LEVIR-MCI and WHU-CDC, the input size is fixed to 256 × 256. For this setup, a single training run takes approximately 12 hours.

## 4.3 Comparison with State-of-the-Art Methods

We compare CoDeRNet with recent state-of-the-art methods on LEVIR-MCI [25] and WHU-CDC [35], including single-task change detection models, single-task change captioning models, and unified change detection-captioning models. This comparison evaluates whether a unified framework can improve spatial localization and semantic description of changes at the same time. As shown in

Table 3: Ablation studies on LEVIR-MCI [25]. (a) Effect of joint training and explicit interaction, comparing single-task training, joint training without CoDeR, and joint training with CoDeR. (b) Effect of CoDeR directionality, comparing two single-direction settings with the bidirectional design. Bold indicates the best result in each column.  
(a) Joint training and CoDeR
<table><tr><td>Seg</td><td>Cap</td><td>CoDeR</td><td>mIoU</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td></tr><tr><td>√</td><td></td><td></td><td>87.03</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>√</td><td></td><td></td><td>63.18</td><td>40.21</td><td>75.27</td><td>137.41</td></tr><tr><td>√</td><td>√</td><td></td><td>87.67</td><td>64.41</td><td>39.93</td><td>74.98</td><td>136.18</td></tr><tr><td>√</td><td>√</td><td>√</td><td>87.79</td><td>66.27</td><td>41.08</td><td>76.36</td><td>141.79</td></tr></table>

(b) CoDeR directionality
<table><tr><td>Direction</td><td>mIoU</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td></tr><tr><td>cap → seg</td><td>87.53</td><td>64.79</td><td>40.27</td><td>75.25</td><td>138.62</td></tr><tr><td>seg → cap</td><td>87.60</td><td>64.96</td><td>40.95</td><td>76.33</td><td>141.95</td></tr><tr><td> $\mathbf { s e g }  \mathbf { c a p }$ </td><td>87.79</td><td>66.27</td><td>41.08</td><td>76.36</td><td>141.79</td></tr></table>

Table 1, CoDeRNet achieves the best overall performance on LEVIR-MCI across both detection and captioning metrics. For example, CoDeRNet achieves BLEU-4 66.27, CIDEr-D 141.79, and mIoU 87.79 on LEVIR-MCI, showing stronger overall performance than the strongest prior unified model.

Table 2 shows a similar trend on WHU-CDC. CoDeRNet achieves the best IoU of 90.25 and improves consensus-oriented captioning metrics, obtaining the best CIDEr-D of 163.37 and ROUGE-L of 83.58, while remaining competitive on BLEU-based metrics.

Compared with prior unified CDC frameworks, CoDeRNet preserves strong localization accuracy while improving semantic description quality across both datasets, leading to a better overall balance between the two tasks. Overall, these results show that CoDeRNet effectively supports both spatial and semantic change understanding within a unified framework, while remaining robust across datasets with different change characteristics and annotation settings.

Qualitative comparisons in Fig. 2 further show that the visual improvements are consistent with the quantitative results on LEVIR-MCI. Additional qualitative examples, including further results on LEVIR-MCI (Fig. 3) and WHU-CDC (Fig. 4), are provided in Appendix.

## 4.4 Ablation Study

Effect of Joint Training and CoDeR. We first examine whether selective routing provides additional benefits beyond simple joint optimization, as shown in Table 3(a). Compared with single-task training and joint training without routing, the joint model with CoDeR achieves the best overall performance across both detection and captioning metrics. This result suggests that joint learning alone is helpful but not sufficient, and that explicit cross-task routing further improves unified change understanding.

Effect of Routing Directionality. To examine how CoDeR should be applied between the two branches, we compare two single-direction settings with the bidirectional design. As shown in Table 3(b), the bidirectional design achieves the best overall balance across metrics among the three compared settings. This suggests that, once cross-task routing is introduced, mutual information exchange between the two branches is more effective than one-way transfer alone.

Effect of Routing Mask Decomposition. To validate the core design of CoDeR, we ablate the decomposed routing mask and compare it with simplified variants. As shown in Table 4(a), the full formulation consistently performs best, while removing either the target-dependent or sourcedependent term degrades performance. Replacing the decomposition with a single gate also results in weaker performance. These results show that the benefit of CoDeR does not arise merely from gating itself, but from explicitly decomposing the routing condition into receiver-side need for support and sender-side reliability cues.

Effect of Interaction Operators. To verify that the gains of CoDeRNet come from the proposed routing design rather than simply adding cross-task interaction, we compare CoDeR with representative interaction operators in Table 4(b), including co-attention [28, 29], cross-attention [41, 44], channel exchange [40], and gated modulation. Since these methods were originally designed for different tasks and architectures, we implement their core interaction principles within the same CoDeRNet pipeline while keeping all other components fixed. Detailed formulations are provided in Appendix 7.4. As shown in the table, generic interaction operators can improve cross-branch communication, but they do not explicitly determine whether the receiving branch needs support or whether the sending branch provides reliable evidence. In contrast, CoDeR structures cross-task communication as conditional routing under heterogeneous supervision, leading to stronger captioning performance while preserving competitive detection accuracy. These results show that the effectiveness of cross-task communication in CDC depends not only on exchanging information, but also on how the transfer condition is defined.

Table 4: Ablation studies on LEVIR-MCI [25] for analyzing the effectiveness of CoDeR. (a) Routing mask decomposition ablation compares CoDeR with reduced variants and a single-gate design. (b) Interaction operator ablation compares CoDeR with representative cross-task communication operators under the same framework. Bold and underline indicate the best and second-best results in each column, respectively.  
(a) CoDeR mask decomposition  
(b) Interaction operators
<table><tr><td>Mask</td><td>mIoU</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td><td>Interaction</td><td>mIoU</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td><td>Params (M)</td><td>FLOPs (G)</td></tr><tr><td>Sender reliability (Cs)</td><td>87.60</td><td>65.12</td><td>40.49</td><td>75.73</td><td>139.57</td><td>Co-Attn</td><td>87.84</td><td>64.15</td><td>40.64</td><td>75.44</td><td>139.33</td><td>102.8</td><td>67.46</td></tr><tr><td>Receiver need (1 − Ct)</td><td>87.65</td><td>64.65</td><td>40.81</td><td>75.94</td><td>141.02</td><td>Cross-Attn</td><td>87.70</td><td>65.01</td><td>40.60</td><td>75.53</td><td>138.87</td><td>103.2</td><td>67.71</td></tr><tr><td>Single gate</td><td>87.59</td><td>65.46</td><td>40.27</td><td>75.54</td><td>138.19</td><td>Exchange</td><td>87.53</td><td>65.15</td><td>40.61</td><td>75.47</td><td>139.12</td><td>102.9</td><td>67.46</td></tr><tr><td>Decomposed routing ((1 − Ct) · Cs)</td><td>87.79</td><td>66.27</td><td>41.08</td><td>76.36</td><td>141.79</td><td>Gating CoDeR (Ours)</td><td>87.75 87.79</td><td>65.66 66.27</td><td>40.73</td><td>76.09 76.36</td><td>140.86 141.79</td><td>103.1 102.8</td><td>67.46 67.46</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>41.08</td><td></td><td></td><td></td><td></td></tr></table>

Further analyses on CoDeR and other design choices are provided in Appendix Secs. 7.1–7.3, where we examine the effect of routing scale, backbone, and additional architectural and training factors.

## 4.5 Human Evaluation of Caption Grounding and Cross-Output Consistency

Standard captioning metrics measure agreement with reference captions but do not directly establish whether generated descriptions faithfully reflect observable changes or agree with the model’s spatial predictions. We therefore conduct a blind human evaluation using two complementary criteria: factual grounding, which assesses the correctness of a caption with respect to the annotated changes, and cross-output consistency, which assesses agreement between a caption and the change mask predicted by the same model.

We evaluate MCINet, our baseline without CoDeR, and CoDeRNet on 660 change-containing image pairs from the LEVIR-MCI test set. The samples are stratified by change composition (building only, road only, or both) and change area. Seven researchers participated in the blind evaluation, with model identities concealed. Each evaluator assesses all three systems on the same assigned set of 120 image pairs in two separate rounds. In the factual-grounding round, captions are evaluated against the ground-truth change masks, without access to predicted masks. In the cross-output-consistency round, captions are evaluated against the corresponding predicted change masks, while ground-truth masks are withheld. Both criteria are rated on a five-point scale, with higher scores indicating closer agreement between the caption and the mask provided in the corresponding evaluation round. Both criteria are restricted to building- and road-related changes covered by the annotations; mentions of categories outside this scope are treated as neutral. Thirty image pairs are shared among all seven evaluators to assess inter-rater agreement. Each round yields 840 ratings per system and 2,520 rating across the three systems, resulting in 5,040 criterion scores across both rounds.

Table 5 reports the mean and standard deviation of the ratings. CoDeRNet achieves the highest mean scores on both criteria, improving factual grounding and cross-output consistency over MCINet by 0.138 and 0.181, respectively. Both improvements are statistically significant under the sign test, with $p = 0 . 0 4 5$ for factual grounding and $p = 0 . 0 0 0 4$ for cross-output consistency. CoDeRNet also achieves slightly higher mean scores than the baseline without CoDeR on both criteria. On the shared evaluation subset, factual-grounding ratings yield Krippendorff’s $\alpha = 0 . 7 3 9$ , while the reliability of the average ratings across the seven evaluators, measured by ICC(A, 7), reaches 0.974 for factual grounding and 0.892 for cross-output consistency. Together, these results provide evidence beyond reference-based captioning metrics that CoDeRNet improves factual grounding and cross-output consistency over MCINet.

## 5 Discussion

Our findings suggest that CDC highlights a broader challenge in multi-task learning under heterogeneous supervision, where dense prediction and language generation require different forms of representation. Rather than assuming uniform feature sharing, CoDeRNet shows that conditional information routing can provide a more suitable way to coordinate such heterogeneous task branches.

Limitation. Although CoDeRNet improves both tasks within a unified CDC framework, the performance gains are asymmetric. While captioning benefits significantly from detection cues, the reverse impact is more constrained by the semantic gap. This suggests that spatially grounded cues from the detection branch are particularly effective for improving semantic change description, whereas the reverse direction provides a more moderate benefit. In addition, our evaluation is restricted to two bi-temporal remote-sensing CDC benchmarks, and the effectiveness of CoDeRNet in multitemporal scenarios and broader dense-prediction-and-language settings remains to be verified. We also do not explicitly supervise the routing cues with region-language correspondence, which may limit semantic faithfulness in cases where visual changes are spatially subtle or linguistically ambiguous.

Table 5: Blind human evaluation on the change-containing subset of LEVIR-MCI. Values are mean ± standard deviation on a five-point scale, with 840 ratings per system for each criterion across 660 unique image pairs. Higher scores indicate better factual grounding or cross-output consistency. Bold indicates the highest mean in each column.
<table><tr><td>Model</td><td>Factual grounding</td><td>Cross-output consistency</td></tr><tr><td>MCINet [25]</td><td> $3 . 5 2 0 \pm 1 . 3 8$ </td><td> $3 . 6 3 2 \pm 1 . 2 5$ </td></tr><tr><td>Ours (w/o CoDeR)</td><td> $3 . 6 4 5 \pm 1 . 2 2$ </td><td> $3 . 7 8 2 \pm 1 . 0 9$ </td></tr><tr><td>Ours (w/ CoDeR)</td><td> ${ \bf 3 . 6 5 8 \pm 1 . 2 7 }$ </td><td> ${ \bf 3 . 8 1 3 \pm 1 . 1 3 }$ </td></tr></table>

Broader Impact. CoDeRNet may support applications such as urban monitoring, disaster assessment, and environmental analysis. However, incorrect change localization or captioning could mislead downstream decisions, so human verification is recommended in high-stakes applications.

Future Work. A promising direction for future work is to strengthen semantic correspondence between the two branches more explicitly. For example, region-language alignment or class-aware semantic transfer may help cross-task routing reflect not only spatial relevance but also the semantic structure shared by the two tasks. Such designs may provide a more effective way to translate semantic cues into region-level predictions within CDC.

## 6 Conclusion

In this paper, we revisit change detection and captioning not simply as a conventional joint learning problem, but as a conditional information routing problem under heterogeneous supervision. We proposed CoDeRNet, a unified framework that preserves task-specific representations while enabling selective bidirectional exchange of complementary information. At its core, Confidence-Decomposed Routing decomposes the routing condition into receiver-side need for support and sender-side reliability cues, allowing cross-task information to be transferred only when it is expected to be beneficial. Experiments on LEVIR-MCI and WHU-CDC show that CoDeRNet improves semantic description quality while preserving strong localization accuracy. Through systematic analyses, we further show that the gains arise not merely from generic feature interaction, but from explicitly modeling selective cross-task communication. The key contribution of CoDeR is therefore not the introduction of another interaction operator, but the reformulation of CDC communication as a conditional routing problem under heterogeneous supervision. This emphasizes the need to model task-specific reliability and support requirements when coordinating branches with different supervision signals. We believe this perspective extends beyond remote sensing CDC to broader multi task and multimodal learning settings involving heterogeneous supervision and task coordination.

## Acknowledgments

This work was supported in part by InSpace Company Ltd. (Inspace-Kyung Hee Artificial Intelligence Research Institute), by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00334321), by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (No. IITP-2026-RS-2023-00258649) supervised by the IITP (Institute for Information and Communications Technology Planning and Evaluation), and by Global - Learning & Academic research institution for Master’s·PhD students, and Postdocs (G-LAMP) Program of the National Research Foundation of Korea (NRF) grant funded by the Ministry of Education(No. RS-2025-25442355). The authors declare no competing interests.

## References

[1] Wele Gedara Chaminda Bandara and Vishal M Patel. A transformer-based siamese network for change detection. In IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium, pages 207–210. IEEE, 2022.

[2] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings ofthe acl workshop on intrinsic and extrinsic evaluation measuresfor machine translation and/or summarization, pages 65–72, 2005.

[3] Bingyi Cao, Koert Chen, Kevis-Kokitsi Maninis, Kaifeng Chen, Arjun Karpur, Ye Xia, Sahil Dua, Tanmaya Dabral, Guangxing Han, Bohyung Han, et al. Tipsv2: Advancing vision-language pretraining with enhanced patch-text alignment. arXiv preprint arXiv:2604.12012, 2026.

[4] Shizhen Chang and Pedram Ghamisi. Changes to captions: An attentive network for remote sensing change captioning. IEEE Transactions on Image Processing, 32:6047–6060, 2023.

[5] Hao Chen, Wenyuan Li, and Zhenwei Shi. Adversarial instance augmentation for building change detection in remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 60:1–16, 2021.

[6] Hao Chen, Zipeng Qi, and Zhenwei Shi. Remote sensing image change detection with transformers. IEEE Transactions on Geoscience and Remote Sensing, 60:1–14, 2021.

[7] Hongruixuan Chen, Jian Song, Chengxi Han, Junshi Xia, and Naoto Yokoya. Changemamba: Remote sensing change detection with spatiotemporal state space model. IEEE Transactions on Geoscience and Remote Sensing, 62:1–20, 2024.

[8] Andrea Codegoni, Gabriele Lombardi, and Alessandro Ferrari. Tinycd: A (not so) deep learning model for change detection. Neural Computing and Applications, 35(11):8471–8486, 2023.

[9] Rodrigo Caye Daudt, Bertr Le Saux, and Alexandre Boulch. Fully convolutional siamese networks for change detection. In 2018 25th IEEE international conference on image processing (ICIP), pages 4063–4067. IEEE, 2018.

[10] Mingyu Ding, Bin Xiao, Noel Codella, Ping Luo, Jingdong Wang, and Lu Yuan. Davit: Dual attention vision transformers. In European conference on computer vision, pages 74–92. Springer, 2022.

[11] Sheng Fang, Kaiyu Li, Jinyuan Shao, and Zhe Li. Snunet-cd: A densely connected siamese network for change detection of vhr images. IEEE Geoscience and Remote Sensing Letters, 19:1–5, 2021.

[12] Sheng Fang, Kaiyu Li, and Zhe Li. Changer: Feature interaction is what you need for change detection. IEEE Transactions on Geoscience and Remote Sensing, 61:1–11, 2023.

[13] Chengxi Han, Chen Wu, Haonan Guo, Meiqi Hu, and Hongruixuan Chen. Hanet: A hierarchical attention network for change detection with bitemporal very-high-resolution remote sensing images. IEEE Journal ofSelected Topics in Applied Earth Observations and Remote Sensing, 16:3867–3878, 2023.

[14] Xin Huang, Liangpei Zhang, and Tingting Zhu. Building change detection from multitemporal highresolution remotely sensed images based on a morphological building index. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 7(1):105–115, 2013.

[15] Shunping Ji, Shiqing Wei, and Meng Lu. Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set. IEEE Transactions on geoscience and remote sensing, 57(1):574–586, 2018.

[16] Robert E Kennedy, Philip A Townsend, John E Gross, Warren B Cohen, Paul Bolstad, YQ Wang, and Phyllis Adams. Remote sensing change detection tools for natural resource managers: Understanding concepts and tradeoffs in the design of landscape monitoring projects. Remote sensing ofenvironment, 113 (7):1382–1396, 2009.

[17] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[18] Zhenglai Li, Chang Tang, Lizhe Wang, and Albert Y Zomaya. Remote sensing change detection via temporal feature interaction and guided refinement. IEEE Transactions on Geoscience and Remote Sensing, 60:1–11, 2022.

[19] Zhenglai Li, Chang Tang, Xinwang Liu, Wei Zhang, Jie Dou, Lizhe Wang, and Albert Y Zomaya. Lightweight remote sensing change detection with progressive feature aggregation and supervised attention. IEEE Transactions on Geoscience and Remote Sensing, 61:1–12, 2023.

[20] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004.

[21] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 2117–2125, 2017.

[22] Chenyang Liu, Rui Zhao, Hao Chen, Zhengxia Zou, and Zhenwei Shi. Remote sensing image change captioning with dual-branch transformers: A new method and a large scale dataset. IEEE Transactions on Geoscience and Remote Sensing, 60:1–20, 2022.

[23] Chenyang Liu, Jiajun Yang, Zipeng Qi, Zhengxia Zou, and Zhenwei Shi. Progressive scale-aware network for remote sensing image change captioning. In IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium, pages 6668–6671. IEEE, 2023.

[24] Chenyang Liu, Rui Zhao, Jianqi Chen, Zipeng Qi, Zhengxia Zou, and Zhenwei Shi. A decoupling paradigm with prompt learning for remote sensing image change captioning. IEEE Transactions on Geoscience and Remote Sensing, 61:1–18, 2023.

[25] Chenyang Liu, Keyan Chen, Haotian Zhang, Zipeng Qi, Zhengxia Zou, and Zhenwei Shi. Change-agent: Toward interactive comprehensive remote sensing change interpretation and analysis. IEEE Transactions on Geoscience and Remote Sensing, 62:1–16, 2024.

[26] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin transformer v2: Scaling up capacity and resolution. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 12009–12019, 2022.

[27] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022.

[28] Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh. Hierarchical question-image co-attention for visual question answering. Advances in neural information processing systems, 29, 2016.

[29] Duy-Kien Nguyen and Takayuki Okatani. Improved fusion of visual and language representations by dense symmetric co-attention for visual question answering. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 6087–6096, 2018.

[30] Mubashir Noman, Mustansar Fiaz, Hisham Cholakkal, Sanath Narayan, Rao Muhammad Anwer, Salman Khan, and Fahad Shahbaz Khan. Remote sensing change detection with transformers trained from scratch. IEEE Transactions on Geoscience and Remote Sensing, 62:1–14, 2024.

[31] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th annual meeting ofthe Associationfor Computational Linguistics, pages 311–318, 2002.

[32] Dong Huk Park, Trevor Darrell, and Anna Rohrbach. Robust change captioning. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 4624–4633, 2019.

[33] Yue Qiu, Shintaro Yamamoto, Kodai Nakashima, Ryota Suzuki, Kenji Iwata, Hirokatsu Kataoka, and Yutaka Satoh. Describing and localizing multiple changes with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1971–1980, 2021.

[34] Blaž Rolih, Matic Fucka, Filip Wolf, and Luka ˇ Cehovin Zajc. Be the change you want to see: Revisiting <sup>ˇ</sup> remote sensing change detection practices. IEEE Transactions on Geoscience and Remote Sensing, 2025.

[35] Jingye Shi, Mengge Zhang, Yuewu Hou, Ruicong Zhi, and Jiqiang Liu. A multitask network and two large-scale datasets for change detection and captioning in remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 62:1–17, 2024.

[36] Dongwei Sun, Yajie Bao, Junmin Liu, and Xiangyong Cao. A lightweight sparse focus transformer for remote sensing image change captioning. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 17:18727–18738, 2024.

[37] Dongwei Sun, Jing Yao, Wu Xue, Changsheng Zhou, Pedram Ghamisi, and Xiangyong Cao. Mask approximation net: A novel diffusion model approach for remote sensing change captioning. IEEE transactions on geoscience and remote sensing, 2025.

[38] Mingxing Tan and Quoc Le. Efficientnetv2: Smaller models and faster training. In International conference on machine learning, pages 10096–10106. PMLR, 2021.

[39] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566–4575, 2015.

[40] Yikai Wang, Wenbing Huang, Fuchun Sun, Tingyang Xu, Yu Rong, and Junzhou Huang. Deep multimodal fusion by channel exchanging. Advances in neural information processing systems, 33:4835–4845, 2020.

[41] Xi Wei, Tianzhu Zhang, Yan Li, Yongdong Zhang, and Feng Wu. Multi-modality cross attention network for image and sentence matching. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10941–10950, 2020.

[42] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in neural information processing systems, 34:12077–12090, 2021.

[43] Joseph Z Xu, Wenhan Lu, Zebo Li, Pranav Khaitan, and Valeriya Zaytseva. Building damage detection in satellite imagery using convolutional neural networks. arXiv preprint arXiv:1910.06444, 2019.

[44] Yuechen Yu, Yilei Xiong, Weilin Huang, and Matthew R Scott. Deformable siamese attention networks for visual object tracking. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 6728–6737, 2020.

[45] Haotian Zhang, Hao Chen, Chenyao Zhou, Keyan Chen, Chenyang Liu, Zhengxia Zou, and Zhenwei Shi. Bifa: Remote sensing image change detection with bitemporal feature alignment. IEEE Transactions on Geoscience and Remote Sensing, 62:1–17, 2024.

[46] Haotian Zhang, Keyan Chen, Chenyang Liu, Hao Chen, Zhengxia Zou, and Zhenwei Shi. Cdmamba: Incorporating local clues into mamba for remote sensing image binary change detection. IEEE Transactions on Geoscience and Remote Sensing, 2025.

[47] Zhuo Zheng, Ailong Ma, Liangpei Zhang, and Yanfei Zhong. Change is everywhere: Single-temporal supervised object change detection in remote sensing imagery. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 15193–15202, 2021.

[48] Zhuo Zheng, Yanfei Zhong, Ailong Ma, and Liangpei Zhang. Single-temporal supervised learning for universal remote sensing change detection. International Journal ofComputer Vision, 132(12):5582–5602, 2024.

[49] Qing Zhou, Junyu Gao, Yuan Yuan, and Qi Wang. Single-stream extractor network with contrastive pre-training for remote-sensing change captioning. IEEE Transactions on Geoscience and Remote Sensing, 62:1–14, 2024.

## 7 Appendix

## 7.1 Scale Selection for CoDeR

We further examine how the receiving scale affects CoDeR. We first vary the routing scale in each direction while fixing the opposite direction to its default setting, as shown in Table 6. We then compare the default asymmetric single-scale design with multi-scale CoDeR variants in Table 8(a) to test whether broader cross-scale routing provides additional benefit.

Table 6(a) shows that, for cap → seg, applying CoDeR at 1/8 yields the strongest overall performance, suggesting that the detection branch benefits most when caption-derived information is injected at a relatively higher-resolution stage. Table 6(b) shows that, for seg → cap, the best overall performance is obtained at 1/16, indicating that the captioning branch benefits more from interaction at a relatively lower-resolution stage aligned with semantic abstraction.

Table 8(a) further shows that the default asymmetric single-scale design remains the most consistent configuration overall. The fully multi-scale variant degrades all caption metrics, while the directionspecific multi-scale variants do not establish a clear advantage over the default. These results indicate that the effectiveness of CoDeR comes not from broader cross-scale transfer, but from selecting an appropriate routing scale for each direction.

Overall, the results support the asymmetric scale design of CoDeRNet: cap → seg is most effective at a higher-resolution scale, whereas seg → cap is most effective at a lower-resolution scale. More importantly, CoDeR should be applied selectively at an appropriate granularity rather than uniformly across all scales.

## 7.2 Backbone-Level Generalizability of CoDeR

We further examine whether the effect of CoDeR is consistent across different encoder choices, rather than being tied to a specific backbone architecture. To this end, Table 7 compares the baseline without CoDeR and CoDeRNet under the same backbone setting, across diverse backbone families with image-only pretraining (e.g., ConvNeXt [27]) or vision-language pretraining (e.g., TIPSv2 [3]), as well as different model sizes within the ConvNeXt series.

As shown in Table 7(a), enabling CoDeR generally improves captioning performance across all tested backbone families, with consistent gains in BLEU-4, METEOR, ROUGE-L, and CIDEr-D in most cases. Although the effect on mIoU is smaller and not always positive, the overall trend indicates that the proposed routing mechanism is not limited to a particular encoder design. Table 7(b) shows a similar pattern across different ConvNeXt scales. From Tiny to Large, CoDeR consistently improves most captioning metrics, while detection performance is generally maintained and occasionally shows modest improvements. These results suggest that the benefit of CoDeR is broadly preserved across backbone families and model scales, supporting the general applicability of the proposed routing design.

## 7.3 Additional Ablation Studies

Effect of region-token pooling size. Table 8(b) studies the pooling size used to construct region-level tokens in the caption branch. Compared with using no region-level tokens, introducing pooled region tokens generally improves both detection and captioning performance, confirming the usefulness of intermediate-granularity representations. Among the tested settings, a pooling size of 2 gives the best overall performance on most metrics. Although a pooling size of 4 remains competitive, it does not surpass the overall performance of size 2, suggesting that an appropriate region granularity is more effective than overly coarse pooling.

Effect of augmentation type. Table 9(a) analyzes the effect of geometric augmentation and directional caption remapping. Applying geometric augmentation alone improves several captioning metrics, but the gains are not consistent across all metrics. In contrast, combining geometric augmentation with caption remapping yields the best overall results for both detection and captioning. This indicates that, when direction-sensitive transformations are applied to the input images, maintaining caption-image consistency through remapping (see Table 10) is important for effective training.

Table 6: Ablation of the CoDeR routing scale on LEVIR-MCI [25]. Each subtable reports performance when same-scale routing is applied at different receiving scales in one direction, while the opposite direction is fixed to its default setting. Bold indicates the best result in each column.  
(a) Cap → seg: routing scale
<table><tr><td>Scale</td><td>mIoU</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td></tr><tr><td>1/4</td><td>87.75</td><td>64.93</td><td>40.62</td><td>75.61</td><td>139.84</td></tr><tr><td>1/8</td><td>87.79</td><td>66.27</td><td>41.08</td><td>76.36</td><td>141.79</td></tr><tr><td>1/16</td><td>87.48</td><td>66.27</td><td>40.49</td><td>76.17</td><td>139.86</td></tr><tr><td>1/32</td><td>87.78</td><td>65.72</td><td>40.71</td><td>75.93</td><td>139.80</td></tr></table>

(b) Seg → cap: routing scale
<table><tr><td>Scale</td><td>mIoU</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td></tr><tr><td>1/8</td><td>87.76</td><td>65.54</td><td>40.38</td><td>75.22</td><td>138.04</td></tr><tr><td>1/16</td><td>87.79</td><td>66.27</td><td>41.08</td><td>76.36</td><td>141.79</td></tr><tr><td>1/32</td><td>87.59</td><td>66.13</td><td>40.23</td><td>75.50</td><td>138.45</td></tr></table>

Table 7: Robustness of CoDeRNet across backbone choices on LEVIR-MCI [25]. (a) Backbonefamily ablation: performance changes with and without CoDeR across different backbones. (b) Backbone-size ablation: performance changes with and without CoDeR across ConvNeXt variants from Tiny to Large. Bold indicates the best result in each column. In the CoDeRNet rows, colored values indicate improvements (gain) or degradations (drop) relative to the corresponding baseline. B4, MET., R-L, and CID. denote BLEU-4, METEOR, ROUGE-L, and CIDEr-D, respectively.  
(a) Backbone family ablation
<table><tr><td>Backbone</td><td>CoDeR</td><td>mIoU</td><td>BLEU-4</td><td>MET.</td><td>R-L</td><td>CID.</td></tr><tr><td>ConvNeXt-B [27]</td><td>X √</td><td>87.67 87.79</td><td>64.41 66.27</td><td>39.93 41.08</td><td>74.98 76.36</td><td>136.18 141.79</td></tr><tr><td>SwinV2-B [26]</td><td>X √</td><td>87.49 87.34</td><td>63.55 64.45</td><td>40.24 40.30</td><td>75.26 75.31</td><td>136.93 138.20</td></tr><tr><td>MiT-B5 [42]</td><td>X √</td><td>87.74 87.78</td><td>63.92 65.38</td><td>40.15 40.54</td><td>75.49 75.54</td><td>137.54 140.28</td></tr><tr><td>DaViT-B [10]</td><td>X √</td><td>87.27 87.37</td><td>63.32 64.59</td><td>39.70 40.34</td><td>74.94 75.24</td><td>136.85 137.03</td></tr><tr><td>EfficientNetV2-M [38]</td><td>× √</td><td>87.05 87.10</td><td>63.31 65.17</td><td>40.31 40.72</td><td>75.66 75.99</td><td>137.97 140.12</td></tr><tr><td>TIPSv2-B/14 [3]</td><td>X r</td><td>86.56 86.73</td><td>64.31 65.23</td><td>40.27 40.91</td><td>75.30 75.74</td><td>138.00 140.86</td></tr></table>

(b) Backbone size ablation
<table><tr><td>Size</td><td>CoDeR</td><td>mIoU</td><td>BLEU-4</td><td>MET.</td><td>R-L</td><td>CID.</td></tr><tr><td>Tiny</td><td>X</td><td>87.05</td><td>65.26</td><td>39.91</td><td>75.47</td><td>134.69</td></tr><tr><td></td><td>√</td><td>87.48</td><td>65.76</td><td>41.02</td><td>76.13</td><td>141.50</td></tr><tr><td>Small</td><td>X</td><td>87.22</td><td>64.29</td><td>40.18</td><td>75.28</td><td>137.32</td></tr><tr><td></td><td>√</td><td>87.32</td><td>65.30</td><td>40.05</td><td>75.33</td><td>138.40</td></tr><tr><td>Base</td><td>X</td><td>87.67</td><td>64.41</td><td>39.93</td><td>74.98</td><td>136.18</td></tr><tr><td></td><td>√</td><td>87.79</td><td>66.27</td><td>41.08</td><td>76.36</td><td>141.79</td></tr><tr><td></td><td>X</td><td>87.81</td><td>64.91</td><td>40.02</td><td>75.36</td><td>138.93</td></tr><tr><td>Large</td><td>√</td><td>87.85</td><td>65.73</td><td>40.65</td><td>76.01</td><td>140.23</td></tr></table>

Effect of change fusion composition. Table 9(b) examines how the composition of the shared change representation affects performance. Starting from the temporal features $\{ t _ { 0 } , t _ { 1 } \}$ , adding the signed change cue $\Delta$ already improves the captioning-related metrics, indicating that directional temporal difference provides useful semantic information. Further adding the magnitude cue |∆| leads to the best overall performance across all reported metrics. These results suggest that the signed and magnitude change cues play complementary roles, and that combining both is more effective than using temporal features alone.

## 7.4 Detailed Formulation of Feature Interaction Operators

This section provides the detailed formulations of the feature interaction operators compared in Table 4(b), which serve as controlled baselines for cross-branch communication. The purpose of this comparison is not to faithfully reproduce the original architectures of prior methods, since they were introduced for different tasks, modalities, feature resolutions, and network designs. Instead, each operator is implemented as an abstraction of its core interaction principle within the same CoDeRNet framework. All variants are inserted at the same cross-task transfer locations and use the same encoder, task decoders, feature dimensions, and training protocol. This controlled setting allows the comparison to focus on the effect of the feature interaction design itself rather than on differences in the surrounding architecture.

Let $F _ { t } , F _ { s } \in \mathbb { R } ^ { B \times C \times H \times W }$ denote the target and source features, respectively. The target feature $F _ { t }$ indicates the branch representation being updated, while the source feature $F _ { s }$ provides complementary information from the other branch. For example, when the segmentation branch receives information from the captioning branch, $F _ { t }$ corresponds to the segmentation feature and $F _ { s }$ corresponds to the captioning feature. The reverse direction is defined analogously. We use $\phi ( \cdot )$ to denote a lightweight source adapter implemented with a $1 \times 1$ convolution, and ⊙ denotes element-wise multiplication. All operators are implemented as lightweight feature interaction blocks with comparable parameter counts and FLOPs.

Table 8: Ablation studies on LEVIR-MCI [25]. (a) Single-scale versus multi-scale CoDeR application. The default design applies CoDeR at one receiving scale per direction (seg ← cap at $1 / 8 ,$ cap ← seg at $1 / 1 6 )$ , while the multi-scale variants apply independent CoDeR blocks at all four scales for the indicated direction(s). (b) Region-token pooling size ablation for caption decoding. Bold indicates the best result in each column.  
(a) Single-scale vs. multi-scale
<table><tr><td>Type</td><td>Setting</td><td>mIoU</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td></tr><tr><td rowspan="3">Multi</td><td>both (all ↔ all)</td><td>87.66</td><td>65.81</td><td>40.18</td><td>75.93</td><td>138.83</td></tr><tr><td>seg only (all ↔ 1/16)</td><td>87.75</td><td>65.44</td><td>41.03</td><td>76.22</td><td>142.18</td></tr><tr><td> $\mathrm { c a p o n i y } \dot { ( 1 / 8  \mathrm { a l l } ) } ^ { \prime }$ </td><td>87.86</td><td>64.56</td><td>40.83</td><td>75.93</td><td>140.32</td></tr><tr><td>Single (Ours)</td><td> $\overline { { ( 1 / 8  1 / 1 6 ) } }$ </td><td>87.79</td><td>66.27</td><td>41.08</td><td>76.36</td><td>141.79</td></tr></table>

(b) Region pooling size
<table><tr><td> $M$ </td><td>mIoU</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td></tr><tr><td>None</td><td>87.56</td><td>65.95</td><td>40.80</td><td>75.99</td><td>139.22</td></tr><tr><td>1</td><td>87.73</td><td>66.27</td><td>40.55</td><td>75.58</td><td>138.88</td></tr><tr><td>2</td><td>87.79</td><td>66.27</td><td>41.08</td><td>76.36</td><td>141.79</td></tr><tr><td>4</td><td>87.77</td><td>65.67</td><td>41.08</td><td>76.22</td><td>141.57</td></tr></table>

Table 9: Additional ablation studies on LEVIR-MCI [25]. (a) Effect of augmentation type, where $G$ denotes geometric augmentation and R denotes directional caption remapping. (b) Effect of change fusion composition, where $F$ is formed from temporal features at $t _ { 0 }$ and $t _ { 1 }$ together with signed and magnitude change cues $\Delta$ and $| \Delta |$ . Bold indicates the best result in each column.

(a) Augmentation type
<table><tr><td>Aug.</td><td>mIoU</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td></tr><tr><td>-</td><td>86.61</td><td>64.51</td><td>39.44</td><td>74.72</td><td>136.06</td></tr><tr><td>G</td><td>86.67</td><td>64.45</td><td>40.64</td><td>75.97</td><td>139.21</td></tr><tr><td>G+R</td><td>87.79</td><td>66.27</td><td>41.08</td><td>76.36</td><td>141.79</td></tr></table>

(b) Change fusion composition
<table><tr><td>F</td><td>mIoU</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr-D</td></tr><tr><td> $\{ t _ { 0 } , t _ { 1 } \}$ </td><td>87.45</td><td>64.21</td><td>40.04</td><td>75.37</td><td>138.76</td></tr><tr><td> $\{ t _ { 0 } , t _ { 1 } , \Delta \}$ </td><td>87.47</td><td>65.69</td><td>40.53</td><td>75.87</td><td>140.86</td></tr><tr><td> $\bar { \{ t _ { 0 } , t _ { 1 } , \Delta , | \Delta | \} }$ </td><td>87.79</td><td>66.27</td><td>41.08</td><td>76.36</td><td>141.79</td></tr></table>

Co-attention-style Interaction. The co-attention-style variant computes a spatial modulation map from the concatenated target and source features:

$$
a = \sigma \left( \mathrm { C o n v } _ { 1 \times 1 } ( [ F _ { t } ; F _ { s } ] ) \right) ,
$$

where $\lceil F _ { t } ; F _ { s } \rceil$ denotes channel-wise concatenation, $\sigma ( \cdot )$ is the sigmoid function, and $a \in$ $[ 0 , 1 ] ^ { B \times 1 \times H \times \dot { W } }$ is a spatial attention map. The target feature is updated by

$$
F _ { t } ^ { \prime } = F _ { t } + a \odot \phi ( F _ { s } ) .
$$

This operator represents a co-attention-style spatial gating mechanism, where the source feature is injected into the target branch according to a spatial weight computed from the joint target-source representation. Since the target and source features are mixed before computing a, the resulting modulation does not explicitly distinguish receiver-side need for support from sender-side reliability.

Cross-attention Interaction. The cross-attention variant uses the target feature as query and the source feature as key and value. After flattening the spatial dimensions, we define

$$
Q = { \mathrm { F l a t t e n } } ( F _ { t } ) , \quad K = { \mathrm { F l a t t e n } } ( F _ { s } ) , \quad V = { \mathrm { F l a t t e n } } ( F _ { s } ) ,
$$

where $Q , K , V \in \mathbb { R } ^ { B \times H W \times C }$ . The target feature is updated as

$$
F _ { t } ^ { \prime } = F _ { t } + { \mathrm { R e s h a p e } } \left( { \mathrm { M H A } } ( Q , K , V ) \right) ,
$$

where MH $\mathrm { A } ( \cdot )$ denotes multi-head attention and Reshape(·) restores the spatial feature shape. This operator captures token-wise target-source relations through attention weights. For a controlled comparison with the other lightweight operators, we use the attention operation as the interaction module without adding a full Transformer block with additional feed-forward layers.

Exchange-style Interaction. The exchange-style variant performs a soft channel-wise interpolation between the target feature and the adapted source feature. It first computes a channel-wise exchange gate:

$$
p = \sigma \left( \mathrm { C o n v } _ { 1 \times 1 } ( \mathrm { G A P } ( F _ { t } ) ) \right) ,
$$

where $\mathrm { G A P } ( \cdot )$ denotes global average pooling and $p \in [ 0 , 1 ] ^ { B \times C \times 1 \times 1 }$ . The target feature is updated by

$$
F _ { t } ^ { \prime } = ( 1 - p ) \odot F _ { t } + p \odot \phi ( F _ { s } ) .
$$

This formulation can be interpreted as an exchange-inspired soft mixing operation: channels with small values of $p$ mainly preserve the target representation, while channels with large values of $p$ receive more information from the source branch. Unlike spatial gating, this operator controls feature transfer at the channel level and uses the same exchange weight across spatial locations. Unlike hard channel replacement strategies, this formulation implements a soft exchange-style interaction within the same comparison framework.

Table 10: Directional caption remapping under geometric augmentation on LEVIR-MCI. To preserve caption–image consistency, direction-sensitive expressions are remapped according to the applied geometric transform. Two-word directional phrases are matched prior to single-word directional levels.
<table><tr><td>Transform</td><td>Category</td><td>Rule / Example</td></tr><tr><td colspan="3">Remapping rules</td></tr><tr><td rowspan="5">H-Flip</td><td>Single-level</td><td>left ↔ right</td></tr><tr><td>Phrase-level</td><td>upper left ↔ upper right</td></tr><tr><td>Phrase-level</td><td>lower left ↔ lower right</td></tr><tr><td>Phrase-level</td><td>top left ↔ top right</td></tr><tr><td>Phrase-level</td><td>bottom left ↔ bottom right</td></tr><tr><td rowspan="4">V-Flip</td><td>Single-level</td><td>top ↔ bottom, upper ↔ lower</td></tr><tr><td>Phrase-level</td><td>top left ↔ bottom left</td></tr><tr><td>Phrase-level</td><td>top right ↔ bottom right</td></tr><tr><td>Phrase-level</td><td>upper left ↔ lower left, upper right ↔ lower right</td></tr><tr><td rowspan="4">90° CCW rot. (×k)</td><td>Single-level</td><td>top → left → bottom → right → top</td></tr><tr><td>Single-level</td><td>upper → left, lower → right</td></tr><tr><td>Phrase-level</td><td>top left → bottom left → bottom right → top right → . . .</td></tr><tr><td>Phrase-level</td><td>upper left → lower left → lower right → upper right → . . </td></tr><tr><td colspan="4">Examples</td></tr><tr><td rowspan="6">Phrase-level</td><td>Original</td><td>a building appeared in the upper left corner of the scene</td></tr><tr><td>H-Flip</td><td>a building appeared in the upper right corner of the scene</td></tr><tr><td>V-Flip</td><td>a building appeared in the lower left corner of the scene</td></tr><tr><td>Rot. 90° CCW</td><td>a building appeared in the lower left corner of the scene</td></tr><tr><td>Rot.  $1 8 0 ^ { \circ }$ </td><td>a building appeared in the lower right corner of the scene</td></tr><tr><td>Rot. 270° CCW</td><td>a building appeared in the upper right corner of the scene</td></tr><tr><td rowspan="3">Single-level</td><td>Original</td><td>the road on the left was removed</td></tr><tr><td>H-Flip</td><td>the road on the right was removed</td></tr><tr><td>Rot. 90° CCW</td><td>the road on the bottom was removed</td></tr></table>

Gating-based Modulation. The gated modulation variant computes a channel-wise gate from the global context of both target and source features:

$$
z = \sigma \left( \mathrm { C o n v } _ { 1 \times 1 } ( \mathrm { G A P } ( [ F _ { t } ; F _ { s } ] ) ) \right) ,
$$

where $z \in [ 0 , 1 ] ^ { B \times C \times 1 \times 1 }$ . The target feature is updated as

$$
F _ { t } ^ { \prime } = F _ { t } + z \odot \phi ( F _ { s } ) .
$$

This operator performs residual source feature injection controlled by a channel-wise gate. Compared with exchange-style interaction, it preserves the original target feature through a residual update rather than interpolating between the target and source features.

Confidence-Decomposed Routing (CoDeR). CoDeR separately computes target-side sufficiency and source-side reliability cues:

$$
C _ { t } = \mathcal { R } _ { t } ( F _ { t } ) , \quad C _ { s } = \mathcal { R } _ { s } ( F _ { s } ) ,
$$

where $\mathcal { R } _ { t }$ and $\mathcal { R } _ { s }$ are lightweight routing estimators implemented with $1 \times 1$ convolution heads followed by sigmoid activation. The confidence-decomposed routing mask is defined as

$$
M = \left( 1 - C _ { t } \right) \odot C _ { s } .
$$

The target feature is then updated by

$$
F _ { t } ^ { \prime } = F _ { t } + M \odot \phi ( F _ { s } ) .
$$

Here, $C _ { t }$ and $C _ { s }$ are learned routing cues rather than externally supervised confidence labels, estimating target-side sufficiency and source-side reliability, respectively. The routing mask M becomes large when the receiving branch requires support and the sending branch provides reliable evidence, encouraging source information to be transferred mainly where it is expected to be complementary. This differs from joint gating mechanisms that compute a single modulation weight from mixed target-source features, as CoDeR explicitly decomposes the transfer condition into receiver-side need and sender-side reliability terms.

Table 11: Additional detection evaluation. (a) Multi-class mIoU and change-class IoU (cIoU) on LEVIR-MCI [25]. Seg and Cap indicate change detection and change captioning training, respectively. cIoU is computed by merging building and road into a single change class. Bold indicates the best result in each metric column. (b) Image-level paired bootstrap analysis. ∆ denotes the full-test-set performance difference between CoDeRNet and MCINet. Differences and 95% confidence intervals are expressed in percentage points. Positive values favor CoDeRNet.  
(a) Change-class IoU evaluation
<table><tr><td>Seg</td><td>Cap</td><td>CoDeR</td><td>mIoU</td><td>cIoU</td></tr><tr><td>√</td><td>X</td><td>X</td><td>87.03</td><td>82.05</td></tr><tr><td>√</td><td>√</td><td>X</td><td>87.67</td><td>82.97</td></tr><tr><td>√</td><td>√</td><td>√</td><td>87.79</td><td>83.07</td></tr></table>

(b) Paired bootstrap analysis
<table><tr><td>Dataset</td><td>Metric</td><td>∆</td><td>95% CI</td></tr><tr><td rowspan="2">LEVIR-MCI</td><td>mIoU</td><td>+1.36</td><td>[+1.07, +1.70]</td></tr><tr><td>cIoU</td><td>+1.79</td><td>[+1.41, +2.17]</td></tr><tr><td>WHU-CDC</td><td>cIoU</td><td>+2.02</td><td>[+0.57, +3.69]</td></tr></table>

## 7.5 Change-Class IoU Evaluation

To complement the multi-class mIoU evaluation on LEVIR-MCI, we additionally report change-class IoU (cIoU), which measures localization performance over the combined change regions. We merge the building and road classes into a single change class in both the predicted and ground-truth masks, and compute cIoU as the ratio of the total intersection to the total union across the test set. Unlike binary mIoU, which averages the IoUs of the background and change classes, cIoU evaluates only the change class. It therefore assesses change localization independently of the distinction between building and road changes.

Table 11(a) compares single-task training, joint training without CoDeR, and joint training with CoDeR. Joint training improves cIoU from 82.05 to 82.97 over the single-task baseline, and adding CoDeR further increases it to 83.07. The same trend holds for mIoU, with CoDeRNet achieving the highest scores on both metrics among the three settings. These results indicate improvements in both multi-class segmentation and localization of the merged change regions, although the incremental detection gains from CoDeR over joint training are modest: 0.12 percentage points in mIoU and 0.10 in cIoU.

## 7.6 Statistical Analysis

To assess the statistical evidence for the detection improvements, we conduct an image-level paired bootstrap analysis comparing CoDeRNet with MCINet [25] on LEVIR-MCI and WHU-CDC. We use one fixed checkpoint per method and evaluate both methods on the same test images under the same evaluation protocol. In each bootstrap replicate, we sample image indices with replacement, drawing as many images as in the original test set, and apply the same sampled indices to both methods. We then pool the pixel-level confusion counts over the sampled images, recompute the dataset-level metrics, and calculate the performance differences between the two methods. We repeat this procedure 10,000 times and obtain 95% confidence intervals using the 2.5th and 97.5th percentiles of the bootstrap distribution of performance differences.

Table 11(b) reports the performance differences computed on the full test sets and their bootstrap confidence intervals. Compared with MCINet, CoDeRNet improves mIoU and cIoU on LEVIR-MCI by 1.36 and 1.79 percentage points, respectively, and cIoU on WHU-CDC by 2.02 percentage points. The 95% confidence intervals lie entirely above zero for all detection metrics reported in the table, supporting the observed improvements on both datasets under image-level resampling.

![](images/7ab1bfede9ef8c30feac63ef842c040f6ead4b37de20181f008cf8efdda751a5.jpg)  
Figure 3: Qualitative comparison on LEVIR-MCI. In the captions, building-related words (e.g., house, building) and road-related words (e.g., road) are highlighted using colors consistent with the corresponding mask classes.

![](images/8e2a704d39071319049e65e6dcfd35eda454ab12c58341225b77b12050a7b9fc.jpg)  
Figure 4: Qualitative comparison on WHU-CDC. In the captions, building-related words (e.g., house, building) are highlighted using colors consistent with the corresponding mask classes.